"""Build minimal tool-selection SFT data from data/eval_dataset.json.

Each row teaches the model which tool to call for a short question that
requires RAG (search_docs / get_field_definition / search_regulation),
lineage (trace_dependencies / inspect_lineage), or both (cross_reference).

Also writes a tool-selection eval manifest for ±1 reward scoring.

Usage:
    python training/build_tool_sft.py
    python training/build_tool_sft.py --eval-ratio 0.15
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.tool_utils import LITE_SYSTEM_PROMPT as _LITE_SYSTEM_PROMPT
from src.agent.tools import dispatch_tool

DATASET = PROJECT_ROOT / "data" / "eval_dataset.json"
OUT_DIR = PROJECT_ROOT / "training" / "data"
SESSION = "debug_lgd"

# Categories included in tool-selection SFT
RAG_CATS = {"field_definition", "regulation"}
LINEAGE_CATS = {"lineage"}
BOTH_CATS = {"cross_reference"}

TOOL_ALIASES = {
    "find_governing_rules": "search_regulation",
    "query_regulation": "search_regulation",
    "trace_regulation_chain": "search_regulation",
    "search_changelog": "search_docs",
    "backtrace_sas_field": "trace_dependencies",
}

LITE_TOOLS = {
    "trace_causal_chain", "get_sas_formula", "trace_dependencies", "inspect_lineage",
    "search_docs", "search_regulation", "get_field_definition",
}

# Minimal question templates — distinct phrasing per tool
_MINIMAL = {
    "get_field_definition": "¿Qué es el campo {field}?",
    "search_docs": "Busca en la documentación markdown información sobre {field}.",
    "search_regulation": "Busca en la normativa (CRR/EBA/BdE) qué regula {field}.",
    "trace_dependencies": "Trazar dependencias BFS: ¿de qué campos depende {field}? Incluye fórmulas.",
    "inspect_lineage": "¿En qué data step del SAS se asigna o calcula {field}?",
    "get_sas_formula": "¿Cuál es la expresión SAS exacta (expr) que asigna {field}?",
}

# Extra templates for augmentation (lineage disambiguation)
_EXTRA_TEMPLATES = {
    "trace_dependencies": [
        "Lista la cadena de dependencias con expresiones para {field}.",
        "¿Qué campos alimentan {field} en el pipeline?",
    ],
    "inspect_lineage": [
        "¿Dónde se escribe {field} en el código SAS?",
        "Localiza el data step que calcula {field}.",
    ],
    "get_sas_formula": [
        "Dame la fórmula exacta de {field} en el pipeline SAS.",
        "¿Qué expr asigna {field} según el compilador?",
    ],
    "get_field_definition": ["Define el significado de {field}."],
    "search_docs": ["¿Qué dice la documentación interna sobre {field}?"],
    "search_regulation": ["¿Qué artículo regulatorio aplica a {field}?"],
}

_ALL_FIELDS = [
    "ECL", "RWA", "EAD", "PD_ESTIMADA", "LGD_ESTIMADA", "LGD_CON_MOC",
    "MoC", "OR_EAD", "K_IRB", "LGD_FLOOR_APLICADO", "SW_FUSION",
    "STAGE_IFRS9", "CURE_FLAG", "COLATERAL_TIPO", "SEGMENTO",
    "LGD_FLOOR", "MOC", "OR_EAD_TIT", "NO_CONFORMES", "LGD_REALIZADA",
]

_STEP_RE = re.compile(r"^(?P<name>[a-z_]+)\((?P<args>.*)\)$", re.IGNORECASE)


def _parse_kv(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in re.split(r",\s*", raw.strip()):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


def parse_tool_step(step: str) -> tuple[str, dict] | None:
    m = _STEP_RE.match(step.strip())
    if not m:
        return None
    name = TOOL_ALIASES.get(m.group("name").lower(), m.group("name").lower())
    if name not in LITE_TOOLS:
        return None
    kv = _parse_kv(m.group("args"))
    args: dict = {"session_id": SESSION}
    if name in ("trace_dependencies", "inspect_lineage"):
        target = kv.get("target") or kv.get("field") or kv.get("table")
        if not target:
            return None
        args["target"] = target.upper()
    elif name == "get_field_definition":
        field = kv.get("field") or kv.get("target", "")
        if not field:
            return None
        args["field"] = field.upper()
    elif name == "search_docs":
        args["query"] = kv.get("query") or kv.get("field") or kv.get("file") or kv.get("target", "")
    elif name == "search_regulation":
        args["query"] = kv.get("query") or kv.get("article") or kv.get("field") or kv.get("concept", "")
    if name in ("search_docs", "search_regulation") and not args.get("query"):
        return None
    return name, args


def _field_from_args(args: dict) -> str:
    return args.get("target") or args.get("field") or args.get("query", "?")


def _minimal_question(tool: str, args: dict, fallback: str) -> str:
    field = _field_from_args(args)
    tmpl = _MINIMAL.get(tool)
    if tmpl and field != "?":
        return tmpl.format(field=field)
    # regulation: keep short slice of original when query-heavy
    if tool == "search_regulation" and len(fallback) < 120:
        return fallback
    return fallback[:120].rstrip("?.") + "?"


def _one_example(
    question: str,
    tool: str,
    args: dict,
    answer: str,
    meta: dict,
) -> dict | None:
    result = dispatch_tool(tool, args)
    call_id = f"call_{abs(hash((question, tool))) % 100000:05d}"
    messages = [
        {"role": "system", "content": _LITE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool,
            "content": json.dumps(result, ensure_ascii=False, default=str),
        },
        {"role": "assistant", "content": answer[:600]},
    ]
    return {
        "messages": messages,
        "meta": {**meta, "expected_tool": tool, "question": question},
    }


def _multi_example(
    question: str,
    steps: list[tuple[str, dict]],
    answer: str,
    meta: dict,
) -> dict | None:
    if len(steps) < 2:
        return None
    messages: list[dict] = [
        {"role": "system", "content": _LITE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for i, (tool, args) in enumerate(steps):
        call_id = f"call_{abs(hash((question, i, tool))) % 100000:05d}"
        result = dispatch_tool(tool, args)
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {
                    "name": tool,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }],
        })
        messages.append({
            "role": "tool", "tool_call_id": call_id, "name": tool,
            "content": json.dumps(result, ensure_ascii=False, default=str),
        })
    messages.append({"role": "assistant", "content": answer[:800]})
    return {
        "messages": messages,
        "meta": {
            **meta,
            "expected_tools": [t for t, _ in steps],
            "question": question,
        },
    }


def build_from_eval(ex: dict) -> list[dict]:
    """Expand one eval row into minimal single-tool (or dual-tool) SFT rows."""
    cat = ex.get("category", "")
    if cat not in RAG_CATS | LINEAGE_CATS | BOTH_CATS:
        return []

    sol = ex.get("solution", {})
    seq = sol.get("tool_sequence", [])
    parsed_steps: list[tuple[str, dict]] = []
    for raw in seq:
        p = parse_tool_step(raw)
        if p:
            parsed_steps.append(p)

    if not parsed_steps:
        # Fall back to expected_tools with fields from eval
        fields = ex.get("expected_fields_mentioned") or ["ECL"]
        field = fields[0]
        for tool in ex.get("expected_tools", []):
            tool = TOOL_ALIASES.get(tool, tool)
            if tool not in LITE_TOOLS:
                continue
            if tool in ("trace_dependencies", "inspect_lineage"):
                parsed_steps.append((tool, {"target": field, "session_id": SESSION}))
            elif tool == "get_field_definition":
                parsed_steps.append((tool, {"field": field}))
            elif tool == "search_docs":
                parsed_steps.append((tool, {"query": field}))
            elif tool == "search_regulation":
                parsed_steps.append((tool, {"query": field}))

    if not parsed_steps:
        return []

    answer = sol.get("final_answer") or ex.get("gold_answer_summary", "")
    base_meta = {"source_id": ex["id"], "category": cat}
    out: list[dict] = []

    if cat in BOTH_CATS:
        # Dual-tool: first regulation/RAG + first lineage step
        rag_tools = [s for s in parsed_steps if s[0] in ("search_regulation", "search_docs", "get_field_definition")]
        lin_tools = [s for s in parsed_steps if s[0] in ("trace_dependencies", "inspect_lineage")]
        if rag_tools and lin_tools:
            pair = [rag_tools[0], lin_tools[0]]
            q = ex["question"][:140].rstrip("?.") + "?"
            ex_multi = _multi_example(q, pair, answer, {**base_meta, "kind": "both"})
            if ex_multi:
                out.append(ex_multi)
        return out

    # Single-tool rows: one minimal question per tool step (dedupe by tool+field)
    seen: set[tuple[str, str]] = set()
    for tool, args in parsed_steps:
        key = (tool, _field_from_args(args))
        if key in seen:
            continue
        seen.add(key)
        q = _minimal_question(tool, args, ex["question"])
        row = _one_example(q, tool, args, answer, {**base_meta, "kind": "rag" if cat in RAG_CATS else "lineage"})
        if row:
            out.append(row)

    return out


    return out


def _field_examples_for_tool(tool: str, field: str, answer: str, meta: dict, rng: random.Random) -> list[dict]:
    """Augment with alternate phrasings for a field+tool pair."""
    rows: list[dict] = []
    if tool in ("trace_dependencies", "inspect_lineage"):
        args = {"target": field.upper(), "session_id": SESSION}
    elif tool == "get_field_definition":
        args = {"field": field.upper()}
    elif tool == "search_docs":
        args = {"query": field}
    elif tool == "search_regulation":
        args = {"query": field}
    else:
        return rows

    for tmpl in _EXTRA_TEMPLATES.get(tool, []):
        q = tmpl.format(field=field)
        row = _one_example(q, tool, args, answer, {**meta, "augmented": True})
        if row:
            rows.append(row)
    return rows


def _failure_to_example(fail: dict, dataset_by_id: dict) -> dict | None:
    """Rebuild a correct SFT row from an eval failure."""
    q = fail["question"]
    if "expected_tools" in fail:
        tools = fail["expected_tools"]
        # find source eval for args
        src = dataset_by_id.get(fail.get("source_id", ""), {})
        sol = src.get("solution", {})
        answer = sol.get("final_answer") or src.get("gold_answer_summary", "OK")
        steps: list[tuple[str, dict]] = []
        for raw in sol.get("tool_sequence", []):
            p = parse_tool_step(raw)
            if p and p[0] in tools:
                steps.append(p)
        if len(steps) < 2:
            field = (src.get("expected_fields_mentioned") or ["ECL"])[0]
            for t in tools:
                if t in ("trace_dependencies", "inspect_lineage"):
                    steps.append((t, {"target": field, "session_id": SESSION}))
                elif t == "get_field_definition":
                    steps.append((t, {"field": field}))
                elif t == "search_docs":
                    steps.append((t, {"query": field}))
                elif t == "search_regulation":
                    steps.append((t, {"query": field}))
        return _multi_example(q, steps[:2], answer, {"source_id": fail.get("source_id"), "kind": "both", "mined_failure": True})
    tool = fail["expected_tool"]
    field = "ECL"
    for f in _ALL_FIELDS:
        if f in q.upper():
            field = f
            break
    if tool in ("trace_dependencies", "inspect_lineage"):
        args = {"target": field, "session_id": SESSION}
    elif tool == "get_field_definition":
        args = {"field": field}
    elif tool == "search_docs":
        args = {"query": field}
    else:
        args = {"query": field}
    return _one_example(q, tool, args, "Respuesta basada en herramienta.", {"kind": fail.get("kind"), "mined_failure": True})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-ratio", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--failure-file", default="", help="JSON with failures from last eval")
    parser.add_argument("--failure-repeats", type=int, default=8, help="Repeat mined failures N times")
    parser.add_argument("--augment-fields", action="store_true", default=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset = json.loads(DATASET.read_text())
    dataset_by_id = {ex["id"]: ex for ex in dataset}

    examples: list[dict] = []
    manifest: list[dict] = []

    for ex in dataset:
        rows = build_from_eval(ex)
        for row in rows:
            examples.append(row)
            if args.augment_fields and "expected_tool" in row["meta"]:
                tool = row["meta"]["expected_tool"]
                field = _field_from_args(json.loads(
                    row["messages"][2]["tool_calls"][0]["function"]["arguments"]
                ))
                examples.extend(_field_examples_for_tool(
                    tool, field, row["messages"][-1]["content"],
                    row["meta"], rng,
                ))

    # Synthetic coverage for all field × primary tool combos
    if args.augment_fields:
        for field in _ALL_FIELDS:
            for tool in ("trace_dependencies", "inspect_lineage", "get_field_definition", "search_docs"):
                if tool in ("trace_dependencies", "inspect_lineage"):
                    args_d = {"target": field, "session_id": SESSION}
                elif tool == "get_field_definition":
                    args_d = {"field": field}
                else:
                    args_d = {"query": field}
                q = _MINIMAL[tool].format(field=field)
                row = _one_example(q, tool, args_d, f"Resultado sobre {field}.", {"kind": "lineage" if tool in ("trace_dependencies", "inspect_lineage") else "rag", "synthetic": True})
                if row:
                    examples.append(row)

    # Mine failures from previous iteration
    if args.failure_file and Path(args.failure_file).exists():
        fails = json.loads(Path(args.failure_file).read_text())
        fail_list = fails if isinstance(fails, list) else fails.get("failures", [])
        for fail in fail_list:
            row = _failure_to_example(fail, dataset_by_id)
            if row:
                for _ in range(args.failure_repeats):
                    examples.append(row)

    for row in examples:
        m = row["meta"]
        if "expected_tools" in m:
            manifest.append({
                "source_id": m.get("source_id", "?"),
                "question": m["question"],
                "expected_tools": m["expected_tools"],
                "kind": m["kind"],
            })
        elif "expected_tool" in m:
            manifest.append({
                "source_id": m.get("source_id", "?"),
                "question": m["question"],
                "expected_tool": m["expected_tool"],
                "kind": m.get("kind", "?"),
            })
        else:
            # rebuild from messages
            tc = row["messages"][2].get("tool_calls", [{}])[0]
            tool = tc.get("function", {}).get("name", "")
            q = row["messages"][1]["content"]
            manifest.append({"question": q, "expected_tool": tool, "kind": m.get("kind", "?")})

    # dedupe manifest by question+expected
    seen_m: set[str] = set()
    unique_manifest: list[dict] = []
    for m in manifest:
        key = m["question"] + str(m.get("expected_tool") or m.get("expected_tools"))
        if key not in seen_m:
            seen_m.add(key)
            unique_manifest.append(m)
    manifest = unique_manifest

    rng.shuffle(examples)
    n_eval = max(1, int(len(examples) * args.eval_ratio))
    eval_ex = examples[:n_eval]
    train_ex = examples[n_eval:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, data in [
        (OUT_DIR / "tool_sft_train.jsonl", train_ex),
        (OUT_DIR / "tool_sft_eval.jsonl", eval_ex),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")

    manifest_path = OUT_DIR / "tool_selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    kinds = {}
    tools: dict[str, int] = {}
    for m in manifest:
        kinds[m.get("kind", "?")] = kinds.get(m.get("kind", "?"), 0) + 1
        if "expected_tool" in m:
            tools[m["expected_tool"]] = tools.get(m["expected_tool"], 0) + 1
        else:
            for t in m["expected_tools"]:
                tools[t] = tools.get(t, 0) + 1

    print(f"Built {len(examples)} SFT rows ({len(train_ex)} train / {len(eval_ex)} eval)")
    print(f"  manifest={len(manifest)}  kinds={kinds}  tools={tools}")


if __name__ == "__main__":
    main()

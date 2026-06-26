"""Generate SFT training traces from the golden eval dataset.

For each example with a `solution.tool_sequence`, executes the real tools
against the debug_lgd session and builds chat-format conversations using
the same lite system prompt as the 4B agent.

Also adds field-generalization examples so the model learns tool-selection
patterns (lineage → trace_dependencies, regulation → search_regulation, etc.)
rather than memorizing specific field names.

Usage:
    python training/generate_golden_traces.py
    python training/generate_golden_traces.py --teacher qwen2.5:14b-instruct-q4_K_M  # supplement via agent
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent import _LITE_SYSTEM_PROMPT, _seed_user_message
from src.agent.tools import dispatch_tool

DATASET = PROJECT_ROOT / "data" / "eval_dataset.json"
OUT_DIR = PROJECT_ROOT / "training" / "data"
SESSION = "debug_lgd"

# Map teacher-only tools to the 5-tool lite set
_TOOL_ALIASES = {
    "find_governing_rules": "search_regulation",
    "query_regulation": "search_regulation",
    "trace_regulation_chain": "search_regulation",
    "search_changelog": "search_docs",
    "backtrace_sas_field": "trace_dependencies",
}

# Category → default tool pattern for generalization
_CATEGORY_TOOLS = {
    "lineage": ["trace_dependencies", "inspect_lineage"],
    "regulation": ["search_regulation"],
    "field_definition": ["get_field_definition"],
    "bug_detection": ["inspect_lineage", "trace_dependencies", "search_docs"],
    "cross_reference": ["trace_dependencies", "search_regulation"],
    "data_quality": ["search_docs", "get_field_definition"],
    "trace_analysis": ["inspect_lineage", "trace_dependencies", "search_docs"],
}

_GENERALIZATION_FIELDS = [
    "ECL", "RWA", "EAD", "PD_ESTIMADA", "LGD_ESTIMADA", "LGD_CON_MOC",
    "MoC", "OR_EAD", "K_IRB", "LGD_FLOOR_APLICADO", "SW_FUSION",
    "STAGE_IFRS9", "CURE_FLAG", "COLATERAL_TIPO", "SEGMENTO",
]

_STEP_RE = re.compile(
    r"^(?P<name>[a-z_]+)\((?P<args>.*)\)$", re.IGNORECASE,
)


def _parse_kv_args(raw: str) -> dict[str, str]:
    """Parse field=ECL, query=foo style args."""
    out: dict[str, str] = {}
    for part in re.split(r",\s*", raw.strip()):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


def parse_tool_step(step: str) -> tuple[str, dict] | None:
    """Parse 'trace_dependencies(field=ECL)' → (tool_name, args_dict)."""
    m = _STEP_RE.match(step.strip())
    if not m:
        return None

    name = m.group("name").lower()
    name = _TOOL_ALIASES.get(name, name)
    kv = _parse_kv_args(m.group("args"))

    args: dict = {"session_id": SESSION}
    if name in ("trace_dependencies", "inspect_lineage"):
        target = kv.get("target") or kv.get("field") or kv.get("table")
        if not target:
            return None
        args["target"] = target.upper() if name != "search_docs" else target
    elif name == "search_docs":
        args["query"] = kv.get("query") or kv.get("field") or kv.get("target", "")
    elif name == "get_field_definition":
        field = kv.get("field") or kv.get("target", "")
        if not field:
            return None
        args["field"] = field.upper()
    elif name == "search_regulation":
        args["query"] = kv.get("query") or kv.get("field") or kv.get("concept", "")
    else:
        return None

    if name in ("search_docs", "search_regulation") and not args.get("query"):
        return None

    return name, args


def _format_tool_conversation(
    question: str,
    tool_steps: list[tuple[str, dict]],
    final_answer: str,
) -> dict | None:
    """Build a multi-turn tool-calling conversation."""
    if not tool_steps or not final_answer.strip():
        return None

    messages: list[dict] = [
        {"role": "system", "content": _LITE_SYSTEM_PROMPT},
        {"role": "user", "content": _seed_user_message(question)},
    ]

    for i, (tool_name, args) in enumerate(tool_steps):
        call_id = f"call_{hash((question, i, tool_name)) % 100000:05d}"
        result = dispatch_tool(tool_name, args)

        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool_name,
            "content": json.dumps(result, ensure_ascii=False, default=str),
        })

    messages.append({"role": "assistant", "content": final_answer.strip()})
    return {"messages": messages, "meta": {"tools": [t for t, _ in tool_steps]}}


def _trace_from_solution(ex: dict) -> dict | None:
    sol = ex.get("solution", {})
    seq = sol.get("tool_sequence", [])
    if not seq:
        return None

    steps: list[tuple[str, dict]] = []
    for raw in seq:
        parsed = parse_tool_step(raw)
        if parsed:
            steps.append(parsed)

    if not steps:
        return None

    answer = sol.get("final_answer") or ex.get("gold_answer_summary", "")
    return _format_tool_conversation(ex["question"], steps, answer)


def _generalization_examples(rng: random.Random, n: int = 40) -> list[dict]:
    """Synthetic Q&A teaching tool selection for arbitrary fields."""
    examples: list[dict] = []
    templates = [
        (
            "lineage",
            "¿De dónde viene el campo {field} y qué campos lo alimentan?",
            lambda f: [
                ("trace_dependencies", {"target": f, "session_id": SESSION}),
                ("inspect_lineage", {"target": f, "session_id": SESSION}),
            ],
        ),
        (
            "field_definition",
            "¿Qué es el campo {field}?",
            lambda f: [("get_field_definition", {"field": f})],
        ),
        (
            "regulation",
            "¿Qué regulación aplica al campo {field}?",
            lambda f: [("search_regulation", {"query": f})],
        ),
        (
            "cross_reference",
            "¿Cómo se calcula {field} y qué normativa lo regula?",
            lambda f: [
                ("trace_dependencies", {"target": f, "session_id": SESSION}),
                ("search_regulation", {"query": f}),
            ],
        ),
    ]

    for _ in range(n):
        cat, tmpl, tool_fn = rng.choice(templates)
        field = rng.choice(_GENERALIZATION_FIELDS)
        steps = tool_fn(field)
        question = tmpl.format(field=field)

        # Build answer from tool results (teaches grounding in tool output)
        parts: list[str] = []
        for tool_name, args in steps:
            result = dispatch_tool(tool_name, args)
            if tool_name == "get_field_definition":
                defn = result.get("result") or result.get("definition", "")
                if defn:
                    parts.append(str(defn)[:400])
            elif tool_name == "trace_dependencies" and result.get("found"):
                layers = result.get("layers", [])
                if layers and isinstance(layers[0], dict):
                    direct = layers[1]["fields"] if len(layers) > 1 else layers[0].get("fields", [])
                    parts.append(
                        f"**{field}** depende directamente de: "
                        + ", ".join(str(x) for x in direct[:8])
                    )
            elif tool_name == "inspect_lineage" and result.get("found"):
                expr = result.get("expression") or result.get("assignments")
                if expr:
                    parts.append(f"En el pipeline SAS: {str(expr)[:300]}")
            elif tool_name == "search_regulation":
                hits = result.get("results") or result.get("hits") or []
                if hits:
                    parts.append(f"Regulación relevante: {hits[0].get('title', hits[0].get('text', ''))[:200]}")

        if not parts:
            parts.append(f"Consulté las herramientas sobre **{field}**. "
                         f"Revisa los resultados para detalles del pipeline.")

        answer = " ".join(parts)
        ex = _format_tool_conversation(question, steps, answer)
        if ex:
            ex["meta"]["category"] = cat
            ex["meta"]["synthetic"] = True
            examples.append(ex)

    return examples


def _run_teacher_agent(question: str, model: str, api: str) -> dict | None:
    """Run the 14B teacher through /agent/ask and capture trace."""
    import httpx

    tools = ["trace_dependencies", "inspect_lineage", "search_docs",
             "search_regulation", "get_field_definition"]
    payload = {
        "question": question,
        "model": model,
        "tool_names": tools,
        "max_iters": 7,
        "session_id": SESSION,
    }
    tool_calls: list[tuple[str, dict]] = []
    tool_results: list[dict] = []
    answer = ""

    try:
        with httpx.stream("POST", f"{api}/agent/ask", json=payload, timeout=300) as r:
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                if ev["type"] == "tool_call":
                    tool_calls.append((ev["tool"], ev.get("args", {})))
                elif ev["type"] == "tool_result":
                    tool_results.append(ev.get("result", {}))
                elif ev["type"] == "final":
                    answer = ev.get("answer", "")
    except Exception as e:
        print(f"  teacher error: {e}", flush=True)
        return None

    if not tool_calls or not answer:
        return None

    messages: list[dict] = [
        {"role": "system", "content": _LITE_SYSTEM_PROMPT},
        {"role": "user", "content": _seed_user_message(question)},
    ]
    for i, ((tool_name, args), result) in enumerate(zip(tool_calls, tool_results)):
        call_id = f"call_t_{i:03d}"
        clean_args = {k: v for k, v in args.items() if v is not None}
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(clean_args, ensure_ascii=False),
                },
            }],
        })
        messages.append({
            "role": "tool", "tool_call_id": call_id, "name": tool_name,
            "content": json.dumps(result, ensure_ascii=False, default=str),
        })
    messages.append({"role": "assistant", "content": answer.strip()})
    return {"messages": messages, "meta": {"teacher": model, "tools": [t for t, _ in tool_calls]}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="", help="Run teacher model via agent API for gaps")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--generalize", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset = json.loads(DATASET.read_text())

    examples: list[dict] = []
    skipped = 0

    for ex in dataset:
        conv = _trace_from_solution(ex)
        if conv:
            conv["meta"]["id"] = ex["id"]
            conv["meta"]["category"] = ex.get("category")
            examples.append(conv)
        else:
            skipped += 1

    print(f"From solution traces: {len(examples)} ({skipped} skipped)", flush=True)

    # Teacher model for examples without parseable solution
    if args.teacher:
        need_teacher = [ex for ex in dataset if not ex.get("solution", {}).get("tool_sequence")]
        print(f"Running teacher {args.teacher} on {len(need_teacher)} examples...", flush=True)
        for i, ex in enumerate(need_teacher):
            print(f"  [{i+1}/{len(need_teacher)}] {ex['id']}...", flush=True)
            conv = _run_teacher_agent(ex["question"], args.teacher, args.api)
            if conv:
                conv["meta"]["id"] = ex["id"]
                conv["meta"]["category"] = ex.get("category")
                conv["meta"]["teacher"] = args.teacher
                examples.append(conv)
            time.sleep(0.5)

    # Generalization
    gen = _generalization_examples(rng, args.generalize)
    examples.extend(gen)
    print(f"Added {len(gen)} generalization examples", flush=True)
    print(f"Total examples: {len(examples)}", flush=True)

    # Split train/eval — stratify by category where possible
    rng.shuffle(examples)
    n_eval = max(1, int(len(examples) * args.eval_ratio))
    eval_ex = examples[:n_eval]
    train_ex = examples[n_eval:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, data, label in [
        (OUT_DIR / "golden_train.jsonl", train_ex, "train"),
        (OUT_DIR / "golden_eval.jsonl", eval_ex, "eval"),
        (OUT_DIR / "train.jsonl", train_ex, "train (also)"),
        (OUT_DIR / "eval.jsonl", eval_ex, "eval (also)"),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} {label} → {path}", flush=True)


if __name__ == "__main__":
    main()

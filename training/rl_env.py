"""RL environment for SAS debugging with simulated tool calls.

Wraps BugGenerator + SASLogicTree evaluator to create a training loop
where the agent receives buggy SAS code, calls tools (simulated), and
outputs a diagnosis that is scored by functional verification.

Key design: tool calls are simulated from the AST — no external system
needed. trace_dependencies returns lineage, inspect_lineage returns code
location. This keeps training self-contained and fast.

Phase B additions:
- search_experience: simulated experience KB from data/experience/*.md
- Efficiency reward: fewer tool calls for same correctness
- Experience-usage reward: bonus for querying experience KB when relevant
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sas_logic_tree import SASLogicTree, AnyNode
from training.bug_generator import BugGenerator, MutatedBug, make_toy_generator
from training.curriculum import StageConfig, STAGES


# ── Simulated tool execution ──────────────────────────────────────────

_tree = SASLogicTree()


def _sim_trace_dependencies(code: str, target: str, max_depth: int = 3) -> dict:
    """Simulate trace_dependencies tool call on SAS code."""
    try:
        nodes = _tree.parse(code)
        result = _tree.trace_lineage(nodes, target.upper(), max_depth=max_depth)
        return {
            "tool": "trace_dependencies",
            "target": target,
            "found": result.get("found", False),
            "ancestors": result.get("ancestors", []),
            "direct_predecessors": result.get("direct_predecessors", []),
            "layers": result.get("layers", 0),
        }
    except Exception as e:
        return {"tool": "trace_dependencies", "error": str(e)}


def _sim_inspect_lineage(code: str, target: str) -> dict:
    """Simulate inspect_lineage tool call — returns where a field is computed."""
    try:
        nodes = _tree.parse(code)
        lineage = _tree.lineage(nodes)
        # Find all edges that write to target
        target_u = target.upper()
        writers = []
        for edge in lineage.edges:
            if edge["target"] == target_u:
                writers.append({
                    "source": edge["source"],
                    "kind": edge["kind"],
                    "data_step": edge["data_step"],
                    "expr": edge["expr"],
                })
        # Find which data steps define the target
        steps = []
        for n in lineage.nodes:
            if n["id"] == target_u:
                steps = n.get("data_steps", [])
                break
        return {
            "tool": "inspect_lineage",
            "target": target,
            "defined_in_steps": steps,
            "incoming_edges": writers[:10],
        }
    except Exception as e:
        return {"tool": "inspect_lineage", "error": str(e)}


def _sim_search_docs(query: str) -> dict:
    """Simulate search_docs — returns relevant field definitions."""
    # Simplified: return schema info based on field names in query
    field_defs = {
        "LGD_ESTIMADA": "Loss Given Default estimada por el modelo interno. Sujeta a suelos regulatorios (CRR Art. 161).",
        "EAD": "Exposure At Default. Importe total de la exposición en el momento del impago.",
        "ECL": "Expected Credit Loss = PD × LGD × EAD. Provisión IFRS 9.",
        "MoC": "Margin of Conservatism. Margen de prudencia sobre LGD (típicamente 5%).",
        "PD_ESTIMADA": "Probability of Default estimada por el modelo PD.",
        "DPDS": "Days Past Due. Días de mora del contrato.",
        "STAGE_IFRS9": "Clasificación IFRS 9: Stage 1 (performing), Stage 2 (SICR), Stage 3 (default).",
        "COLATERAL_TIPO": "Tipo de garantía: HIPOTECA, PERSONAL, NINGUNA, PIGNORATICIA.",
        "SEGMENTO": "Segmento regulatorio: CORP (corporativo), RETAIL (minorista).",
        "CURE_FLAG": "Indicador de curación: 1 si el contrato se curó en el ciclo.",
        "OR_EAD": "Original EAD. EAD original antes de ajustes.",
        "LGD_CON_MOC": "LGD incluyendo Margin of Conservatism = LGD + MoC.",
    }
    results = []
    for k, v in field_defs.items():
        if k.lower() in query.lower() or any(w in query.upper() for w in k.split('_')):
            results.append({"field": k, "definition": v})
    if not results:
        results.append({"note": "No matching documentation found for query."})
    return {"tool": "search_docs", "query": query, "results": results[:5]}


def _sim_search_regulation(query: str) -> dict:
    """Simulate search_regulation — returns regulatory references."""
    regs = [
        {"article": "CRR Art. 161(1)(b)", "content": "Suelo LGD corporativa no garantizada: 45%", "keywords": ["LGD", "CORP", "45", "floor", "suelo"]},
        {"article": "CRR Art. 154(3)", "content": "Suelo LGD hipotecaria: 30% (vivienda)", "keywords": ["LGD", "HIPOTECA", "30", "floor", "suelo"]},
        {"article": "CRR Art. 160(1)", "content": "Suelo PD: 0.03% para exposiciones no defaultadas", "keywords": ["PD", "0.03", "floor", "suelo"]},
        {"article": "IFRS 9 B5.5.12", "content": "Backstop 30 DPD: presunción refutable de SICR con ≥30 días de mora", "keywords": ["DPDS", "30", "stage", "SICR", "backstop"]},
        {"article": "Circular 4/2017 BdE Anejo 9", "content": "Provisiones por riesgo de crédito: ECL = PD × LGD × EAD", "keywords": ["ECL", "provision", "PD", "LGD", "EAD"]},
    ]
    results = []
    q_upper = query.upper()
    for reg in regs:
        if any(kw.upper() in q_upper for kw in reg["keywords"]):
            results.append({"article": reg["article"], "content": reg["content"]})
    return {"tool": "search_regulation", "query": query, "results": results[:3]}


# ── Experience KB simulation (Phase B) ────────────────────────────────

_EXPERIENCE_DIR = PROJECT_ROOT / "data" / "experience"
_EXPERIENCE_CACHE: list[dict] | None = None


def _load_experience_kb() -> list[dict]:
    """Load experience markdown files into a searchable list."""
    global _EXPERIENCE_CACHE
    if _EXPERIENCE_CACHE is not None:
        return _EXPERIENCE_CACHE

    entries = []
    if not _EXPERIENCE_DIR.exists():
        _EXPERIENCE_CACHE = []
        return entries

    for f in _EXPERIENCE_DIR.glob("*.md"):
        if f.name == "README.md":
            continue
        text = f.read_text(encoding="utf-8")
        # Parse YAML frontmatter
        meta: dict = {}
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                import yaml
                try:
                    meta = yaml.safe_load(parts[1]) or {}
                except Exception:
                    pass
                text = parts[2].strip()

        entries.append({
            "id": meta.get("id", f.stem),
            "type": meta.get("type", "unknown"),
            "priority": meta.get("priority", 0.5),
            "tags": meta.get("tags", []),
            "fields": meta.get("fields", []),
            "content": text[:500],  # truncate for prompt
            "full_content": text,
        })

    _EXPERIENCE_CACHE = entries
    return entries


def _sim_search_experience(query: str, k: int = 3) -> dict:
    """Simulate search_experience — returns past bug/insight experiences."""
    entries = _load_experience_kb()
    q_lower = query.lower()
    q_words = set(re.findall(r'\w+', q_lower))

    scored = []
    for entry in entries:
        # Score by tag/field/content overlap
        score = 0.0
        tag_words = set(t.lower() for t in entry["tags"])
        field_words = set(f.lower() for f in entry["fields"])
        score += len(q_words & tag_words) * 2.0
        score += len(q_words & field_words) * 3.0
        content_lower = entry["content"].lower()
        score += sum(1 for w in q_words if w in content_lower) * 0.5
        score += entry["priority"]
        if score > 0.5:
            scored.append((score, entry))

    scored.sort(key=lambda x: -x[0])
    results = []
    for _, entry in scored[:k]:
        results.append({
            "id": entry["id"],
            "type": entry["type"],
            "priority": entry["priority"],
            "tags": entry["tags"],
            "fields": entry["fields"],
            "content": entry["content"],
        })

    return {"tool": "search_experience", "query": query, "results": results}


# ── SQLite table query simulation ────────────────────────────────────

import sqlite3

def _materialize_tables(
    code: str,
    input_rows: list[dict],
) -> dict[str, list[dict]]:
    """Evaluate SAS code on input rows, capturing per-DATA-step table snapshots.

    Runs each DATA step sequentially per row, snapshotting env after each
    step to produce one table per output dataset. When the code contains
    PROC SQL steps, additionally runs the full ``SASCompiler`` so the
    PROC-produced tables are queryable too (otherwise the model would be
    told tables exist that are missing from the simulated DB).
    """
    from src.sas_logic_tree import DataStepNode, ProcNode
    from src.sas_compiler import SASCompiler

    tree = SASLogicTree()
    nodes = tree.parse(code)
    data_steps = [n for n in nodes if isinstance(n, DataStepNode)]
    has_proc_sql = any(isinstance(n, ProcNode) and n.kind == "SQL" for n in nodes)

    tables: dict[str, list[dict]] = {}
    for row in input_rows:
        env = dict(row)
        for step in data_steps:
            # Evaluate just this one DATA step, passing current env forward
            step_trace = tree.evaluate([step], dict(env))
            env = step_trace.final
            tbl_name = step.output_dataset.replace("work.", "").upper()
            tables.setdefault(tbl_name, []).append(dict(env))

    # PROC SQL steps are invisible to the single-row DATA-step evaluator above
    # (the evaluator treats ProcNode as a no-op). Materialize them via the
    # full compiler, seeded with the first SET dataset so joins/aggregations
    # have rows to work with.
    if has_proc_sql:
        first_set = ""
        for n in data_steps:
            if n.input_dataset:
                first_set = n.input_dataset
                break
        if first_set:
            try:
                comp = SASCompiler()
                out = comp.run(nodes, initial_data={first_set: [dict(r) for r in input_rows]})
                for name, rows in out.items():
                    tables[name.replace("work.", "").upper()] = rows
            except Exception as e:
                logger.warning("PROC SQL materialization failed: %s", e)

    return tables


def _build_sqlite_db(tables: dict[str, list[dict]]) -> sqlite3.Connection:
    """Load materialized tables into an in-memory sqlite database."""
    conn = sqlite3.connect(":memory:")
    for tbl_name, rows in tables.items():
        if not rows:
            continue
        cols = list(rows[0].keys())
        col_defs = []
        for col in cols:
            sample = rows[0][col]
            if isinstance(sample, (int, bool)):
                col_defs.append(f'"{col}" INTEGER')
            elif isinstance(sample, float):
                col_defs.append(f'"{col}" REAL')
            else:
                col_defs.append(f'"{col}" TEXT')
        ddl = f'CREATE TABLE "{tbl_name}" ({", ".join(col_defs)})'
        conn.execute(ddl)
        placeholders = ", ".join(["?"] * len(cols))
        for row in rows:
            vals = [row.get(c) for c in cols]
            conn.execute(f'INSERT INTO "{tbl_name}" VALUES ({placeholders})', vals)
    conn.commit()
    return conn


# Cache: (code_hash, rows_hash) → sqlite connection
_SQLITE_CACHE: dict[str, sqlite3.Connection] = {}
_SQLITE_QUERY_TIMEOUT_S = 3.0
_SQLITE_MAX_ROWS = 50


def _get_sqlite_db(code: str, input_rows: list[dict]) -> sqlite3.Connection:
    """Get or create a cached sqlite DB for the given SAS code + rows.

    Cache key is a stable SHA-256 (not Python ``hash()``, which is salted
    per process and collision-prone — a collision silently returns the wrong
    tables and corrupts the simulated tool's answers).
    """
    payload = json.dumps({"code": code, "rows": input_rows}, sort_keys=True, default=str)
    cache_key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if cache_key not in _SQLITE_CACHE:
        tables = _materialize_tables(code, input_rows)
        _SQLITE_CACHE[cache_key] = _build_sqlite_db(tables)
        # Keep cache small
        if len(_SQLITE_CACHE) > 20:
            oldest = next(iter(_SQLITE_CACHE))
            _SQLITE_CACHE[oldest].close()
            del _SQLITE_CACHE[oldest]
    return _SQLITE_CACHE[cache_key]


class _QueryBudget:
    """sqlite progress handler that aborts a query once the time budget is up."""

    def __init__(self, seconds: float) -> None:
        self._deadline = time.time() + seconds

    def __call__(self) -> int:
        return 1 if time.time() > self._deadline else 0


def _sim_query_table(
    code: str,
    sql: str,
    input_rows: list[dict] | None = None,
) -> dict:
    """Simulate query_table tool — run SQL against materialized SAS tables.

    Hardened: SELECT-only, LIMIT injected when absent, a wall-clock budget
    via sqlite's progress handler (so a pathological join cannot hang the
    reward step and freeze GRPO), and a row cap.
    """
    if input_rows is None:
        from training.bug_generator import TOY_INPUT_ROWS
        input_rows = TOY_INPUT_ROWS

    try:
        conn = _get_sqlite_db(code, input_rows)
        sql_stripped = sql.strip()
        if not sql_stripped.upper().startswith("SELECT"):
            return {"tool": "query_table", "error": "Only SELECT queries allowed."}
        # Inject a LIMIT if the user didn't supply one (bound result size)
        if not re.search(r"\bLIMIT\b", sql_stripped, re.IGNORECASE):
            sql_stripped = sql_stripped.rstrip(";").rstrip() + f" LIMIT {_SQLITE_MAX_ROWS}"
        # Bound query cost — aborts with OperationalError if it runs too long
        conn.set_progress_handler(_QueryBudget(_SQLITE_QUERY_TIMEOUT_S), 10000)
        try:
            cursor = conn.execute(sql_stripped)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = [dict(zip(columns, r)) for r in cursor.fetchmany(_SQLITE_MAX_ROWS)]
        finally:
            conn.set_progress_handler(None, 0)
        return {
            "tool": "query_table",
            "sql": sql,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
        }
    except sqlite3.OperationalError as e:
        return {"tool": "query_table", "sql": sql, "error": f"query aborted: {e}"}
    except Exception as e:
        return {"tool": "query_table", "sql": sql, "error": str(e)}


SIMULATED_TOOLS = {
    "trace_dependencies": _sim_trace_dependencies,
    "inspect_lineage": _sim_inspect_lineage,
    "search_docs": _sim_search_docs,
    "search_regulation": _sim_search_regulation,
    "search_experience": _sim_search_experience,
    "query_table": _sim_query_table,
}


# ── Tool call extraction from model output ────────────────────────────

def _analysis_text(completion: str) -> str:
    """Text outside tool-call tags — excludes echoed tool arguments."""
    text = re.sub(r"<tool_call>\s*\{.*?\}\s*</tool_call>", "", completion, flags=re.DOTALL)
    text = re.sub(r"<\|im_start\|>user\s*<tool_response>.*?</tool_response>", "", text, flags=re.DOTALL)
    return text


def _step_match(text_lower: str, step_name: str) -> float:
    """Return 1.0 exact, 0.5 basename, 0.0 no match."""
    step = step_name.lower()
    if step in text_lower or step.replace("work.", "") in text_lower:
        return 1.0
    base = step.split(".")[-1]
    if base and len(base) > 3 and re.search(rf"\b{re.escape(base)}\b", text_lower):
        return 0.5
    return 0.0


def _truncate_sas_for_prompt(code: str, step_name: str, max_lines: int = 45) -> str:
    """Keep the mutated DATA step visible instead of truncating from line 0."""
    lines = code.split("\n")
    filtered: list[str] = []
    in_datalines = False
    for line in lines:
        if "DATALINES" in line.upper():
            in_datalines = True
            filtered.append("    /* ... datos de entrada omitidos ... */")
            continue
        if in_datalines and line.strip() == ";":
            in_datalines = False
            continue
        if not in_datalines:
            filtered.append(line)

    step_key = step_name.lower().replace("work.", "")
    start = 0
    for i, line in enumerate(filtered):
        if re.search(rf"\bDATA\s+{re.escape(step_name)}\b", line, re.I):
            start = max(0, i - 2)
            break
        if step_key and step_key in line.lower() and "data" in line.lower():
            start = max(0, i - 2)
            break

    window = filtered[start : start + max_lines]
    if start > 0:
        window = ["/* ... código anterior omitido ... */", *window]
    if start + max_lines < len(filtered):
        window.append("/* ... código posterior omitido ... */")
    return "\n".join(window)


def _iter_json_objects(text: str):
    """Yield every balanced ``{ ... }`` substring parsed as JSON."""
    for m in re.finditer(r"\{", text):
        start = m.start()
        depth = 0
        in_str = False
        esc = False
        quote = ""
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == quote:
                    in_str = False
            else:
                if ch in ("'", '"'):
                    in_str, esc, quote = True, False, ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            yield json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            pass
                        break


def _norm_tool_calls(calls: list[dict]) -> list[dict]:
    """Dedupe by tool name (keep first), coerce shape, drop empty names."""
    seen: set[str] = set()
    out: list[dict] = []
    for c in calls:
        name = c.get("name") or c.get("tool") or ""
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        if name in seen:
            continue
        seen.add(name)
        args = c.get("arguments") if isinstance(c.get("arguments"), dict) else {}
        out.append({"name": name, "arguments": args})
    return out


def extract_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from model completion text.

    Supports formats:
    1. Ollama/OpenAI function calling: <tool_call>{"name":..., "arguments":...}</tool_call>
    2. Bare JSON objects with a "name" key (brace-balanced, handles nested args)
    3. Call-syntax fallback: ``tool(...)`` or fenced blocks (tightened — a bare
       mention of a tool name in prose no longer counts as a call, which used
       to inflate the reward for free).
    """
    calls: list[dict] = []

    # Format 1: <tool_call> tags
    for m in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                calls.append(obj)
        except json.JSONDecodeError:
            pass
    if calls:
        return _norm_tool_calls(calls)

    # Format 2: any balanced JSON object carrying a "name" key
    for obj in _iter_json_objects(text):
        if isinstance(obj, dict) and "name" in obj:
            args = obj.get("arguments", obj.get("parameters", {}))
            calls.append({"name": obj["name"],
                          "arguments": args if isinstance(args, dict) else {}})
    if calls:
        return _norm_tool_calls(calls)

    # Format 3: tightened call-syntax fallback (tool(...) or fenced block)
    for tool in SIMULATED_TOOLS:
        if re.search(rf'\b{re.escape(tool)}\s*\(', text) or \
           re.search(rf'```[^\n]*\b{re.escape(tool)}\b', text):
            target_m = re.search(rf'{re.escape(tool)}\s*\(\s*["\']?(\w+)', text)
            target = target_m.group(1) if target_m else ""
            calls.append({"name": tool, "arguments": {"target": target}})
    return _norm_tool_calls(calls)


def _trace_targets_from_calls(tool_calls: list[dict]) -> list[str]:
    from training.gold_trace import LINEAGE_TOOLS
    targets: list[str] = []
    for c in tool_calls:
        name = c.get("name", "")
        if name not in LINEAGE_TOOLS:
            continue
        args = c.get("arguments") if isinstance(c.get("arguments"), dict) else {}
        t = (args.get("target") or args.get("field") or "").upper()
        if t:
            targets.append(t)
    return targets


def _trace_path_score(
    completion: str,
    bug: MutatedBug,
    tool_calls: list[dict],
    analysis: str,
) -> float:
    """Score how well the model followed the gold symptom → root trace."""
    gt = bug.gold_trace
    if not gt:
        from training.gold_trace import compute_gold_trace
        gt = compute_gold_trace(bug)
        bug.gold_trace = gt

    path_fields = [f.upper() for f in gt.get("path_fields", [])]
    path_steps = gt.get("path_steps", [])
    min_hops = gt.get("min_hops", max(1, bug.depth + 1))
    required_prefix = path_fields[: min(len(path_fields), min_hops + 1)]

    trace_targets = _trace_targets_from_calls(tool_calls)
    field_score = 0.0
    if required_prefix and trace_targets:
        hits = 0
        cursor = 0
        for t in trace_targets:
            for j in range(cursor, len(required_prefix)):
                if t == required_prefix[j] or t in required_prefix:
                    hits += 1
                    cursor = j + 1
                    break
        field_score = min(1.0, hits / len(required_prefix))
    elif trace_targets and path_fields:
        overlap = len(set(trace_targets) & set(path_fields))
        field_score = min(1.0, overlap / max(len(path_fields[:3]), 1))

    text_lower = analysis.lower()
    step_score = 0.0
    if path_steps:
        step_hits = sum(1 for s in path_steps if _step_match(text_lower, s) >= 0.5)
        step_score = min(1.0, step_hits / len(path_steps))

    root_score = 0.0
    if _step_match(text_lower, gt.get("root_step", "")) >= 0.5:
        root_score += 0.5
    root_var = gt.get("root_var", "").lower()
    if root_var and root_var not in ("if_condition", "where_filter") and root_var in text_lower:
        root_score += 0.5

    tool_names = [c.get("name", "") for c in tool_calls]
    query_bonus = 0.15 if bug.depth >= 1 and "query_table" in tool_names else 0.0

    return min(1.0, 0.45 * field_score + 0.30 * step_score + 0.25 * root_score + query_bonus)


# ── Reward computation ─────────────────────────────────────────────────

@dataclass
class RewardBreakdown:
    """Detailed reward breakdown for analysis."""
    total: float
    components: dict[str, float]
    tool_calls_found: list[str]
    correct_step_found: bool
    correct_var_found: bool
    fix_restores_values: bool


def compute_reward(
    completion: str,
    bug: MutatedBug,
    stage: StageConfig,
) -> RewardBreakdown:
    """Score a model completion against a mutated bug.

    Reward components depend on the curriculum stage:
    - tool_call: did the model attempt any tool call?
    - correct_tool: did it call trace_dependencies or inspect_lineage?
    - correct_step: did it name the right DATA step?
    - correct_var: did it name the right variable/expression?
    - propagation_trace: did it trace through multiple steps?
    - trace_path: did tool calls follow the gold BFS path symptom → root?
    - no_hallucination: absence of hallucination markers
    - functional_fix: does the proposed fix restore correct values?
    """
    weights = stage.reward_weights
    components: dict[str, float] = {}
    analysis = _analysis_text(completion)
    text_lower = analysis.lower()
    full_lower = completion.lower()

    # 1. Tool call detection
    tool_calls = extract_tool_calls(completion)
    tool_names = [c.get("name", "") for c in tool_calls]

    if "tool_call" in weights:
        components["tool_call"] = 1.0 if tool_calls else 0.0

    # 2. Correct tool selection
    if "correct_tool" in weights:
        if bug.expected_tools:
            expected = set(bug.expected_tools)
            called = set(tool_names)
            # Also accept tool name mentions in completion text
            for t in expected:
                if t in completion:
                    called.add(t)
            components["correct_tool"] = 1.0 if expected & called else 0.0
            # Penalize clearly wrong tool families for pure tool tasks
            if bug.source == "tool_task" and called and not (expected & called):
                components["correct_tool"] = -0.5
        else:
            lineage_tools = {"trace_dependencies", "inspect_lineage", "query_table"}
            has_lineage = bool(lineage_tools & set(tool_names))
            if not has_lineage:
                has_lineage = any(t in completion for t in lineage_tools)
            components["correct_tool"] = 1.0 if has_lineage else 0.0

    # 3. Correct step identification (analysis text only — not tool args,
    #    and NOT hallucinated "data_step" fields in made-up tool JSON)
    if "correct_step" in weights:
        score = _step_match(text_lower, bug.mutation.step_name)
        if score < 1.0:
            paso_m = re.search(r"paso\s*:\s*(\S+)", text_lower)
            if paso_m:
                score = max(score, _step_match(paso_m.group(1), bug.mutation.step_name))
        components["correct_step"] = score

    # 4. Correct variable identification (analysis text only).
    #    De-gamed: the previous version gave full credit whenever ANY of
    #    changed_fields[:2] appeared — but those (ECL, LGD_ESTIMADA, …) are
    #    generic and appear in almost any reasonable answer. Now the model
    #    must name the SPECIFIC target variable or distinctive tokens of the
    #    original expression (excluding the generic changed fields).
    if "correct_var" in weights:
        target = bug.mutation.target_var.lower()
        generic = {f.lower() for f in bug.changed_fields[:2]}
        score = 0.0
        if target and target not in ("if_condition", "where_filter") and target in text_lower:
            score = 1.0
        if score < 1.0:
            var_m = re.search(r"variable\s*:\s*(.+)", text_lower)
            if var_m:
                if target and target in var_m.group(1):
                    score = 1.0
                elif any(f.lower() in var_m.group(1) for f in bug.changed_fields if f.lower() not in generic):
                    score = max(score, 0.5)
        if score < 1.0:
            orig_tokens = {p for p in re.findall(r"\b\w+\b", bug.mutation.original_expr.lower())
                           if len(p) > 2 and p not in generic}
            if orig_tokens:
                hits = sum(1 for p in orig_tokens if p in text_lower)
                score = max(score, min(hits / max(len(orig_tokens), 1), 1.0))
        components["correct_var"] = score

    # 5. Propagation trace (legacy keyword proxy)
    if "propagation_trace" in weights:
        step_names = _bug_step_names(bug)
        mentions = sum(1 for s in step_names if _step_match(text_lower, s) >= 0.5)
        if mentions < 2:
            lineage_markers = sum(1 for kw in (
                "propaga", "depende de", "linaje", "upstream", "propagation",
                "anterior", "input", "antecesor",
            ) if kw in text_lower)
            mentions = max(mentions, min(lineage_markers, 2))
        components["propagation_trace"] = min(mentions / 2.0, 1.0)

    # 5b. Gold trace path (depth-wise debugging reward)
    if "trace_path" in weights:
        components["trace_path"] = _trace_path_score(completion, bug, tool_calls, analysis)

    # 6. Hallucination check
    if "no_hallucination" in weights:
        penalty = 0.0
        # Fabricated precise values not from context
        if re.search(r'(?:he encontrado|puedo ver|según mi análisis).*?\d+\.\d{4,}', completion, re.I):
            penalty += 0.4
        # Claiming to have executed queries (only penalize if NOT using query_table)
        if "query_table" not in tool_names and re.search(r'(?:ejecut[ée]|corr[ií]|lance)\s+(?:la|el|una)\s+(?:consulta|query|proc)', completion, re.I):
            penalty += 0.3
        components["no_hallucination"] = max(0.0, 1.0 - penalty)

    # 7. Functional fix verification
    if "functional_fix" in weights:
        fix_score = _check_functional_fix(completion, bug)
        components["functional_fix"] = fix_score

    # 8. Efficiency (Phase B): fewer tool calls for same correctness
    if "efficiency" in weights:
        n_calls = len(tool_calls)
        # Optimal: 2 calls (trace + inspect). Penalize >4.
        if n_calls == 0:
            components["efficiency"] = 0.0
        elif n_calls <= 2:
            components["efficiency"] = 1.0
        elif n_calls <= 4:
            components["efficiency"] = 0.6
        else:
            components["efficiency"] = max(0.0, 1.0 - (n_calls - 2) * 0.15)

    # 9. Experience usage (Phase B): bonus for querying experience KB
    if "experience_used" in weights:
        used_exp = "search_experience" in tool_names or "search_experience" in completion
        # Check if experience content influenced the answer
        if used_exp:
            # Extra credit if the experience is relevant to the bug type
            exp_tags = set(t.lower() for t in (bug.mutation.mutation_type.split("_") + bug.changed_fields[:2]))
            mentioned = sum(1 for t in exp_tags if t in text_lower)
            components["experience_used"] = min(1.0, 0.5 + mentioned * 0.25)
        else:
            components["experience_used"] = 0.0

    # Weighted sum
    total = 0.0
    for key, weight in weights.items():
        total += weight * components.get(key, 0.0)

    # Shift to [-1, 1] for GRPO
    total = 2.0 * total - 1.0

    return RewardBreakdown(
        total=round(total, 4),
        components={k: round(v, 4) for k, v in components.items()},
        tool_calls_found=tool_names,
        correct_step_found=components.get("correct_step", 0) > 0.5,
        correct_var_found=components.get("correct_var", 0) > 0.5,
        fix_restores_values=components.get("functional_fix", 0) > 0.5,
    )


# ── Functional fix verification (REAL — the evaluator is the judge) ─────

_STEP_NAME_CACHE: dict[str, list[str]] = {}


def _bug_step_names(bug: MutatedBug) -> list[str]:
    """DATA-step + PROC-SQL output names present in the bug's code (cached).

    Used to generalize the propagation-trace reward so it isn't hardcoded to
    the toy_lgd step names.
    """
    code = bug.mutated_code
    if not code:
        return []
    key = hashlib.sha256(code.encode("utf-8")).hexdigest()
    cached = _STEP_NAME_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from src.sas_logic_tree import DataStepNode, ProcNode
        tree = SASLogicTree()
        nodes = tree.parse(code)
        names = [n.output_dataset for n in nodes if isinstance(n, DataStepNode)]
        names += [n.output_table for n in nodes
                  if isinstance(n, ProcNode) and n.output_table]
        names = [n for n in names if n]
    except Exception:
        names = []
    if len(_STEP_NAME_CACHE) > 500:
        _STEP_NAME_CACHE.clear()
    _STEP_NAME_CACHE[key] = names
    return names


def _extract_fix_candidates(completion: str) -> list[str]:
    """Pull proposed-fix expression(s) from the completion text."""
    candidates: list[str] = []
    inline_patterns = [
        r'(?:correcci[oó]n|fix)\s*:\s*([^\n;]+)',
        r'(?:deber[ií]a ser|should be|corregir a)\s*:?\s*[`"\']?([^\n;]+)',
        # Require a colon after "correcto" so "el valor correcto es 0.123"
        # (a hallucinated value, not a code fix) doesn't yield a phantom.
        r'correcto\s*:\s*[`"\']?([^\n;]+)',
        r'(?:cambiar|replace|sustituir)\s+.*?\s+(?:por|with|→)\s+[`"\']?([^\n;]+)',
    ]
    for pat in inline_patterns:
        for m in re.finditer(pat, completion, re.IGNORECASE):
            # Strip wrapping quotes and any leading colon/space so the candidate
            # is a clean SAS expression (e.g. ": DPDS >= 30" -> "DPDS >= 30").
            cand = m.group(1).strip().lstrip(":` '\"").strip()
            if cand:
                candidates.append(cand)
    # Fenced code blocks — split into statements
    for m in re.finditer(r'```(?:sas)?\s*(.*?)```', completion, re.DOTALL | re.IGNORECASE):
        for stmt in m.group(1).split(";"):
            stmt = stmt.strip()
            if "=" in stmt or re.search(r'\b(?:IF|WHERE|AND|OR)\b', stmt, re.IGNORECASE):
                candidates.append(stmt)
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        k = c.lower()
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _is_specific(expr: str) -> bool:
    """Safe to single-replace globally only if distinctive (not e.g. "2")."""
    s = expr.strip()
    if len(s) < 4:
        return False
    return bool(re.search(r"[A-Za-z_]", s))


def _substitute_expr(
    code: str,
    mutated_expr: str,
    candidate: str,
    target_var: str | None = None,
) -> str | None:
    """Replace the mutated expression with the candidate, scoped to its
    statement when possible.

    A naive ``code.replace(mutated_expr, candidate, 1)`` is dangerous when the
    mutated expression is a tiny literal (e.g. ``"2"``) — the first occurrence
    in the file may be unrelated. So for assignment mutations we anchor on
    ``<target_var> = <mutated_expr>``; only distinctive expressions are ever
    replaced globally.
    """
    if not mutated_expr:
        return None
    cand = candidate.strip()
    # 1) Scoped assignment replacement: "var = mutated_expr" -> "var = candidate"
    if target_var and target_var not in ("IF_condition", "WHERE_filter"):
        lhs = re.escape(target_var)
        rhs = re.escape(" ".join(mutated_expr.split())).replace(r"\ ", r"\s+")
        m = re.search(rf"\b{lhs}\s*=\s*{rhs}", code, re.IGNORECASE)
        if m:
            return code[:m.start()] + f"{target_var} = {cand}" + code[m.end():]
    # 2) Direct substring — only when the expression is distinctive enough
    if _is_specific(mutated_expr):
        if mutated_expr in code:
            return code.replace(mutated_expr, cand, 1)
        norm = " ".join(mutated_expr.split())
        pattern = re.escape(norm).replace(r"\ ", r"\s+")
        m = re.search(pattern, code, re.IGNORECASE)
        if m:
            return code[:m.start()] + cand + code[m.end():]
    return None


def _values_close(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a == b
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        try:
            if math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6):
                return True
        except (TypeError, ValueError):
            pass
        return float(a) == float(b)
    return str(a) == str(b)


def _compare_values(fixed: dict, correct: dict, changed: list[str]) -> float:
    if not changed:
        return 0.0
    numeric = [f for f in changed
               if isinstance(correct.get(f), (int, float)) and not isinstance(correct.get(f), bool)]
    checks = numeric or changed
    matched = sum(1 for f in checks if _values_close(fixed.get(f), correct.get(f)))
    ratio = matched / len(checks)
    if ratio >= 0.999:
        return 1.0
    if ratio >= 0.75:
        return 0.7
    if ratio > 0.0:
        return 0.4
    return 0.0


def _text_match_floor(candidate: str, orig: str) -> float:
    """Conservative text floor — only used when execution is impossible.

    Only meaningful for distinctive expressions: a 1–2 char ``orig`` (e.g.
    ``"1"``) would spuriously substring-match almost anything.
    """
    if not orig or not _is_specific(orig):
        return 0.0
    cl = candidate.lower()
    ol = orig.lower()
    if ol in cl:
        return 0.7
    orig_parts = {p for p in re.findall(r'\b\w+\b', ol) if len(p) > 2}
    cand_parts = set(re.findall(r'\b\w+\b', cl))
    if orig_parts and len(orig_parts & cand_parts) / len(orig_parts) > 0.7:
        return 0.5
    return 0.0


def _score_fix_by_execution(
    mutated_code: str,
    mutated_expr: str,
    candidate: str,
    input_row: dict,
    correct: dict,
    changed: list[str],
    target_var: str = "",
) -> float | None:
    """Apply candidate fix → re-evaluate → score vs correct baseline.

    Returns None when the substitution can't be performed (caller falls back
    to text matching). The evaluator caps DO loops at 1000 iterations, and
    every step is wrapped so a malformed proposal never crashes the reward.
    """
    cand = candidate.strip().rstrip(";").strip()
    if not cand:
        return None
    fixed_code = _substitute_expr(mutated_code, mutated_expr, cand, target_var)
    if fixed_code is None or fixed_code == mutated_code:
        return None
    try:
        tree = SASLogicTree()
        fixed_nodes = tree.parse(fixed_code)
        fixed_eval = tree.evaluate(fixed_nodes, dict(input_row)).final
    except Exception:
        return None
    return _compare_values(fixed_eval, correct, changed)


def _check_functional_fix(completion: str, bug: MutatedBug) -> float:
    """Score whether the model's proposed fix restores the correct values.

    TRULY functional: applies each candidate fix to the buggy code, runs it
    through the SAS evaluator, and compares the output to the known-correct
    baseline (evaluator is the judge — no keyword gaming). Falls back to a
    conservative text-match floor only when the fix can't be applied
    mechanically (PROC SQL truncation, malformed proposals). Fully sandboxed:
    any failure contributes 0 rather than aborting the GRPO run.
    """
    orig = bug.mutation.original_expr.strip()
    mutated_expr = bug.mutation.mutated_expr.strip()
    input_row = getattr(bug, "input_row", {}) or {}
    correct = bug.correct_values

    candidates = _extract_fix_candidates(completion)
    if not candidates:
        if _is_specific(orig) and orig.lower() in completion.lower():
            return 0.4
        return 0.0

    # Execution only works when the mutated expr is a real, non-truncated
    # token (PROC wrong_agg truncates original/mutated to 120 chars).
    can_execute = bool(
        mutated_expr and len(mutated_expr) < 120
        and bug.mutated_code and input_row and correct
    )

    best = 0.0
    for cand in candidates:
        score: float | None = None
        if can_execute:
            score = _score_fix_by_execution(
                bug.mutated_code, mutated_expr, cand, input_row, correct,
                bug.changed_fields, bug.mutation.target_var,
            )
        if score is None:
            score = _text_match_floor(cand, orig)
        if score > best:
            best = score

    if _is_specific(orig) and orig.lower() in completion.lower():
        best = max(best, 0.4)

    return min(best, 1.0)


# ── Prompt construction ────────────────────────────────────────────────

def _symptom_user_message(bug: MutatedBug, *, include_code: bool) -> str:
    """Symptom-first prompt — expands search space at depth ≥ 1."""
    gt = bug.gold_trace or {}
    field = gt.get("symptom_field") or (bug.changed_fields[0] if bug.changed_fields else "ECL")
    cv = bug.correct_values.get(field)
    bv = bug.buggy_values.get(field)
    ciclo = bug.input_row.get("CICLO_ID", "")
    header = f"Ciclo **{ciclo}**: " if ciclo else ""

    body = (
        f"{header}El campo `{field}` tiene valor `{bv}` (esperado `{cv}`).\n"
        f"Campos afectados: {', '.join(f'`{f}`' for f in bug.changed_fields[:5])}.\n\n"
        "Investiga la **causa raíz** con `trace_dependencies` e `inspect_lineage`"
    )
    if bug.depth >= 1:
        body += " y `query_table` (tablas intermedias: CYCLES, FLOORED, ECL, FINAL)"
    body += (
        ".\n\nResponde en español con:\n"
        "- PASO: <data step>\n"
        "- VARIABLE: <campo o condición>\n"
        "- CAUSA: <explicación>\n"
        "- CORRECCIÓN: <expresión correcta si la identificas>"
    )
    if include_code:
        code = _truncate_sas_for_prompt(bug.mutated_code, bug.mutation.step_name)
        body = (
            "Analiza el pipeline SAS LGD. Hay un bug con resultados incorrectos.\n\n"
            f"```sas\n{code}\n```\n\n"
            + body
        )
    return body


def build_prompt(bug: MutatedBug, stage: StageConfig) -> list[dict]:
    """Build chat messages for a bug investigation prompt."""
    # Stages 3+ include search_experience (multi_depth onward)
    is_phase_b = stage.name in ("multi_depth", "experience_accelerated", "experience_efficiency")

    experience_tool = ""
    if is_phase_b:
        experience_tool = (
            "- **search_experience(query)**: Buscar experiencias pasadas de validación. "
            "Devuelve bugs conocidos, patrones de error, y soluciones previas.\n"
        )

    system = (
        "Asistente de validación regulatoria bancaria IRB. "
        "DEBES llamar herramientas antes de responder.\n\n"
        "Herramientas: trace_dependencies(target, max_depth), inspect_lineage(target), "
        "search_docs(query), search_regulation(query), "
        "query_table(sql)"
    )
    if is_phase_b:
        system += ", search_experience(query)"
    system += (
        ".\n\nReglas: Llama herramientas primero. "
        "Usa trace_dependencies para dependencias, inspect_lineage para ubicación en código. "
        "Usa query_table para inspeccionar datos intermedios (tablas: CYCLES, FLOORED, ECL, FINAL). "
        "Responde en español."
    )
    if is_phase_b:
        system += " Consulta search_experience PRIMERO. Sé eficiente."

    if stage.name == "tool_selection" and bug.source == "tool_task":
        user = bug.mutation.description
    elif stage.name in ("single_depth", "multi_depth", "experience_accelerated", "experience_efficiency"):
        if bug.source in ("procedural", "curated_level") and not bug.gold_trace:
            from training.gold_trace import attach_gold_trace
            attach_gold_trace(bug)
        hide_code = bug.depth >= 1 or stage.name == "multi_depth"
        user = _symptom_user_message(bug, include_code=not hide_code)
    elif stage.name == "tool_selection":
        changed = bug.changed_fields[:3]
        example_field = changed[0] if changed else "ECL"
        correct_val = bug.correct_values.get(example_field)
        buggy_val = bug.buggy_values.get(example_field)
        user = (
            f"El campo {example_field} tiene valor {buggy_val} pero esperaba {correct_val}. "
            f"¿De qué campos depende {example_field}? Usa las herramientas para investigar."
        )
    else:
        user = _symptom_user_message(bug, include_code=True)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ── Batch generation for GRPO ─────────────────────────────────────────

_CORPUS_CACHE: dict[tuple, Any] = {}


def generate_training_batch(
    n: int,
    stage: StageConfig,
    seed: int = 42,
    correct_code_path: str | Path | None = None,
    use_corpus: bool = True,
    mutation_weights: dict[str, float] | None = None,
) -> list[tuple[MutatedBug, list[dict]]]:
    """Generate n (bug, prompt) pairs for a curriculum stage."""
    import random
    rng = random.Random(seed)

    if stage.name == "tool_selection":
        return _batch_tool_selection(n, stage, rng)

    batch: list[tuple[MutatedBug, list[dict]]] = []
    if use_corpus:
        try:
            from training.sas_corpus import SASCorpus
            cache_key = (seed, stage.max_depth)
            if cache_key not in _CORPUS_CACHE:
                _CORPUS_CACHE[cache_key] = SASCorpus(seed=seed, max_depth=stage.max_depth)
            corpus = _CORPUS_CACHE[cache_key]
            if corpus.generators:
                batch = _batch_from_corpus(n, stage, corpus, mutation_weights)
            # Blend curated levels (~25%)
            n_cur = max(0, min(n // 4, len(corpus.curated_levels)))
            if n_cur:
                curated = corpus.generate_curated(n_cur, seed=seed + 17)
                for lid, bug in curated:
                    if bug.mutation.mutation_type in (stage.mutation_types or [bug.mutation.mutation_type]):
                        batch.append((bug, build_prompt(bug, stage)))
                rng.shuffle(batch)
                batch = batch[:n]
        except Exception as e:
            # A smaller/degenerate batch can cause zero-variance rewards and
            # collapse GRPO silently — surface the cause instead of swallowing.
            logger.warning("generate_training_batch: corpus path failed (%s); "
                           "falling back to toy generator", e)

    if len(batch) < n:
        gen = make_toy_generator(
            seed=seed,
            max_depth=stage.max_depth,
            correct_code_path=correct_code_path,
        )
        extra = _batch_from_generator(n - len(batch), stage, gen, mutation_weights)
        batch.extend(extra)

    return batch[:n]


def _batch_tool_selection(
    n: int,
    stage: StageConfig,
    rng: random.Random,
) -> list[tuple[MutatedBug, list[dict]]]:
    """Stage 0: mix pure tool tasks + light debug bugs + curated levels."""
    from training.tool_selection_tasks import generate_tool_selection_batch

    out: list[tuple[MutatedBug, list[dict]]] = []
    n_tool = max(1, int(n * 0.45))
    n_cur = max(0, int(n * 0.25))
    n_proc = max(0, n - n_tool - n_cur)

    for bug in generate_tool_selection_batch(n_tool, seed=rng.randint(0, 2**31)):
        out.append((bug, build_prompt(bug, stage)))

    try:
        from training.toy_lgd_levels import ToyLgdLevelCatalog
        cat = ToyLgdLevelCatalog()
        for lid, bug in cat.sample(n_cur, seed=rng.randint(0, 2**31)):
            out.append((bug, build_prompt(bug, stage)))
    except Exception as e:
        logger.warning("_batch_tool_selection: curated levels failed (%s)", e)

    if n_proc:
        gen = make_toy_generator(seed=rng.randint(0, 2**31), max_depth=0)
        out.extend(_batch_from_generator(n_proc, stage, gen, None))

    while len(out) < n:
        bug = generate_tool_selection_batch(1, seed=rng.randint(0, 2**31))[0]
        out.append((bug, build_prompt(bug, stage)))

    rng.shuffle(out)
    return out[:n]


def _bug_sort_key(bug: MutatedBug, weights: dict[str, float] | None) -> float:
    if not weights:
        return 0.0
    return weights.get(bug.mutation.mutation_type, 0.1)


def _batch_from_corpus(
    n: int,
    stage: StageConfig,
    corpus,  # SASCorpus
    mutation_weights: dict[str, float] | None = None,
) -> list[tuple[MutatedBug, list[dict]]]:
    """Generate bugs from multi-pipeline corpus."""
    results = []
    raw = corpus.generate_batch(n * 4, balanced=True)
    raw.sort(key=lambda x: -_bug_sort_key(x[1], mutation_weights))
    for name, bug in raw:
        if len(results) >= n:
            break
        if bug.depth < stage.min_depth:
            continue
        if stage.mutation_types and bug.mutation.mutation_type not in stage.mutation_types:
            continue
        prompt = build_prompt(bug, stage)
        results.append((bug, prompt))
    return results


def _batch_from_generator(
    n: int,
    stage: StageConfig,
    gen: BugGenerator,
    mutation_weights: dict[str, float] | None = None,
) -> list[tuple[MutatedBug, list[dict]]]:
    """Generate bugs from a single BugGenerator."""
    candidates: list[tuple[MutatedBug, list[dict]]] = []
    attempts = 0
    while len(candidates) < n * 3 and attempts < n * 20:
        bug = gen.generate()
        attempts += 1
        if bug is None:
            continue
        if bug.depth < stage.min_depth:
            continue
        if stage.mutation_types and bug.mutation.mutation_type not in stage.mutation_types:
            continue
        prompt = build_prompt(bug, stage)
        candidates.append((bug, prompt))
    candidates.sort(key=lambda x: -_bug_sort_key(x[0], mutation_weights))
    return candidates[:n]


if __name__ == "__main__":
    for i, stage in enumerate(STAGES):
        print(f"\n{'='*60}")
        print(f"Stage {i}: {stage.name} — {stage.description}")
        print(f"  depth: {stage.min_depth}–{stage.max_depth or '∞'}")
        print(f"  weights: {stage.reward_weights}")
        print(f"  advance: mean reward > {stage.advance_threshold} over {stage.advance_window} episodes")

        batch = generate_training_batch(3, stage, seed=42 + i)
        print(f"  generated {len(batch)} examples")
        for bug, prompt in batch[:2]:
            print(f"    bug: {bug.mutation.mutation_type} @ {bug.mutation.step_name} depth={bug.depth}")
            print(f"    desc: {bug.mutation.description}")
            print(f"    changed: {bug.changed_fields[:3]}")

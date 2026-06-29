"""Gold investigation trace from MutatedBug — symptom → upstream path → root."""

from __future__ import annotations

from typing import Any

from training.bug_generator import MutatedBug

OUTPUT_FIELD_PRIORITY = (
    "ECL", "RWA", "LGD_REALIZADA", "STAGE_IFRS9", "STAGE_RECLASIFICADO",
    "MoC", "LGD_CON_MOC", "LGD_ESTIMADA", "EAD", "PD_ESTIMADA", "OR_EAD",
)

LINEAGE_TOOLS = frozenset({
    "trace_dependencies", "inspect_lineage", "trace_causal_chain", "get_sas_formula",
})


def _pick_symptom_field(bug: MutatedBug) -> str:
    for pref in OUTPUT_FIELD_PRIORITY:
        if pref in bug.changed_fields:
            return pref
    if bug.changed_fields:
        return bug.changed_fields[0]
    tv = bug.mutation.target_var
    if tv.upper() not in ("IF_CONDITION", "WHERE_FILTER"):
        return tv
    return "ECL"


def compute_gold_trace(bug: MutatedBug) -> dict[str, Any]:
    """BFS lineage path from observable symptom to mutation root."""
    from src.sas_logic_tree import SASLogicTree

    symptom = _pick_symptom_field(bug).upper()
    root_step = bug.mutation.step_name
    root_var = bug.mutation.target_var.upper()
    if root_var in ("IF_CONDITION", "WHERE_FILTER") and bug.changed_fields:
        root_var = bug.changed_fields[0].upper()

    empty = {
        "symptom_field": symptom,
        "root_step": root_step,
        "root_var": root_var,
        "path_fields": [symptom],
        "path_steps": [root_step] if root_step else [],
        "min_hops": max(1, bug.depth + 1),
    }
    if not bug.mutated_code:
        return empty

    try:
        tree = SASLogicTree()
        nodes = tree.parse(bug.mutated_code)
        trace = tree.trace_lineage(nodes, symptom, max_depth=None)
    except Exception:
        return empty

    ordered_fields: list[str] = []
    seen: set[str] = set()
    for layer in trace.get("layers", []):
        for f in layer.get("fields", []):
            fu = f.upper()
            if fu not in seen:
                seen.add(fu)
                ordered_fields.append(fu)

    if symptom not in seen:
        ordered_fields.insert(0, symptom)

    if root_var not in seen and root_var not in ("IF_CONDITION", "WHERE_FILTER"):
        ordered_fields.append(root_var)

    path_steps: list[str] = []
    for edge in trace.get("edges", []):
        ds = edge.get("data_step") or ""
        if ds and ds not in path_steps:
            src, tgt = edge.get("source", "").upper(), edge.get("target", "").upper()
            if src in seen or tgt in seen:
                path_steps.append(ds)
    if root_step and root_step not in path_steps:
        path_steps.append(root_step)

    return {
        "symptom_field": symptom,
        "root_step": root_step,
        "root_var": root_var,
        "path_fields": ordered_fields,
        "path_steps": path_steps,
        "min_hops": max(1, bug.depth + 1, len(path_steps)),
        "ancestor_count": trace.get("ancestor_count", 0),
    }


def attach_gold_trace(bug: MutatedBug) -> MutatedBug:
    bug.gold_trace = compute_gold_trace(bug)
    return bug


def oracle_trace_completion(bug: MutatedBug) -> str:
    """Gold completion: trace path + root identification."""
    import json

    gt = bug.gold_trace
    if not gt:
        attach_gold_trace(bug)
        gt = bug.gold_trace

    parts: list[str] = []
    prefix = gt.get("path_fields", [])[: gt.get("min_hops", 2) + 1]
    for field in prefix:
        parts.append(
            f'<tool_call>{{"name": "trace_dependencies", "arguments": '
            f'{{"target": "{field}", "session_id": "debug_lgd"}}}}</tool_call>'
        )
    if bug.depth >= 1:
        parts.append(
            '<tool_call>{"name": "query_table", "arguments": '
            '{"sql": "SELECT * FROM ECL LIMIT 3"}}</tool_call>'
        )
    parts.append(
        f'<tool_call>{{"name": "inspect_lineage", "arguments": '
        f'{{"target": "{gt.get("root_var", "ECL")}", "session_id": "debug_lgd"}}}}</tool_call>'
    )
    parts.append(
        f"PASO: {gt.get('root_step', bug.mutation.step_name)}\n"
        f"VARIABLE: {gt.get('root_var', bug.mutation.target_var)}\n"
        f"CAUSA: el bug está en la cadena desde {gt.get('symptom_field', 'ECL')} "
        f"({bug.mutation.mutation_type}).\n"
        f"CORRECCIÓN: {bug.mutation.original_expr.strip()}"
    )
    return "\n".join(parts)

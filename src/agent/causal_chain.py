"""Build a full causal chain from SAS lineage for root-cause reasoning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_SAS_SESSIONS_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "sas" / "sessions"

_ARITH = re.compile(r"[+\-*/]")


def _map_data_steps_to_files(session_id: str) -> dict[str, str]:
    folder = _SAS_SESSIONS_ROOT / session_id
    if not folder.exists():
        return {}
    mapping: dict[str, str] = {}
    for path in sorted(folder.rglob("*.sas")):
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"\bDATA\s+(work\.\w+)", text, re.I):
            mapping[m.group(1).upper()] = path.name
    return mapping


def _extract_sas_comments(
    session_id: str,
    fields: set[str],
    *,
    max_comments: int = 12,
) -> list[dict[str, Any]]:
    """Pull /* */ and * line comments mentioning target fields or BUG/missing."""
    folder = _SAS_SESSIONS_ROOT / session_id
    if not folder.exists():
        return []

    field_u = {f.upper() for f in fields}
    keywords = ("BUG", "MISSING", "FUSION", "COALESCE", "NO INFORM")
    hits: list[tuple[int, dict[str, Any]]] = []

    def _score(text: str) -> int:
        upper = text.upper()
        s = 0
        if "BUG" in upper:
            s += 20
        if "MISSING" in upper or "NO INFORM" in upper:
            s += 10
        s += sum(5 for f in field_u if f in upper)
        if any(k in upper for k in keywords):
            s += 2
        return s

    for path in sorted(folder.rglob("*.sas")):
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"/\*([^*]|\*(?!/))*\*/", text, re.DOTALL):
            block = m.group(0)
            inner = block[2:-2].strip()
            if not inner:
                continue
            upper = inner.upper()
            if not any(k in upper for k in keywords) and not any(f in upper for f in field_u):
                continue
            line = text[: m.start()].count("\n") + 1
            flat = " ".join(inner.split())
            hits.append((_score(flat), {
                "file": path.name,
                "line": line,
                "text": flat,
            }))

        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith("*") and "/*" not in stripped:
                continue
            upper = stripped.upper()
            if any(k in upper for k in keywords) or any(f in upper for f in field_u):
                flat = stripped.lstrip("* ").strip()
                hits.append((_score(flat), {
                    "file": path.name, "line": i, "text": flat,
                }))

    hits.sort(key=lambda x: -x[0])
    return [h for _, h in hits[:max_comments]]


def _primary_assign_expr(edges: list[dict], field: str) -> tuple[str, str, list[str]]:
    """Best assign expression writing ``field`` (longest expr wins ties)."""
    from src.sas_logic_tree import _vars_in_expr

    field_u = field.upper()
    best = ("", "", [])
    for e in edges:
        if e.get("target", "").upper() != field_u:
            continue
        if e.get("kind") != "assigns":
            continue
        expr = (e.get("expr") or "").strip()
        if len(expr) > len(best[0]):
            best = (expr, e.get("data_step") or "", sorted(_vars_in_expr(expr)))
    return best


def build_causal_chain(
    trace: dict[str, Any],
    session_id: str,
    *,
    include_propagation: bool = True,
) -> dict[str, Any]:
    """Turn a trace_lineage result into a hypothesis-friendly causal view."""
    target = trace.get("target", "")
    depth_map: dict[str, int] = trace.get("depth") or {}
    edges = trace.get("edges") or []
    step_files = _map_data_steps_to_files(session_id)

    steps: list[dict[str, Any]] = []
    seen_fields: set[str] = set()

    for field, depth in sorted(depth_map.items(), key=lambda kv: (kv[1], kv[0])):
        field_u = field.upper()
        if field_u in seen_fields:
            continue
        seen_fields.add(field_u)
        expr, data_step, inputs = _primary_assign_expr(edges, field_u)
        step: dict[str, Any] = {
            "depth": depth,
            "field": field_u,
            "expr": expr or None,
            "inputs": inputs,
            "data_step": data_step or None,
            "source_file": step_files.get((data_step or "").upper()),
        }
        if not expr and depth > 0:
            preds = [
                e["source"].upper()
                for e in edges
                if e.get("target", "").upper() == field_u and e.get("kind") == "conditions"
            ]
            if preds:
                step["conditioned_by"] = sorted(set(preds))
            step["role"] = "input_or_upstream"
        steps.append(step)

    # Missing propagation: deepest computed fields first
    computed = [s for s in steps if s.get("expr")]
    propagation: list[str] = []
    for s in sorted(computed, key=lambda x: -x["depth"]):
        expr = s["expr"] or ""
        if not _ARITH.search(expr):
            continue
        field = s["field"]
        for inp in s.get("inputs") or []:
            propagation.append(f"Si {inp}=. → {field}=. (por `{expr}`)")

    if not include_propagation:
        propagation = []

    fields_in_chain = {s["field"] for s in steps}
    comments = _extract_sas_comments(session_id, fields_in_chain)

    narrative_parts = [f"Cadena causal hacia {target}:"]
    by_depth = sorted(computed, key=lambda x: x["depth"])
    target_u = target.upper()
    for s in by_depth:
        indent = "  " * s["depth"]
        arrow = "→" if s["field"] == target_u else "←"
        narrative_parts.append(f"{indent}{arrow} {s['field']} = {s['expr']}")
    for s in steps:
        if s.get("expr"):
            continue
        if s["depth"] <= 0:
            continue
        indent = "  " * s["depth"]
        role = s.get("role", "upstream")
        narrative_parts.append(f"{indent}· {s['field']} ({role})")

    if include_propagation and propagation:
        narrative_parts.append("Propagación missing (inferida — contrastar con data.rows):")
        for p in propagation[:6]:
            narrative_parts.append(f"  • {p}")

    if comments:
        narrative_parts.append("Notas en código SAS:")
        for c in comments[:4]:
            narrative_parts.append(f"  • {c['file']}:{c['line']} — {c['text'][:160]}")

    return {
        "target": target,
        "session_id": session_id,
        "found": trace.get("found", False),
        "ancestor_count": trace.get("ancestor_count", 0),
        "steps": steps,
        "missing_propagation": propagation[:10],
        "sas_comments": comments,
        "pipeline_files": sorted(set(step_files.values())),
        "narrative": "\n".join(narrative_parts),
    }

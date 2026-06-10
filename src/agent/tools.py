"""Tool registry for the SAS-diff agent.

Each tool is described by a JSON schema (the ``OpenAI / Ollama tools``
format) and a Python callable. The registry is the *single source of
truth* for what the LLM can do.

Tools are deliberately small and side-effect free — they read the
filesystem, never write — so the agent loop is safe to run on any user
question without further sandboxing.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Filesystem layout ────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SAS_ROOT = _PROJECT_ROOT / "data" / "sas"
_SAMPLES = _PROJECT_ROOT / "data" / "samples"
_DOCS_ROOT = _PROJECT_ROOT / "data" / "docs"
_CSV_V2 = _SAMPLES / "cycles_v2.csv"
_CSV_V3 = _SAMPLES / "cycles_v3.csv"
_PK = "CICLO_ID"

_NUMERIC_COLS = {
    "PD_ESTIMADA", "LGD_ESTIMADA", "EAD", "DPDS", "STAGE_IFRS9",
    "PROVISION_PERIOD_MONTHS", "VENTANA_OBSERVACION_YEARS",
    "VENTANA_CALIBRACION_YEARS", "RATING_GRADO", "ECL",
    "LGD_FLOOR_APLICADO", "MOC", "CURE_FLAG", "RWA",
    "OR_EAD_TIT", "OR_EAD",  # potential demo fields
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_sas(version: str) -> str:
    """Load all `.sas` files under data/sas/{version}/, falling back to the
    bundled sample if the folder is empty/missing."""
    folder = _SAS_ROOT / version
    if folder.exists():
        files = sorted(folder.rglob("*.sas"))
        if files:
            return "\n\n".join(f.read_text(encoding="utf-8") for f in files)
    sample = _SAMPLES / "sample_lgd.sas"
    if sample.exists():
        return sample.read_text(encoding="utf-8")
    return ""


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        ku = k.upper() if isinstance(k, str) else k
        if v in ("", None):
            continue
        if ku in _NUMERIC_COLS and isinstance(v, str):
            try:
                out[ku] = float(v)
                continue
            except ValueError:
                pass
        out[ku] = v
    return out


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [_coerce_row(r) for r in csv.DictReader(f)]


def _truncate(obj: Any, limit: int = 4000) -> Any:
    """Compact a result so it doesn't blow up the LLM context."""
    s = json.dumps(obj, default=str)
    if len(s) <= limit:
        return obj
    return {"truncated": True, "preview": s[:limit] + "..."}


# ── Tool implementations ────────────────────────────────────────────────────


def _t_find_row(pk: str, version: str = "v3") -> dict[str, Any]:
    csv_path = _CSV_V2 if version.lower() == "v2" else _CSV_V3
    rows = _read_csv(csv_path)
    pk_u = pk.upper()
    for r in rows:
        if str(r.get(_PK, "")).upper() == pk_u:
            return {"found": True, "version": version, "pk": pk, "row": r}
    return {"found": False, "version": version, "pk": pk, "row": None}


def _approx_eq(a: Any, b: Any, tolerance: float) -> bool:
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) <= tolerance
    except (TypeError, ValueError):
        return False


def _t_find_rows_by_field_value(
    field: str,
    value: Any,
    version: str = "v3",
    tolerance: float = 1e-6,
    limit: int = 10,
) -> dict[str, Any]:
    csv_path = _CSV_V2 if version.lower() == "v2" else _CSV_V3
    rows = _read_csv(csv_path)
    field_u = field.upper()
    hits: list[dict[str, Any]] = []
    for r in rows:
        if _approx_eq(r.get(field_u), value, tolerance):
            hits.append({"pk": r.get(_PK), field_u: r.get(field_u)})
            if len(hits) >= limit:
                break
    return {"version": version, "field": field_u, "value": value, "matches": hits, "total_scanned": len(rows)}


def _t_inspect_lineage(target: str, sas_version: str = "v3") -> dict[str, Any]:
    from src.sas_logic_tree import SASLogicTree
    sas = _load_sas(sas_version)
    if not sas:
        return {"error": f"no SAS found for version {sas_version}"}
    tree = SASLogicTree()
    nodes = tree.parse(sas)
    trace = tree.trace_lineage(nodes, target)
    trace["sas_version"] = sas_version
    return trace


def _t_compute_attribution(pk: str, target: str, sas_version: str = "v3") -> dict[str, Any]:
    from src.sas_diff import explain_field_diff
    sas = _load_sas(sas_version)
    if not sas:
        return {"error": f"no SAS found for version {sas_version}"}
    rows_v2 = _read_csv(_CSV_V2)
    rows_v3 = _read_csv(_CSV_V3)
    pk_u = pk.upper()
    r_v2 = next((r for r in rows_v2 if str(r.get(_PK, "")).upper() == pk_u), None)
    r_v3 = next((r for r in rows_v3 if str(r.get(_PK, "")).upper() == pk_u), None)
    if not r_v2 or not r_v3:
        return {"error": f"row {pk} not found in V2 or V3 sample CSVs"}
    rep = explain_field_diff(
        sas_code=sas, row_v2=r_v2, row_v3=r_v3,
        target=target, method="both",
    )
    out = rep.to_dict()
    # Trim verbose fields, keep what the LLM needs
    keep = {
        "target": out["target"],
        "y_v2": out["y_v2"], "y_v3": out["y_v3"], "delta_y": out["delta_y"],
        "excluded_v2": out["excluded_v2"], "excluded_v3": out["excluded_v3"],
        "suspects": out["suspects"][:8],
        "field_deltas": out["field_deltas"][:20],
        "branch_flips": [
            {k: b[k] for k in ("condition", "data_step", "kind", "v2_taken", "v3_taken", "vars_in_condition")}
            for b in out["branch_flips"]
        ],
        "gradient_top": (out.get("gradient", {}) or {}).get("attributions", [])[:8],
        "shapley_top": (out.get("shapley", {}) or {}).get("attributions", [])[:8],
    }
    return keep


def _t_compare_sas_versions(target: str | None = None) -> dict[str, Any]:
    from src.agent.code_diff import compare
    res = compare(target=target)
    out = res.to_dict()
    # Truncate diff text to keep the context lean
    if len(out.get("unified_diff", "")) > 3500:
        out["unified_diff"] = out["unified_diff"][:3500] + "\n... [truncated]"
    # Don't let the LLM be flooded by a huge unchanged list
    out["unchanged_steps"] = out["unchanged_steps"][:20]
    return out


def _t_search_docs(query: str, k: int = 5) -> dict[str, Any]:
    from src.agent.docs_index import get_default_index
    idx = get_default_index()
    hits = idx.search(query, k=k)
    return {
        "query": query,
        "k": k,
        "hits": [h.to_dict() for h in hits],
        "doc_count": idx.section_count(),
    }


def _t_get_field_definition(field: str) -> dict[str, Any]:
    from src.agent.docs_index import get_default_index
    idx = get_default_index()
    res = idx.get_field_definition(field)
    return {"field": field.upper(), "result": res}


def _t_search_changelog(query: str, k: int = 3) -> dict[str, Any]:
    try:
        from src.knowledge import GraphRAG, collect_evidence, field_subgraph, linearise_subgraph
        rag = GraphRAG()
        g = rag.graph
        # Treat the query as a field name first; fall back to a substring scan over labels.
        q_u = query.upper().strip()
        sub = field_subgraph(g, q_u, hops=2)
        out = {
            "query": query,
            "evidence": collect_evidence(g, q_u),
            "linearised": linearise_subgraph(sub, q_u)[:1500],
        }
        return out
    except Exception as e:
        logger.warning("changelog tool failed: %s", e)
        return {"error": str(e), "query": query}


# ── Registry ────────────────────────────────────────────────────────────────


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        """Return the OpenAI/Ollama-format tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "find_row": ToolSpec(
        name="find_row",
        description=(
            "Fetch a specific cycle's full row from V2 or V3 sample data. "
            "Use this once you know the primary key (CICLO_ID) to inspect "
            "every input field for that cycle."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pk": {"type": "string", "description": "Cycle primary key, e.g. 'CIC_00031'"},
                "version": {"type": "string", "enum": ["v2", "v3"], "default": "v3"},
            },
            "required": ["pk"],
        },
        fn=_t_find_row,
    ),
    "find_rows_by_field_value": ToolSpec(
        name="find_rows_by_field_value",
        description=(
            "Find cycles where a given field has approximately a given value "
            "in V2 or V3. Useful when the user mentions a value but no PK."
        ),
        parameters={
            "type": "object",
            "properties": {
                "field": {"type": "string", "description": "Field name, e.g. 'OR_EAD_TIT'"},
                "value": {
                    "description": "Numeric or string value to match.",
                    "anyOf": [{"type": "number"}, {"type": "string"}, {"type": "boolean"}],
                },
                "version": {"type": "string", "enum": ["v2", "v3"], "default": "v3"},
                "tolerance": {"type": "number", "default": 1e-6, "description": "Numeric tolerance"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["field", "value"],
        },
        fn=_t_find_rows_by_field_value,
    ),
    "inspect_lineage": ToolSpec(
        name="inspect_lineage",
        description=(
            "Return the data-flow ancestors (and full graph) of a target field "
            "in the V2 or V3 SAS pipeline. Use this to understand which input "
            "fields *can* affect the target before computing attributions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "sas_version": {"type": "string", "enum": ["v2", "v3"], "default": "v3"},
            },
            "required": ["target"],
        },
        fn=_t_inspect_lineage,
    ),
    "compute_attribution": ToolSpec(
        name="compute_attribution",
        description=(
            "Run the full V2-vs-V3 attribution for a specific cycle and target "
            "field: path-integrated gradients (numeric) + Shapley values "
            "(categorical) + branch-flip detection. Returns y_v2, y_v3, "
            "ranked suspect fields, top contributions and any branch flips."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pk": {"type": "string"},
                "target": {"type": "string"},
                "sas_version": {"type": "string", "enum": ["v2", "v3"], "default": "v3", "description": "Which SAS code to evaluate the rows under."},
            },
            "required": ["pk", "target"],
        },
        fn=_t_compute_attribution,
    ),
    "compare_sas_versions": ToolSpec(
        name="compare_sas_versions",
        description=(
            "Diff the V2 vs V3 SAS code (data steps that produce or read the "
            "target field if given). Returns added/removed/modified steps "
            "and a unified text diff."
        ),
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Optional target field to scope the diff."},
            },
        },
        fn=_t_compare_sas_versions,
    ),
    "search_docs": ToolSpec(
        name="search_docs",
        description=(
            "Full-text search over the markdown corpus under data/docs/ "
            "(table dictionaries, flux explanations, field semantics)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
        fn=_t_search_docs,
    ),
    "get_field_definition": ToolSpec(
        name="get_field_definition",
        description=(
            "Look up the semantic definition of a specific field in the "
            "markdown docs corpus."
        ),
        parameters={
            "type": "object",
            "properties": {"field": {"type": "string"}},
            "required": ["field"],
        },
        fn=_t_get_field_definition,
    ),
    "search_changelog": ToolSpec(
        name="search_changelog",
        description=(
            "Search the database change-log GraphRAG for documented V2→V3 "
            "changes that mention a field. Returns evidence sections plus a "
            "linearised subgraph."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Field name preferred; otherwise free text."},
                "k": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
        fn=_t_search_changelog,
    ),
}


def tool_schemas() -> list[dict[str, Any]]:
    return [t.schema() for t in TOOL_REGISTRY.values()]


def dispatch_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run a tool by name with the given JSON args; always returns a dict."""
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return {"error": f"unknown_tool: {name}", "available": sorted(TOOL_REGISTRY)}
    try:
        result = spec.fn(**(args or {}))
    except TypeError as e:
        return {"error": f"bad_arguments: {e}", "tool": name, "args": args}
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return {"error": f"runtime_error: {e!s}", "tool": name}
    if not isinstance(result, dict):
        result = {"result": result}
    return _truncate(result)

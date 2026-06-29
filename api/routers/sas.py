"""SAS compiler router — parse, lineage extraction, sample fetch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/sas", tags=["sas"])

_ROOT = Path(__file__).resolve().parent.parent.parent
_SAS_ROOT = _ROOT / "data" / "sas"
_TRACE_VERSION = "v3"
_FALLBACK_SAS = _ROOT / "data" / "samples" / "sample_lgd.sas"


def _load_trace_sas() -> str:
    """Concatenate every ``.sas`` file under ``data/sas/v3/`` (sorted by path).

    The trace view operates on the full v3 program — all data steps and PROC
    SQL across every v3 file — so a field's lineage spans the whole module,
    not just one sample file. Falls back to the bundled sample when v3 is
    empty or missing.
    """
    folder = _SAS_ROOT / _TRACE_VERSION
    if folder.exists():
        files = sorted(folder.rglob("*.sas"))
        if files:
            return "\n\n".join(f.read_text(encoding="utf-8") for f in files)
    if _FALLBACK_SAS.exists():
        return _FALLBACK_SAS.read_text(encoding="utf-8")
    return ""


class CodeRequest(BaseModel):
    code: str


class LineageRequest(BaseModel):
    code: str
    target: str | None = None
    ancestors_only: bool = False
    max_depth: int | None = None


class EvaluateRequest(BaseModel):
    code: str
    values: dict[str, Any] = {}


class ValidateRequest(BaseModel):
    code: str


@router.get("/sample")
def get_sample() -> dict:
    code = _load_trace_sas()
    if not code.strip():
        raise HTTPException(404, "No v3 SAS code found")
    return {"code": code}


@router.post("/parse")
def parse(req: CodeRequest) -> dict:
    from src.sas_logic_tree import SASLogicTree
    tree = SASLogicTree()
    nodes = tree.parse(req.code)
    return {"ast": tree.to_dict(nodes)}


@router.post("/validate")
def validate(req: ValidateRequest) -> dict:
    from src.sas_logic_tree import SASLogicTree
    tree = SASLogicTree()
    nodes = tree.parse(req.code)
    diags = tree.validate(nodes)
    return {
        "diagnostics": diags,
        "uninterpreted": tree.uninterpreted,
        "total_uninterpreted": len(tree.uninterpreted),
    }


@router.post("/evaluate")
def evaluate(req: EvaluateRequest) -> dict:
    from src.sas_logic_tree import SASLogicTree
    tree = SASLogicTree()
    nodes = tree.parse(req.code)
    trace = tree.evaluate(nodes, req.values)
    return {
        "initial": trace.initial,
        "final": trace.final,
        "steps": [
            {"kind": s.kind, "label": s.label, "var": s.var,
             "old_val": s.old_val, "new_val": s.new_val,
             "data_step": s.data_step}
            for s in trace.steps
        ],
        "row_passes_filter": trace.row_passes_filter,
        "filter_results": trace.filter_results,
    }


@router.post("/lineage")
def lineage(req: LineageRequest) -> dict:
    from src.sas_logic_tree import SASLogicTree, trace_field_ancestors
    tree = SASLogicTree()
    nodes = tree.parse(req.code)
    lg = tree.lineage(nodes)

    if not req.target:
        return {
            "nodes": lg.nodes,
            "edges": lg.edges,
            "data_steps": lg.data_steps,
        }

    trace = trace_field_ancestors(lg, req.target, max_depth=req.max_depth)
    if req.ancestors_only:
        return trace

    return {
        "nodes": lg.nodes,
        "edges": lg.edges,
        "data_steps": lg.data_steps,
        "trace": trace,
    }

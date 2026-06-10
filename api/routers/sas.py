"""SAS compiler router — parse, lineage extraction, sample fetch."""

from __future__ import annotations

from pathlib import Path

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

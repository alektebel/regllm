"""SAS compiler router — parse, lineage extraction, sample fetch."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/sas", tags=["sas"])

_SAMPLE_SAS = Path(__file__).resolve().parent.parent.parent / "data" / "samples" / "sample_lgd.sas"


class CodeRequest(BaseModel):
    code: str


class LineageRequest(BaseModel):
    code: str
    target: str | None = None
    ancestors_only: bool = False
    max_depth: int | None = None


@router.get("/sample")
def get_sample() -> dict:
    if not _SAMPLE_SAS.exists():
        raise HTTPException(404, "Sample SAS file not found")
    return {"code": _SAMPLE_SAS.read_text(encoding="utf-8")}


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

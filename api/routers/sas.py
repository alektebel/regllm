"""SAS compiler router — parse, lineage extraction, and cycle simulation.

Endpoints
---------
GET  /sas/sample                — return sample_lgd.sas code
GET  /sas/cycles                — paginated list of cycles from cycles_sample.csv
POST /sas/parse                 — SAS code → JSON AST
POST /sas/lineage               — SAS code → field dependency graph
POST /sas/simulate              — SAS code + variable values → execution trace
POST /sas/simulate-cycle        — CICLO_ID → load row from CSV, run trace
POST /sas/upload                — .egp/.zip/.sas file → combined lineage graph
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

router = APIRouter(prefix="/sas", tags=["sas"])

_DATA = Path(__file__).parent.parent.parent / "data" / "samples"
_SAMPLE_SAS = _DATA / "sample_lgd.sas"
_CYCLES_CSV = _DATA / "cycles_sample.csv"

# Fields that should be parsed as numbers when loading CSV rows
_NUMERIC = {
    "PD_ESTIMADA", "LGD_ESTIMADA", "EAD", "DPDS", "STAGE_IFRS9",
    "PROVISION_PERIOD_MONTHS", "VENTANA_OBSERVACION_YEARS", "VENTANA_CALIBRACION_YEARS",
    "RATING_GRADO", "ECL", "LGD_FLOOR_APLICADO", "MOC", "CURE_FLAG", "RWA",
}

# Fields to exclude from simulation input (computed outputs or metadata)
_SIM_EXCLUDE = {
    "CICLO_ID", "CONTRATO", "ECL", "LGD_FLOOR_APLICADO",
    "FECHA_INCUMPLIMIENTO", "CURE_FLAG", "PERIODO_OBSERVACION", "RWA", "MOC",
}


def _load_cycles() -> list[dict[str, Any]]:
    if not _CYCLES_CSV.exists():
        return []
    rows: list[dict[str, Any]] = []
    with _CYCLES_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, Any] = {}
            for k, v in row.items():
                ku = k.upper()
                if ku in _NUMERIC:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
                else:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _trace_to_dict(trace) -> dict:
    return {
        "initial": trace.initial,
        "final": trace.final,
        "row_passes_filter": trace.row_passes_filter,
        "steps": [
            {
                "kind": s.kind,
                "label": s.label,
                "var": s.var,
                "old_val": s.old_val,
                "new_val": s.new_val,
            }
            for s in trace.steps
        ],
    }


# ── Models ────────────────────────────────────────────────────────────────────

class CodeRequest(BaseModel):
    code: str


class SimulateRequest(BaseModel):
    code: str
    values: dict[str, Any]


class SimulateCycleRequest(BaseModel):
    ciclo_id: str
    code: str | None = None  # defaults to sample_lgd.sas


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/sample")
def get_sample():
    if not _SAMPLE_SAS.exists():
        raise HTTPException(404, "Sample SAS file not found")
    return {"code": _SAMPLE_SAS.read_text(encoding="utf-8")}


@router.get("/cycles")
def list_cycles(limit: int = 200, offset: int = 0):
    cycles = _load_cycles()
    return {
        "total": len(cycles),
        "items": cycles[offset: offset + limit],
    }


@router.post("/parse")
def parse_sas(req: CodeRequest):
    from src.sas_logic_tree import SASLogicTree
    tree = SASLogicTree()
    nodes = tree.parse(req.code)
    return {"ast": tree.to_dict(nodes)}


@router.post("/lineage")
def sas_lineage(req: CodeRequest):
    from src.sas_logic_tree import SASLogicTree
    tree = SASLogicTree()
    nodes = tree.parse(req.code)
    lg = tree.lineage(nodes)
    return {
        "nodes": lg.nodes,
        "edges": lg.edges,
        "data_steps": lg.data_steps,
    }


@router.post("/simulate")
def simulate_sas(req: SimulateRequest):
    from src.sas_logic_tree import SASLogicTree
    tree = SASLogicTree()
    nodes = tree.parse(req.code)
    trace = tree.evaluate(nodes, req.values)
    return _trace_to_dict(trace)


@router.post("/simulate-cycle")
def simulate_cycle(req: SimulateCycleRequest):
    from src.sas_logic_tree import SASLogicTree

    cycles = _load_cycles()
    row = next((c for c in cycles if c.get("CICLO_ID") == req.ciclo_id), None)
    if row is None:
        raise HTTPException(404, f"Cycle {req.ciclo_id!r} not found")

    code = req.code
    if not code:
        if not _SAMPLE_SAS.exists():
            raise HTTPException(400, "No SAS code provided and sample file not found")
        code = _SAMPLE_SAS.read_text(encoding="utf-8")

    # Use uppercase keys; exclude computed/metadata fields
    values: dict[str, Any] = {
        k.upper(): v
        for k, v in row.items()
        if k.upper() not in _SIM_EXCLUDE and v != "" and v is not None
    }

    tree = SASLogicTree()
    nodes = tree.parse(code)
    trace = tree.evaluate(nodes, values)
    return {"ciclo_id": req.ciclo_id, "input_values": values, **_trace_to_dict(trace)}


_ALLOWED_EXTS = {".egp", ".zip", ".sas", ".txt"}


@router.post("/upload")
async def upload_sas(file: UploadFile = File(...)):
    """
    Upload a .egp / .zip / .sas file.
    Extracts all SAS code blocks, concatenates them, and returns:
      - combined lineage graph (nodes + edges)
      - list of code blocks found (name, type, char count)
      - the combined SAS source (so the frontend can pass it to /simulate-cycle)
    """
    from src.sas_parser import SASParser
    from src.sas_logic_tree import SASLogicTree

    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. Allowed: {sorted(_ALLOWED_EXTS)}",
        )

    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        blocks = SASParser().parse(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not blocks:
        raise HTTPException(422, "No SAS code blocks found in the uploaded file")

    # Combine all blocks into one source string for unified lineage
    combined_code = "\n\n".join(b.code for b in blocks)

    tree = SASLogicTree()
    ast_nodes = tree.parse(combined_code)
    lg = tree.lineage(ast_nodes)

    return {
        "filename": file.filename,
        "blocks": [
            {
                "task_name": b.task_name,
                "block_type": b.block_type,
                "chars": len(b.code),
            }
            for b in blocks
        ],
        "combined_code": combined_code,
        "lineage": {
            "nodes": lg.nodes,
            "edges": lg.edges,
            "data_steps": lg.data_steps,
        },
    }

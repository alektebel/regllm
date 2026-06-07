"""Field-discrepancy explainer router.

Endpoints
---------
GET  /diff/sample-rows        — load V2/V3 sample CSVs paired by primary key
POST /diff/explain            — single-row explainer with Gemma justification
POST /diff/explain-batch      — multi-row SSE stream
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/diff", tags=["diff"])

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SAMPLES = _PROJECT_ROOT / "data" / "samples"
_SAS_ROOT = _PROJECT_ROOT / "data" / "sas"
_CSV_V2 = _SAMPLES / "cycles_v2.csv"
_CSV_V3 = _SAMPLES / "cycles_v3.csv"
_SAMPLE_SAS = _SAMPLES / "sample_lgd.sas"


def _load_version_sas(version: str) -> str:
    """Concatenate every *.sas file under data/sas/{version}/ (sorted), with
    a fallback to the bundled sample SAS so the manual flow always has
    something to chew on."""
    folder = _SAS_ROOT / version
    if folder.exists():
        files = sorted(folder.glob("*.sas"))
        if files:
            return "\n\n".join(f.read_text(encoding="utf-8") for f in files)
    if _SAMPLE_SAS.exists():
        return _SAMPLE_SAS.read_text(encoding="utf-8")
    return ""

_NUMERIC_COLS = {
    "PD_ESTIMADA", "LGD_ESTIMADA", "EAD", "DPDS", "STAGE_IFRS9",
    "PROVISION_PERIOD_MONTHS", "VENTANA_OBSERVACION_YEARS",
    "VENTANA_CALIBRACION_YEARS", "RATING_GRADO", "ECL",
    "LGD_FLOOR_APLICADO", "MOC", "CURE_FLAG", "RWA",
}

_PK = "CICLO_ID"


# ── Models ───────────────────────────────────────────────────────────────────


class ExplainRequest(BaseModel):
    sas_code: str | None = None
    sas_v2: str | None = None
    sas_v3: str | None = None
    row_v2: dict[str, Any]
    row_v3: dict[str, Any]
    target_field: str
    method: Literal["grad", "shapley", "both"] = "both"
    grad_steps: int = 32
    shapley_n_samples: int = 256
    use_gemma: bool = True
    top_k_for_llm: int = 5
    include_code_diff: bool = True
    include_code_vs_data: bool = True


class ExplainBatchRequest(BaseModel):
    sas_code: str | None = None
    target_field: str
    method: Literal["grad", "shapley", "both"] = "both"
    use_gemma: bool = False           # off by default; SSE+LLM is slow
    pk: str = _PK
    only_changed: bool = True


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_sas_code(sas_code: str | None) -> str:
    if sas_code:
        return sas_code
    if _SAMPLE_SAS.exists():
        return _SAMPLE_SAS.read_text(encoding="utf-8")
    raise HTTPException(400, "No SAS code provided and sample file not found")


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    """Cast numeric strings to floats; uppercase keys; drop blanks."""
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


def _attach_method(items: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [{**i, "_method": method} for i in items]


def _annotate_with_gemma(
    target: str,
    grad_attrs: list[dict[str, Any]],
    shap_attrs: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any] | None:
    try:
        from src.knowledge import GraphRAG
    except Exception as e:
        logger.warning("GraphRAG unavailable: %s", e)
        return None
    # Combine: prefer Shapley contributions for ranking when both are present
    combined = _attach_method(shap_attrs or grad_attrs or [], "shapley" if shap_attrs else "gradient")
    rag = GraphRAG()
    rep = rag.justify_report(target=target, attributions=combined, top_k=top_k)
    return rep.to_dict()


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("/sas-pair")
def sas_pair() -> dict:
    """Return the V2 and V3 SAS source code from data/sas/{v2,v3}/.

    Falls back to data/samples/sample_lgd.sas for both versions if the
    folders are empty. Used by the manual-mode editor to populate the V2
    and V3 tabs.
    """
    v2 = _load_version_sas("v2")
    v3 = _load_version_sas("v3")
    return {
        "v2": v2,
        "v3": v3,
        "differs": v2.strip() != v3.strip(),
        "v2_lines": v2.count("\n") + 1 if v2 else 0,
        "v3_lines": v3.count("\n") + 1 if v3 else 0,
        "fallback_v2": not (_SAS_ROOT / "v2").exists() or not list((_SAS_ROOT / "v2").glob("*.sas")),
        "fallback_v3": not (_SAS_ROOT / "v3").exists() or not list((_SAS_ROOT / "v3").glob("*.sas")),
    }


@router.get("/sample-rows")
def sample_rows(limit: int = 200, only_changed: bool = True, target: str = "ECL") -> dict:
    """Pair-up V2 and V3 CSVs by primary key; return up to ``limit`` matched pairs."""
    v2 = _read_csv(_CSV_V2)
    v3 = _read_csv(_CSV_V3)
    if not v2 or not v3:
        raise HTTPException(404, "Sample V2/V3 CSVs not found — run scripts/generate_v3.py")
    by_pk_v3 = {r.get(_PK): r for r in v3}
    pairs: list[dict[str, Any]] = []
    for r2 in v2:
        pk = r2.get(_PK)
        r3 = by_pk_v3.get(pk)
        if r3 is None:
            continue
        target_u = target.upper()
        if only_changed and r2.get(target_u) == r3.get(target_u):
            continue
        pairs.append({
            "pk": pk,
            "v2": r2,
            "v3": r3,
            "target_v2": r2.get(target_u),
            "target_v3": r3.get(target_u),
        })
        if len(pairs) >= limit:
            break
    return {"target": target.upper(), "total": len(pairs), "items": pairs}


def _evaluate_target(sas_code: str, row: dict[str, Any], target: str) -> float | None:
    """Run the eager Python evaluator on ``row`` under ``sas_code``; return
    the numeric value of ``target`` (or ``None`` if it isn't produced or
    the row is filtered out)."""
    try:
        from src.sas_logic_tree import SASLogicTree
        tree = SASLogicTree()
        nodes = tree.parse(sas_code)
        trace = tree.evaluate(nodes, row)
        if not trace.row_passes_filter:
            return None
        v = trace.final.get(target.upper())
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    except Exception as e:
        logger.warning("Code-vs-data evaluation failed: %s", e)
        return None


def _code_vs_data_decomposition(
    sas_v2: str, sas_v3: str,
    row_v2: dict[str, Any], row_v3: dict[str, Any],
    target: str,
) -> dict[str, Any] | None:
    """Approximate decomposition of ΔY = Y(row_v3, code_v3) − Y(row_v2, code_v2)
    into a *data* component and a *code* component using the V3 code as the
    common reference frame:

        ΔY_data ≈ Y(row_v3, code_v3) − Y(row_v2, code_v3)
        ΔY_code ≈ Y(row_v2, code_v3) − Y(row_v2, code_v2)
        ΔY_total = ΔY_data + ΔY_code   (exactly, by construction)

    Two extra evaluations on top of the standard explainer pipeline. We
    skip the decomposition if any anchor returns ``None`` (filtered rows,
    missing target, etc.).
    """
    if sas_v2.strip() == sas_v3.strip():
        return None  # identical pipelines → no code component

    y_v2_under_v2 = _evaluate_target(sas_v2, row_v2, target)
    y_v3_under_v3 = _evaluate_target(sas_v3, row_v3, target)
    y_v2_under_v3 = _evaluate_target(sas_v3, row_v2, target)

    if any(v is None for v in (y_v2_under_v2, y_v3_under_v3, y_v2_under_v3)):
        return {
            "available": False,
            "reason": "one or more anchor evaluations returned None",
            "y_v2_under_v2": y_v2_under_v2,
            "y_v3_under_v3": y_v3_under_v3,
            "y_v2_under_v3": y_v2_under_v3,
        }

    delta_total = y_v3_under_v3 - y_v2_under_v2
    delta_data = y_v3_under_v3 - y_v2_under_v3   # holds code = V3, swaps row
    delta_code = y_v2_under_v3 - y_v2_under_v2   # holds row = V2, swaps code

    denom = abs(delta_total) if abs(delta_total) > 1e-12 else 1.0
    return {
        "available": True,
        "y_v2_under_v2": y_v2_under_v2,
        "y_v3_under_v3": y_v3_under_v3,
        "y_v2_under_v3": y_v2_under_v3,
        "delta_total": delta_total,
        "delta_data": delta_data,
        "delta_code": delta_code,
        "share_data": delta_data / denom,
        "share_code": delta_code / denom,
    }


def _code_diff_for_target(sas_v2: str, sas_v3: str, target: str) -> dict[str, Any] | None:
    """Compute a target-scoped V2-vs-V3 SAS code diff."""
    if sas_v2.strip() == sas_v3.strip():
        return None
    try:
        from src.agent.code_diff import compare
        res = compare(target=target, sas_v2=sas_v2, sas_v3=sas_v3)
        out = res.to_dict()
        # Trim the unified diff for transport
        if len(out.get("unified_diff", "")) > 6000:
            out["unified_diff"] = out["unified_diff"][:6000] + "\n... [truncated]"
        return out
    except Exception as e:
        logger.warning("code_diff failed: %s", e)
        return None


@router.post("/explain")
def explain(req: ExplainRequest) -> dict:
    from src.sas_diff import explain_field_diff

    # Resolve V2/V3 code with sensible fallbacks. The new dual-tab UI sends
    # both; older callers can still pass a single sas_code (used for both
    # versions) or rely on the bundled sample.
    sas_v3 = req.sas_v3 or req.sas_code or _load_version_sas("v3")
    sas_v2 = req.sas_v2 or req.sas_code or _load_version_sas("v2")
    if not sas_v3:
        raise HTTPException(400, "No SAS V3 code available")
    v2 = _coerce_row(req.row_v2)
    v3 = _coerce_row(req.row_v3)

    # Attribution is computed under V3 SAS — that is the "post-change"
    # pipeline; the gradient/Shapley reports tell you which input fields
    # would have produced the observed delta if both rows had been run
    # under V3 code.
    report = explain_field_diff(
        sas_code=sas_v3,
        row_v2=v2,
        row_v3=v3,
        target=req.target_field,
        method=req.method,
        grad_steps=req.grad_steps,
        shapley_n_samples=req.shapley_n_samples,
    )

    out = report.to_dict()

    # ── V2-vs-V3 code aware extras ───────────────────────────────────────
    code_differs = sas_v2.strip() != sas_v3.strip()
    out["code_differs"] = code_differs
    if req.include_code_diff and code_differs:
        cd = _code_diff_for_target(sas_v2, sas_v3, req.target_field)
        if cd is not None:
            out["code_diff"] = cd
    if req.include_code_vs_data and code_differs:
        decomp = _code_vs_data_decomposition(sas_v2, sas_v3, v2, v3, req.target_field)
        if decomp is not None:
            out["code_vs_data"] = decomp

    if req.use_gemma:
        grad_attrs = out.get("gradient", {}).get("attributions") if out.get("gradient") else []
        shap_attrs = out.get("shapley", {}).get("attributions") if out.get("shapley") else []
        out["graphrag"] = _annotate_with_gemma(
            target=req.target_field,
            grad_attrs=grad_attrs or [],
            shap_attrs=shap_attrs or [],
            top_k=req.top_k_for_llm,
        )
    return out


@router.post("/explain-batch")
async def explain_batch(req: ExplainBatchRequest):
    """SSE stream: explain every paired row whose target field changed."""
    sas_code = _load_sas_code(req.sas_code)

    async def gen():
        from src.sas_diff import explain_field_diff
        v2_rows = _read_csv(_CSV_V2)
        v3_rows = _read_csv(_CSV_V3)
        by_pk_v3 = {r.get(req.pk): r for r in v3_rows}
        target_u = req.target_field.upper()

        pending = []
        for r2 in v2_rows:
            pk = r2.get(req.pk)
            r3 = by_pk_v3.get(pk)
            if r3 is None:
                continue
            if req.only_changed and r2.get(target_u) == r3.get(target_u):
                continue
            pending.append((pk, r2, r3))

        yield f"data: {json.dumps({'event': 'start', 'total': len(pending)})}\n\n"

        for i, (pk, r2, r3) in enumerate(pending):
            try:
                report = explain_field_diff(
                    sas_code=sas_code,
                    row_v2=r2, row_v3=r3,
                    target=req.target_field,
                    method=req.method,
                )
                payload = {
                    "event": "row",
                    "i": i,
                    "pk": pk,
                    "report": report.to_dict(),
                }
            except Exception as e:
                payload = {"event": "row_error", "i": i, "pk": pk, "error": str(e)}
            yield f"data: {json.dumps(payload, default=str)}\n\n"
            await asyncio.sleep(0)

        yield f"data: {json.dumps({'event': 'done'})}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

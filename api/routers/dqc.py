"""DQC generator — minimal MVP.

One LLM call grounded in:
  1. A field dictionary (Excel upload)
  2. Natural-language DQC instructions

Generated checks land in the validation store as ``pending``; the Angular UI
reviews, validates, and exports them.
"""

from __future__ import annotations

import io
import json
import logging
import re
import sqlite3
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from src.knowledge import get_client
from training.dq import checks_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dqc", tags=["dqc"])


# ── Models ────────────────────────────────────────────────────────────────────

class DQCItem(BaseModel):
    dqc_id: str = ""
    variable: str = ""
    descripcion: str = ""
    tipo: str = ""
    severidad: str = ""
    regla_sql: str = ""
    condicion_error: str = ""
    campos_entrada: list[str] = []
    referencia_regulatoria: str = ""
    umbral: str = ""
    periodicidad: str = ""
    justificacion: str = ""


class GenerateResponse(BaseModel):
    dqcs: list[DQCItem]
    dictionary_fields: int
    context_summary: str = ""


class CheckRecord(BaseModel):
    check_id: str
    rule_id: str | None = None
    name: str
    description: str = ""
    severity: str
    category: str
    sql: str
    visible: bool = True
    status: str = "pending"
    reward: float | None = None
    variable: str | None = None
    tipo: str | None = None
    condicion_error: str | None = None
    campos_entrada: list[str] = []
    referencia_regulatoria: str | None = None
    umbral: str | None = None
    periodicidad: str | None = None
    justificacion: str | None = None
    created_at: str | None = None
    validated_at: str | None = None


class StatusUpdate(BaseModel):
    status: str  # "validated" | "rejected"


class DashboardResponse(BaseModel):
    ready: bool
    pending_visible: int
    validated: int
    rejected: int
    oculto: int
    sql: str | None = None
    checks: list[CheckRecord] = []


# ── Prompt ──────────────────────────────────────────────────────────────────

DQC_SYSTEM_PROMPT = """\
Eres un experto en calidad de datos para reporting regulatorio bancario.

Recibirás:
1. Un diccionario de campos (tabla con nombre, tipo, descripción, etc.)
2. Una lista de instrucciones en lenguaje natural que describen reglas DQC

Para CADA instrucción genera al menos un control DQC como consulta SQL:
- Usa la tabla `mylib.ciclos_recuperacion` salvo que el diccionario indique otra.
- Usa nombres de campo EXACTOS del diccionario (MAYÚSCULAS).
- Cada DQC debe cubrir format checks (dominio, nulos, rangos) o reperformance \
  (fórmulas documentadas) según la instrucción.
- Si la instrucción no se puede fundamentar con el diccionario, indica \
  "Sin referencia en diccionario" en `referencia_regulatoria`.
- No omitas ninguna instrucción.

Responde SOLO con JSON: {"dqcs": [...]}. Cada objeto:
{
  "dqc_id": "DQC_<CAMPO>_<NNN>",
  "variable": "<campo principal>",
  "descripcion": "<qué verifica>",
  "tipo": "formula|rango|completitud|consistencia|referencial",
  "severidad": "bloqueante|advertencia|informativo",
  "regla_sql": "<SQL SELECT de filas que violan la regla>",
  "condicion_error": "<cuándo falla>",
  "campos_entrada": ["campo1", "campo2"],
  "referencia_regulatoria": "<del diccionario o 'Sin referencia en diccionario'>",
  "umbral": "<si aplica>",
  "periodicidad": "mensual",
  "justificacion": "<por qué>"
}
"""


_SEV_MAP = {"bloqueante": "HIGH", "advertencia": "MED", "informativo": "LOW"}
_CAT_MAP = {
    "formula": "formula", "consistencia": "consistencia",
    "referencial": "referencial", "rango": "rango", "completitud": "completitud",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _db() -> sqlite3.Connection:
    return checks_db.connect()


def _parse_excel_dictionary(data: bytes) -> tuple[str, int]:
    """Read an Excel field dictionary into compact text for the LLM prompt."""
    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    ws = wb.active
    if ws is None:
        wb.close()
        return "", 0

    rows_iter = ws.iter_rows(values_only=True)
    header_row = next(rows_iter, None)
    if not header_row:
        wb.close()
        return "", 0

    headers = [str(c or "").strip() for c in header_row]
    # Normalise common Spanish/English column names
    field_col = _pick_col(headers, ("field", "campo", "columna", "nombre", "name"))
    type_col = _pick_col(headers, ("type", "tipo", "datatype"))
    desc_col = _pick_col(headers, ("description", "descripcion", "descripción", "definicion"))
    null_col = _pick_col(headers, ("null", "nulo", "nullable", "obligatorio"))
    formula_col = _pick_col(headers, ("formula", "derivacion", "derivación", "calculo"))
    reg_col = _pick_col(headers, ("reg ref", "reg_ref", "referencia", "regulacion"))

    lines: list[str] = []
    count = 0
    for row in rows_iter:
        if not row or all(v is None or str(v).strip() == "" for v in row):
            continue
        cells = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
        field = _cell(cells, field_col)
        if not field:
            continue
        count += 1
        parts = [f"- {field}"]
        if type_col:
            parts.append(f"type={_cell(cells, type_col)}")
        if null_col:
            parts.append(f"null={_cell(cells, null_col)}")
        if desc_col:
            parts.append(f"desc={_cell(cells, desc_col)}")
        if formula_col:
            parts.append(f"formula={_cell(cells, formula_col)}")
        if reg_col:
            parts.append(f"reg={_cell(cells, reg_col)}")
        lines.append(", ".join(parts))

    wb.close()
    return "\n".join(lines), count


def _pick_col(headers: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {h.lower(): h for h in headers if h}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def _cell(cells: dict[str, Any], col: str | None) -> str:
    if not col:
        return ""
    v = cells.get(col)
    if v is None:
        return ""
    return str(v).strip()


def _extract_dqc_list(result: Any) -> list[dict]:
    if not isinstance(result, dict):
        return []
    lower_map = {k.lower(): k for k in result}
    for candidate in ("dqcs", "dqc_list", "checks", "controles"):
        real_key = lower_map.get(candidate)
        if real_key and isinstance(result[real_key], list):
            return result[real_key]
    for v in result.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    return []


def _parse_dqc_items(result: Any) -> list[DQCItem]:
    items: list[DQCItem] = []
    for d in _extract_dqc_list(result):
        cleaned: dict[str, Any] = {}
        for k, v in d.items():
            if k not in DQCItem.model_fields or v is None:
                continue
            ft = DQCItem.model_fields[k].annotation
            if ft is str and not isinstance(v, str):
                v = str(v)
            cleaned[k] = v
        items.append(DQCItem(**cleaned))
    return items


def _persist_dqc_items(items: list[DQCItem]) -> list[str]:
    ids: list[str] = []
    conn = _db()
    try:
        for it in items:
            sev = _SEV_MAP.get(it.severidad, it.severidad or "MED")
            cat = _CAT_MAP.get(it.tipo, it.tipo or "consistencia")
            try:
                cid = checks_db.insert_check(
                    conn,
                    name=(it.dqc_id or "dqc").lower(),
                    description=it.descripcion,
                    severity=sev,
                    category=cat,
                    sql=it.regla_sql,
                    visible=True,
                    status="pending",
                    variable=it.variable,
                    tipo=it.tipo,
                    condicion_error=it.condicion_error,
                    campos_entrada=it.campos_entrada,
                    referencia_regulatoria=it.referencia_regulatoria,
                    umbral=it.umbral,
                    periodicidad=it.periodicidad,
                    justificacion=it.justificacion,
                )
                ids.append(cid)
            except sqlite3.IntegrityError as exc:
                logger.warning("DQC persist clash: %s", exc)
    finally:
        conn.close()
    return ids


def _split_instructions(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines
    return [text.strip()] if text.strip() else []


# ── Generation ──────────────────────────────────────────────────────────────

@router.post("/generate", response_model=GenerateResponse)
async def generate_dqc(
    dictionary: UploadFile = File(..., description="Field dictionary (.xlsx)"),
    instructions: str = Form(..., description="DQC rules, one per line"),
    table_name: str = Form("mylib.ciclos_recuperacion"),
) -> GenerateResponse:
    """Generate DQCs from an Excel field dictionary + NL instructions."""
    if not dictionary.filename or not dictionary.filename.lower().endswith((".xlsx", ".xls")):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="dictionary must be an Excel file (.xlsx)")

    raw = await dictionary.read()
    dict_text, field_count = _parse_excel_dictionary(raw)
    if field_count == 0:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="Could not read any fields from the Excel dictionary. "
                   "Expected columns like Field/Campo, Type/Tipo, Description/Descripcion.",
        )

    instr_lines = _split_instructions(instructions)
    if not instr_lines:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="instructions cannot be empty")

    user_prompt = (
        f"Tabla objetivo: {table_name}\n\n"
        f"DICCIONARIO DE CAMPOS ({field_count} campos):\n{dict_text}\n\n"
        f"INSTRUCCIONES DQC ({len(instr_lines)} reglas):\n"
        + "\n".join(f"{i+1}. {ln}" for i, ln in enumerate(instr_lines))
    )

    try:
        result = get_client().chat_json(
            system=DQC_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=8192,
        )
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return GenerateResponse(
            dqcs=[],
            dictionary_fields=field_count,
            context_summary=f"Error al generar DQCs: {exc}",
        )

    dqcs = _parse_dqc_items(result)
    try:
        saved = _persist_dqc_items(dqcs)
        logger.info("persisted %d/%d DQCs", len(saved), len(dqcs))
    except Exception as exc:
        logger.warning("persist failed: %s", exc)

    return GenerateResponse(
        dqcs=dqcs,
        dictionary_fields=field_count,
        context_summary=(
            f"Se generaron {len(dqcs)} DQCs a partir de {len(instr_lines)} "
            f"instrucción(es) y {field_count} campos del diccionario."
        ),
    )


# ── Validation pipeline ───────────────────────────────────────────────────────

@router.get("/checks", response_model=list[CheckRecord])
def list_checks(
    status: str | None = None,
    visible: bool | None = True,
) -> list[CheckRecord]:
    conn = _db()
    try:
        rows = checks_db.list_checks(conn, status=status, visible=visible)
        return [CheckRecord(**r) for r in rows]
    finally:
        conn.close()


@router.get("/checks/counts")
def checks_counts() -> dict:
    conn = _db()
    try:
        return checks_db.counts(conn)
    finally:
        conn.close()


@router.post("/checks/{check_id}/status", response_model=CheckRecord)
def update_check_status(check_id: str, body: StatusUpdate) -> CheckRecord:
    from fastapi import HTTPException
    conn = _db()
    try:
        if not checks_db.set_status(conn, check_id, body.status):
            raise HTTPException(status_code=404, detail=f"check_id {check_id} not found")
        row = checks_db.get_check(conn, check_id)
        return CheckRecord(**row)
    finally:
        conn.close()


@router.delete("/checks/{check_id}")
def delete_check(check_id: str) -> dict:
    from fastapi import HTTPException
    conn = _db()
    try:
        if not checks_db.delete_check(conn, check_id):
            raise HTTPException(status_code=404, detail=f"check_id {check_id} not found")
        return {"deleted": check_id}
    finally:
        conn.close()


@router.get("/dashboard", response_model=DashboardResponse)
def dashboard() -> DashboardResponse:
    conn = _db()
    try:
        c = checks_db.counts(conn)
        sql = checks_db.build_dashboard_query(conn, status="validated") if c["dashboard_ready"] else None
        validated = checks_db.export_validated(conn) if c["dashboard_ready"] else []
        return DashboardResponse(
            ready=c["dashboard_ready"],
            pending_visible=c["pending_visible"],
            validated=c["validated"],
            rejected=c["rejected"],
            oculto=c["oculto"],
            sql=sql,
            checks=[CheckRecord(**r) for r in validated],
        )
    finally:
        conn.close()

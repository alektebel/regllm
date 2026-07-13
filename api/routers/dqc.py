"""DQC generator — minimal MVP.

One LLM call grounded in:
  1. A field dictionary (Excel upload)
  2. Natural-language DQC instructions

Generated checks land in the validation store as ``pending``; the Angular UI
reviews, validates, and exports them.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from src.knowledge import get_client
from training.dq import checks_db

from . import dqc_dictionary as dict_ai

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
    sheet_used: str = ""
    mapping_source: str = ""       # llm | heuristic | user
    formats_inferred: int = 0
    agents_used: int = 0           # stateless LLM calls spent on this run


class SheetSummary(BaseModel):
    name: str
    rows: int
    headers: list[str]
    score: int


class InspectResponse(BaseModel):
    sheets: list[SheetSummary]
    proposed_sheet: str | None = None
    column_mapping: dict[str, str | None] = {}
    confidence: float = 0.0
    source: str = "heuristic"      # llm | heuristic
    question: str | None = None    # non-null ⇒ the UI should ask the user
    options: list[str] = []


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


# ── Dictionary inspection (sheet + mapping proposal) ─────────────────────────

@router.post("/inspect_dictionary", response_model=InspectResponse)
async def inspect_dictionary(
    dictionary: UploadFile = File(..., description="Field dictionary (.xlsx)"),
) -> InspectResponse:
    """Inspect an uploaded workbook: list its sheets and let ONE stateless
    LLM agent propose which sheet is the field dictionary and how its
    columns map. When confidence is low the response carries a `question`
    for the chat UI; the user's click resolves it (no free-form parsing)."""
    _require_xlsx(dictionary)
    raw = await dictionary.read()
    inspection = dict_ai.inspect_workbook(raw)
    if not inspection.sheets:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Workbook has no sheets")

    proposal = dict_ai.propose_mapping(inspection, get_client())
    ask = proposal["question"] if (proposal.get("question")
                                   or inspection.ambiguous) else None
    if inspection.ambiguous and not ask:
        ask = "¿Qué hoja del Excel contiene el diccionario de campos?"
    return InspectResponse(
        sheets=[SheetSummary(name=s.name, rows=s.n_rows, headers=s.headers,
                             score=s.score) for s in inspection.sheets],
        proposed_sheet=proposal.get("sheet"),
        column_mapping=proposal.get("column_mapping", {}),
        confidence=float(proposal.get("confidence", 0.0)),
        source=proposal.get("source", "heuristic"),
        question=ask,
        options=proposal.get("options", []),
    )


def _require_xlsx(dictionary: UploadFile) -> None:
    if not dictionary.filename or not dictionary.filename.lower().endswith((".xlsx", ".xls")):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="dictionary must be an Excel file (.xlsx)")


# ── Generation ──────────────────────────────────────────────────────────────

@router.post("/generate", response_model=GenerateResponse)
async def generate_dqc(
    dictionary: UploadFile = File(..., description="Field dictionary (.xlsx)"),
    instructions: str = Form("", description="DQC rules, one per line"),
    instructions_file: UploadFile | None = File(
        None, description="Natural-language test list (.txt/.md/.csv/.xlsx)"),
    table_name: str = Form("mylib.ciclos_recuperacion"),
    sheet: str | None = Form(None, description="Workbook sheet holding the dictionary"),
    column_mapping: str | None = Form(None, description="JSON role->header mapping"),
    batch_size: int = Form(5, description="Instructions per generation agent"),
    infer_formats: bool = Form(True, description="LLM-infer missing field types"),
) -> GenerateResponse:
    """Generate DQCs from an Excel field dictionary + NL instructions.

    Context-window discipline: instructions are processed in batches and
    each batch is ONE fresh, stateless agent call that receives only the
    dictionary fields relevant to it (never more than the char budget).
    """
    from fastapi import HTTPException

    _require_xlsx(dictionary)
    raw = await dictionary.read()

    mapping: dict | None = None
    if column_mapping:
        try:
            parsed = json.loads(column_mapping)
            if isinstance(parsed, dict):
                mapping = parsed
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="column_mapping must be JSON")

    agents_used = 0
    mapping_source = "user" if (sheet or mapping) else "heuristic"
    if not sheet:
        inspection = dict_ai.inspect_workbook(raw)
        # ask only when there is a real choice: several sheets, at least
        # one plausible, and the heuristics cannot separate them (a single
        # or empty sheet falls through to parsing and its 400)
        best = inspection.best
        should_ask = (len(inspection.sheets) > 1 and inspection.ambiguous
                      and best is not None and best.score >= 2)
        if should_ask:
            # the UI must ask the user which sheet to use (422 carries the
            # same payload /inspect_dictionary would return)
            proposal = dict_ai.propose_mapping(inspection, get_client())
            raise HTTPException(status_code=422, detail={
                "needs_sheet_selection": True,
                "question": proposal.get("question")
                or "¿Qué hoja del Excel contiene el diccionario de campos?",
                "options": [s.name for s in inspection.sheets],
                "proposed_sheet": proposal.get("sheet"),
                "column_mapping": proposal.get("column_mapping", {}),
            })
        if mapping is None:
            proposal = dict_ai.propose_mapping(inspection, get_client())
            agents_used += 1
            sheet = proposal.get("sheet")
            mapping = proposal.get("column_mapping")
            mapping_source = proposal.get("source", "heuristic")
        else:
            sheet = inspection.best.name if inspection.best else None

    fields = dict_ai.parse_dictionary(raw, sheet=sheet, mapping=mapping)
    if not fields:
        raise HTTPException(
            status_code=400,
            detail="Could not read any fields from the Excel dictionary. "
                   "Expected columns like Field/Campo, Type/Tipo, Description/Descripcion.",
        )

    formats_inferred = 0
    if infer_formats and any(not f.type for f in fields):
        untyped = sum(1 for f in fields if not f.type)
        formats_inferred = dict_ai.infer_missing_formats(fields, get_client())
        agents_used += -(-untyped // 25)  # ceil(untyped / batch)

    # typed rules + uploaded test list, deduplicated in order
    instr_lines = _split_instructions(instructions)
    if instructions_file is not None and instructions_file.filename:
        file_rules = dict_ai.read_instructions_upload(
            instructions_file.filename, await instructions_file.read())
        if not file_rules:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read any rules from "
                       f"'{instructions_file.filename}' — expected one "
                       f"natural-language test per line/row.")
        seen = {ln for ln in instr_lines}
        instr_lines += [r for r in file_rules if r not in seen]
    if not instr_lines:
        raise HTTPException(
            status_code=400,
            detail="Provide instructions (textarea) or upload a test list")

    # one fresh agent per instruction batch, fed only the relevant fields
    dqcs: list[DQCItem] = []
    errors: list[str] = []
    batches = dict_ai.plan_batches(instr_lines, batch_size)
    offset = 0
    for batch in batches:
        relevant = dict_ai.select_relevant_fields(fields, batch)
        dict_text, sent = dict_ai.fields_to_text(relevant)
        user_prompt = (
            f"Tabla objetivo: {table_name}\n\n"
            f"DICCIONARIO DE CAMPOS ({sent} campos relevantes de {len(fields)}):\n"
            f"{dict_text}\n\n"
            f"INSTRUCCIONES DQC ({len(batch)} reglas):\n"
            + "\n".join(f"{offset + i + 1}. {ln}" for i, ln in enumerate(batch))
        )
        offset += len(batch)
        try:
            result = get_client().chat_json(
                system=DQC_SYSTEM_PROMPT, user=user_prompt, max_tokens=4096)
            agents_used += 1
        except Exception as exc:  # noqa: BLE001 — a batch failing must not kill the run
            logger.error("LLM batch failed: %s", exc)
            errors.append(str(exc))
            continue
        dqcs.extend(_parse_dqc_items(result))

    _dedupe_ids(dqcs)
    try:
        saved = _persist_dqc_items(dqcs)
        logger.info("persisted %d/%d DQCs", len(saved), len(dqcs))
    except Exception as exc:
        logger.warning("persist failed: %s", exc)

    summary = (
        f"Se generaron {len(dqcs)} DQCs a partir de {len(instr_lines)} "
        f"instrucción(es) y {len(fields)} campos del diccionario "
        f"(hoja '{sheet}', {len(batches)} agente(s) de generación"
        + (f", {formats_inferred} formato(s) inferido(s)" if formats_inferred else "")
        + ")."
    )
    if errors and not dqcs:
        summary = f"Error al generar DQCs: {errors[0]}"

    return GenerateResponse(
        dqcs=dqcs,
        dictionary_fields=len(fields),
        context_summary=summary,
        sheet_used=sheet or "",
        mapping_source=mapping_source,
        formats_inferred=formats_inferred,
        agents_used=agents_used,
    )


def _dedupe_ids(items: list[DQCItem]) -> None:
    seen: dict[str, int] = {}
    for item in items:
        base = item.dqc_id or "DQC"
        if base in seen:
            seen[base] += 1
            item.dqc_id = f"{base}_{seen[base]}"
        else:
            seen[base] = 1


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

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

from src.knowledge import get_client, get_inspect_client
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


PLAN_SYSTEM_PROMPT = """\
Eres un experto en calidad de datos para reporting regulatorio bancario.
Recibes una lista de instrucciones en lenguaje natural que describen reglas
DQC. Tu tarea es SEPARARLAS en reglas individuales (una instrucción puede
contener varias reglas) y proponer un plan de acción breve para cada una.

Responde SOLO con JSON:
{"plan": [{"id": 1, "regla": "<la regla, reformulada breve y clara>",
           "accion": "<qué control DQC se construirá: tipo de check, campos implicados>"}]}

No omitas ninguna regla. Mantén el orden recibido."""


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

    # sheet + column mapping is a small structured task — run it on the
    # lightweight inspection model (config: llm.inspect_model) so the
    # per-upload inspection stays fast
    proposal = dict_ai.propose_mapping(inspection, get_inspect_client())
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


def _parse_mapping_form(column_mapping: str | None) -> dict | None:
    if not column_mapping:
        return None
    from fastapi import HTTPException
    try:
        parsed = json.loads(column_mapping)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="column_mapping must be JSON")
    return parsed if isinstance(parsed, dict) else None


def _resolve_dictionary_context(
    raw: bytes,
    sheet: str | None,
    mapping: dict | None,
    infer_formats: bool,
) -> tuple[list, str | None, str, int, int]:
    """Shared /generate + /generate_stream preamble: resolve sheet/mapping
    (422 when the user must choose), parse the dictionary, infer formats.

    Returns (fields, sheet, mapping_source, formats_inferred, agents_used).
    """
    from fastapi import HTTPException

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
            proposal = dict_ai.propose_mapping(inspection, get_inspect_client())
            raise HTTPException(status_code=422, detail={
                "needs_sheet_selection": True,
                "question": proposal.get("question")
                or "¿Qué hoja del Excel contiene el diccionario de campos?",
                "options": [s.name for s in inspection.sheets],
                "proposed_sheet": proposal.get("sheet"),
                "column_mapping": proposal.get("column_mapping", {}),
            })
        if mapping is None:
            proposal = dict_ai.propose_mapping(inspection, get_inspect_client())
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

    return fields, sheet, mapping_source, formats_inferred, agents_used


def _collect_rules(instructions: str, file_name: str | None,
                   file_bytes: bytes | None) -> list[str]:
    """Typed rules + uploaded test list, deduplicated in order (400 if none)."""
    from fastapi import HTTPException

    instr_lines = _split_instructions(instructions)
    if file_name and file_bytes is not None:
        file_rules = dict_ai.read_instructions_upload(file_name, file_bytes)
        if not file_rules:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read any rules from "
                       f"'{file_name}' — expected one "
                       f"natural-language test per line/row.")
        seen = {ln for ln in instr_lines}
        instr_lines += [r for r in file_rules if r not in seen]
    if not instr_lines:
        raise HTTPException(
            status_code=400,
            detail="Provide instructions (textarea) or upload a test list")
    return instr_lines


def _run_generation_agent(batch: list[str], fields: list, table_name: str,
                          offset: int = 0) -> list[DQCItem]:
    """ONE fresh, stateless generation agent for a batch of rules, fed only
    the dictionary fields relevant to it. Raises on LLM failure."""
    relevant = dict_ai.select_relevant_fields(fields, batch)
    dict_text, sent = dict_ai.fields_to_text(relevant)
    user_prompt = (
        f"Tabla objetivo: {table_name}\n\n"
        f"DICCIONARIO DE CAMPOS ({sent} campos relevantes de {len(fields)}):\n"
        f"{dict_text}\n\n"
        f"INSTRUCCIONES DQC ({len(batch)} reglas):\n"
        + "\n".join(f"{offset + i + 1}. {ln}" for i, ln in enumerate(batch))
    )
    result = get_client().chat_json(
        system=DQC_SYSTEM_PROMPT, user=user_prompt, max_tokens=4096)
    return _parse_dqc_items(result)


def _plan_rules(instr_lines: list[str], client) -> list[dict]:
    """ONE planner agent splits the raw rules into an ordered action plan
    (one entry per DQC to build). Falls back to one item per input line."""
    fallback = [{"id": i + 1, "regla": ln,
                 "accion": "Generar el control DQC correspondiente"}
                for i, ln in enumerate(instr_lines)]
    user = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(instr_lines))
    try:
        result = client.chat_json(
            system=PLAN_SYSTEM_PROMPT,
            user=user[:dict_ai.PROMPT_CHAR_BUDGET],
            max_tokens=2048)
    except Exception as exc:  # noqa: BLE001 — planning is best-effort
        logger.warning("plan agent failed: %s", exc)
        return fallback
    raw_items = result.get("plan") if isinstance(result, dict) else None
    if not isinstance(raw_items, list):
        return fallback
    plan: list[dict] = []
    for entry in raw_items:
        if not isinstance(entry, dict):
            continue
        regla = str(entry.get("regla") or "").strip()
        if not regla:
            continue
        plan.append({"id": len(plan) + 1, "regla": regla,
                     "accion": str(entry.get("accion") or "").strip()})
    return plan or fallback


# ── Generation ──────────────────────────────────────────────────────────────

# TODO(streaming-reasoning): add a POST /generate_stream twin of this endpoint
# that streams the pipeline live to the chat UI over Server-Sent Events:
#   return StreamingResponse(gen(), media_type="text/event-stream")
#   (or the `sse-starlette` package for proper event framing/heartbeats)
# The generator should emit one `event: step` (e.g. "propose_mapping",
# "infer_formats", "batch 2/5") BEFORE each LLM call below, then relay the
# ("thinking", delta) / ("text", delta) events from llm_client.chat_stream()
# as `event: thinking` / `event: answer` SSE messages, and finish each step
# with `event: result` carrying the parsed JSON. Parse JSON ONLY from the
# text/done channel — never from thinking. The LLM call sites to wrap are:
#   1. dict_ai.propose_mapping(...)        (sheet + column mapping proposal)
#   2. dict_ai.infer_missing_formats(...)  (field type inference)
#   3. each batch agent in the `for batch in batches:` loop
# Keep this buffered /generate endpoint as-is for non-streaming clients.

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
    _require_xlsx(dictionary)
    raw = await dictionary.read()

    mapping = _parse_mapping_form(column_mapping)
    fields, sheet, mapping_source, formats_inferred, agents_used = (
        _resolve_dictionary_context(raw, sheet, mapping, infer_formats))

    file_bytes = None
    if instructions_file is not None and instructions_file.filename:
        file_bytes = await instructions_file.read()
    instr_lines = _collect_rules(
        instructions,
        instructions_file.filename if instructions_file else None,
        file_bytes)

    # one fresh agent per instruction batch, fed only the relevant fields
    dqcs: list[DQCItem] = []
    errors: list[str] = []
    batches = dict_ai.plan_batches(instr_lines, batch_size)
    offset = 0
    for batch in batches:
        try:
            items = _run_generation_agent(batch, fields, table_name, offset)
            agents_used += 1
        except Exception as exc:  # noqa: BLE001 — a batch failing must not kill the run
            logger.error("LLM batch failed: %s", exc)
            errors.append(str(exc))
            offset += len(batch)
            continue
        offset += len(batch)
        dqcs.extend(items)

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


# ── Plan-mode generation (SSE) ───────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/generate_stream")
async def generate_dqc_stream(
    dictionary: UploadFile = File(..., description="Field dictionary (.xlsx)"),
    instructions: str = Form("", description="DQC rules, one per line"),
    instructions_file: UploadFile | None = File(
        None, description="Natural-language test list (.txt/.md/.csv/.xlsx)"),
    table_name: str = Form("mylib.ciclos_recuperacion"),
    sheet: str | None = Form(None, description="Workbook sheet holding the dictionary"),
    column_mapping: str | None = Form(None, description="JSON role->header mapping"),
    infer_formats: bool = Form(True, description="LLM-infer missing field types"),
):
    """Plan-mode twin of /generate, streamed over Server-Sent Events.

    ONE planner agent first separates the instructions into an ordered JSON
    action plan (one entry per DQC to build); then one fresh generation
    agent runs per plan item so the UI can tick items off live, Claude-Code
    style. Event sequence::

        meta  {dictionary_fields, sheet_used, formats_inferred}
        plan  {items: [{id, regla, accion, estado: "pendiente"}]}
        item  {id, estado: "en_curso"}
        item  {id, estado: "completado", dqcs: [...]} | {id, estado: "error", error}
        ...                                  (item pair repeats per plan entry)
        done  {dqcs, context_summary, dictionary_fields, sheet_used,
               mapping_source, formats_inferred, agents_used}

    Pre-stream validation errors (bad file, ambiguous sheet) surface as the
    same 400/422 responses /generate raises.
    """
    from fastapi.responses import StreamingResponse

    _require_xlsx(dictionary)
    raw = await dictionary.read()

    mapping = _parse_mapping_form(column_mapping)
    fields, sheet, mapping_source, formats_inferred, agents_used = (
        _resolve_dictionary_context(raw, sheet, mapping, infer_formats))

    file_bytes = None
    if instructions_file is not None and instructions_file.filename:
        file_bytes = await instructions_file.read()
    instr_lines = _collect_rules(
        instructions,
        instructions_file.filename if instructions_file else None,
        file_bytes)

    def event_stream():
        agents = agents_used
        yield _sse("meta", {
            "dictionary_fields": len(fields),
            "sheet_used": sheet or "",
            "formats_inferred": formats_inferred,
        })

        plan = _plan_rules(instr_lines, get_client())
        agents += 1
        yield _sse("plan", {"items": [
            {**p, "estado": "pendiente"} for p in plan
        ]})

        dqcs: list[DQCItem] = []
        errors: list[str] = []
        for entry in plan:
            yield _sse("item", {"id": entry["id"], "estado": "en_curso"})
            try:
                items = _run_generation_agent(
                    [entry["regla"]], fields, table_name, entry["id"] - 1)
                agents += 1
            except Exception as exc:  # noqa: BLE001 — one rule failing must not kill the run
                logger.error("LLM plan item %d failed: %s", entry["id"], exc)
                errors.append(str(exc))
                yield _sse("item", {"id": entry["id"], "estado": "error",
                                    "error": str(exc)})
                continue
            dqcs.extend(items)
            yield _sse("item", {"id": entry["id"], "estado": "completado",
                                "dqcs": [i.model_dump() for i in items]})

        _dedupe_ids(dqcs)
        try:
            saved = _persist_dqc_items(dqcs)
            logger.info("persisted %d/%d DQCs", len(saved), len(dqcs))
        except Exception as exc:  # noqa: BLE001
            logger.warning("persist failed: %s", exc)

        summary = (
            f"Se generaron {len(dqcs)} DQCs a partir de un plan de "
            f"{len(plan)} regla(s) ({len(fields)} campos del diccionario, "
            f"hoja '{sheet}'"
            + (f", {formats_inferred} formato(s) inferido(s)" if formats_inferred else "")
            + ")."
        )
        if errors and not dqcs:
            summary = f"Error al generar DQCs: {errors[0]}"

        yield _sse("done", {
            "dqcs": [d.model_dump() for d in dqcs],
            "context_summary": summary,
            "dictionary_fields": len(fields),
            "sheet_used": sheet or "",
            "mapping_source": mapping_source,
            "formats_inferred": formats_inferred,
            "agents_used": agents,
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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

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
from . import dqc_react as react

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dqc", tags=["dqc"])


# ── Models ────────────────────────────────────────────────────────────────────

class DQCItem(BaseModel):
    dqc_id: str = ""
    prev_id: str = ""              # previous/official id given by the user
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


class EvalResult(BaseModel):
    check_id: str
    name: str
    prev_id: str | None = None
    ok: bool
    error: str = ""
    n_casos: int = 0
    columnas: list[str] = []
    ejemplos: list[dict] = []
    precision: float | None = None
    recall: float | None = None
    esperados: int | None = None


class EvaluateResponse(BaseModel):
    casos: int
    resultados: list[EvalResult]
    evaluados: int                 # queries that executed successfully
    fallidos: int                  # queries that errored on the cases
    precision_media: float | None = None
    recall_medio: float | None = None


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
                    rule_id=it.prev_id or None,
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
    from fastapi import HTTPException

    _require_xlsx(dictionary)
    raw = await dictionary.read()
    try:
        inspection = dict_ai.inspect_workbook(raw)
    except Exception as exc:  # noqa: BLE001 — unreadable workbook is a user error, not a 500
        raise HTTPException(
            status_code=400,
            detail=_workbook_read_error(dictionary.filename, exc))
    if not inspection.sheets:
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


def _require_xlsx(upload: UploadFile, what: str = "dictionary") -> None:
    if not upload.filename or not upload.filename.lower().endswith((".xlsx", ".xls")):
        from fastapi import HTTPException
        raise HTTPException(status_code=400,
                            detail=f"{what} must be an Excel file (.xlsx)")


def _workbook_read_error(filename: str | None, exc: Exception) -> str:
    """User-facing detail for a workbook openpyxl could not load."""
    name = (filename or "").lower()
    if name.endswith(".xls"):
        return ("El formato .xls antiguo no está soportado. Abre el fichero "
                "en Excel, guárdalo como .xlsx y vuelve a subirlo.")
    return (f"No se pudo leer el Excel ({exc.__class__.__name__}). Comprueba "
            "que es un .xlsx válido y que no está protegido con contraseña.")


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
    filename: str | None = None,
) -> tuple[list, str | None, str, int, int]:
    """Shared /generate + /generate_stream preamble: resolve sheet/mapping
    (422 when the user must choose), parse the dictionary, infer formats.

    Returns (fields, sheet, mapping_source, formats_inferred, agents_used).
    """
    from fastapi import HTTPException

    agents_used = 0
    mapping_source = "user" if (sheet or mapping) else "heuristic"
    if not sheet:
        try:
            inspection = dict_ai.inspect_workbook(raw)
        except Exception as exc:  # noqa: BLE001 — unreadable workbook is a user error
            raise HTTPException(
                status_code=400, detail=_workbook_read_error(filename, exc))
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

    try:
        fields = dict_ai.parse_dictionary(raw, sheet=sheet, mapping=mapping)
    except Exception as exc:  # noqa: BLE001 — unreadable workbook is a user error
        raise HTTPException(
            status_code=400, detail=_workbook_read_error(filename, exc))
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
        _resolve_dictionary_context(raw, sheet, mapping, infer_formats,
                                    dictionary.filename))

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
    data_file: UploadFile | None = File(
        None, description="Extracted cases Excel to execute the DQCs against"),
    table_name: str = Form("mylib.ciclos_recuperacion"),
    sheet: str | None = Form(None, description="Workbook sheet holding the dictionary"),
    column_mapping: str | None = Form(None, description="JSON role->header mapping"),
    infer_formats: bool = Form(True, description="LLM-infer missing field types"),
    value_grounding: bool = Form(
        False, description="Experimental: sample real domain values from the "
                           "cases Excel into the generation prompt"),
    semantic_judge: bool = Form(
        False, description="Experimental: LLM-as-judge semantic review of "
                           "each accepted query"),
):
    """Plan-mode ReAct generation, streamed over Server-Sent Events.

    ONE planner agent separates the instructions into an ordered action
    plan; then each plan item runs a fresh-context ReAct loop
    (sufficiency check → SAS generation → static+dynamic validation →
    correction, see ``dqc_react``) so the UI can tick items off live.
    Rules may carry a previous DQC id (``DQC_X: regla`` / ``[DQC_X] regla``).
    When ``data_file`` (extracted cases Excel) is uploaded, each accepted
    query is executed against it: example violating cases stream back, and
    a label column (DQC_ID/ID_DQC/…) plus previous ids yield precision and
    recall per DQC. Event sequence::

        meta  {dictionary_fields, sheet_used, formats_inferred, casos}
        plan  {items: [{id, regla, accion, prev_id, estado: "pendiente"}]}
        item  {id, estado: "en_curso", fase: "suficiencia"|"generacion"|"validacion", intento}
        item  {id, estado: "ambigua", falta, campos}
              | {id, estado: "completado", dqcs, validacion?}
              | {id, estado: "error", error}
        done  {dqcs, context_summary, ..., evaluacion?}

    Pre-stream validation errors (bad file, ambiguous sheet) surface as the
    same 400/422 responses /generate raises.
    """
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse

    _require_xlsx(dictionary)
    raw = await dictionary.read()

    mapping = _parse_mapping_form(column_mapping)
    fields, sheet, mapping_source, formats_inferred, agents_used = (
        _resolve_dictionary_context(raw, sheet, mapping, infer_formats,
                                    dictionary.filename))

    file_bytes = None
    if instructions_file is not None and instructions_file.filename:
        file_bytes = await instructions_file.read()
    instr_lines = _collect_rules(
        instructions,
        instructions_file.filename if instructions_file else None,
        file_bytes)

    cases = None
    if data_file is not None and data_file.filename:
        data_bytes = await data_file.read()
        try:
            cases = react.load_cases(data_bytes, table_name)
        except Exception as exc:  # noqa: BLE001 — unreadable cases file is a user error
            raise HTTPException(
                status_code=400,
                detail=_workbook_read_error(data_file.filename, exc))

    # previous DQC ids ("DQC_X: regla") are extracted BEFORE planning and
    # re-attached by position when the planner keeps one item per line
    prev_ids = []
    clean_lines = []
    for ln in instr_lines:
        pid, rest = react.split_prev_id(ln)
        prev_ids.append(pid)
        clean_lines.append(rest)

    def event_stream():
        agents = agents_used
        yield _sse("meta", {
            "dictionary_fields": len(fields),
            "sheet_used": sheet or "",
            "formats_inferred": formats_inferred,
            "casos": cases.n_rows if cases else 0,
        })

        plan = _plan_rules(clean_lines, get_client())
        agents += 1
        if len(plan) == len(clean_lines):
            for p, pid in zip(plan, prev_ids):
                p["prev_id"] = pid or ""
        else:
            for p in plan:
                p["prev_id"] = ""
        yield _sse("plan", {"items": [
            {**p, "estado": "pendiente"} for p in plan
        ]})

        dqcs: list[DQCItem] = []
        errors: list[str] = []
        ambiguas = 0
        precisions: list[float] = []
        recalls: list[float] = []
        comprobados = 0

        for entry in plan:
            eid = entry["id"]

            # ── 1. sufficiency: do we have everything, or is it ambiguous?
            yield _sse("item", {"id": eid, "estado": "en_curso",
                                "fase": "suficiencia"})
            suf = react.check_sufficiency(entry["regla"], fields, get_client())
            agents += 1
            if not suf["suficiente"]:
                ambiguas += 1
                yield _sse("item", {"id": eid, "estado": "ambigua",
                                    "falta": suf["falta"],
                                    "campos": suf["campos"]})
                continue

            # ── optional value grounding: sample real domain values once per
            # rule so the generator compares against values that exist
            valores = None
            if value_grounding and cases is not None:
                yield _sse("item", {"id": eid, "estado": "en_curso",
                                    "fase": "grounding"})
                valores = react.sample_values(
                    cases, suf["campos"] or [f.name for f in fields])

            # ── 2-4. generate SAS → validate → correct (fresh agent each try)
            items: list[DQCItem] = []
            validacion: dict | None = None
            feedback: list[str] | None = None
            last_errors: list[str] = []
            for intento in range(1, react.MAX_ATTEMPTS + 1):
                yield _sse("item", {"id": eid, "estado": "en_curso",
                                    "fase": "generacion", "intento": intento})
                try:
                    result = react.generate_sas(
                        entry["regla"], fields, table_name, get_client(),
                        campos=suf["campos"], feedback=feedback,
                        valores=valores)
                    agents += 1
                except Exception as exc:  # noqa: BLE001
                    last_errors = [str(exc)]
                    feedback = last_errors
                    continue
                items = _parse_dqc_items(result)
                if not items:
                    last_errors = ["la respuesta no contenía ningún DQC"]
                    feedback = last_errors
                    continue

                yield _sse("item", {"id": eid, "estado": "en_curso",
                                    "fase": "validacion", "intento": intento})
                val_errors = react.static_validate(
                    items[0].regla_sql, fields, table_name)
                validacion = {"estatica": "ok" if not val_errors else "error",
                              "ejecutada": False}
                if not val_errors and cases is not None:
                    run = react.run_query(cases, items[0].regla_sql, table_name)
                    if not run["ok"]:
                        val_errors = [run["error"]]
                    else:
                        comprobados += 1
                        validacion.update({
                            "ejecutada": True,
                            "n_casos": run["n_casos"],
                            "columnas": run["columnas"],
                            "ejemplos": run["ejemplos"],
                        })
                        m = react.metrics(cases, run["casos"],
                                          entry.get("prev_id"))
                        if m:
                            validacion.update(m)
                            precisions.append(m["precision"])
                            recalls.append(m["recall"])
                if val_errors:
                    last_errors = val_errors
                    feedback = val_errors
                    items = []
                    continue

                # ── optional semantic judge: the query runs, but does it
                # implement THIS rule? A rejection feeds the correction loop.
                if semantic_judge:
                    yield _sse("item", {"id": eid, "estado": "en_curso",
                                        "fase": "juicio", "intento": intento})
                    juicio = react.judge_dqc(
                        entry["regla"], items[0].regla_sql,
                        items[0].descripcion, items[0].condicion_error,
                        validacion.get("ejemplos"), get_client())
                    agents += 1
                    if not juicio["correcto"]:
                        last_errors = [f"juez semántico: {juicio['motivo']}"]
                        feedback = last_errors
                        items = []
                        continue
                    validacion["juez_ok"] = True
                    validacion["juez_motivo"] = juicio["motivo"]
                    validacion["juez_confianza"] = juicio["confianza"]
                break

            if not items:
                error = "; ".join(last_errors) or "sin resultado"
                errors.append(error)
                yield _sse("item", {"id": eid, "estado": "error",
                                    "error": f"No validado tras "
                                             f"{react.MAX_ATTEMPTS} intentos: "
                                             f"{error}"})
                continue

            for it in items:
                it.prev_id = entry.get("prev_id") or ""
            dqcs.extend(items)
            yield _sse("item", {"id": eid, "estado": "completado",
                                "dqcs": [i.model_dump() for i in items],
                                "validacion": validacion})

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
            + (f", {ambiguas} regla(s) ambigua(s)" if ambiguas else "")
            + ")."
        )
        if errors and not dqcs and not ambiguas:
            summary = f"Error al generar DQCs: {errors[0]}"

        evaluacion = None
        if cases is not None:
            evaluacion = {
                "casos": cases.n_rows,
                "tests_comprobados": comprobados,
                "ambiguas": ambiguas,
                "precision_media": round(sum(precisions) / len(precisions), 3)
                if precisions else None,
                "recall_medio": round(sum(recalls) / len(recalls), 3)
                if recalls else None,
            }

        yield _sse("done", {
            "dqcs": [d.model_dump() for d in dqcs],
            "context_summary": summary,
            "dictionary_fields": len(fields),
            "sheet_used": sheet or "",
            "mapping_source": mapping_source,
            "formats_inferred": formats_inferred,
            "agents_used": agents,
            "evaluacion": evaluacion,
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Cases evaluation of stored DQCs ──────────────────────────────────────────

@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_checks(
    data_file: UploadFile = File(..., description="Extracted cases Excel"),
    table_name: str = Form("mylib.ciclos_recuperacion"),
    status: str | None = Form(None, description="pending|validated|rejected; all visible if empty"),
) -> EvaluateResponse:
    """Execute the stored DQC queries against an extracted-cases Excel.

    No LLM involved: each check's SQL runs on the cases (in-memory SQLite);
    the response carries example violating rows per check and, when the
    check has a previous id (``rule_id``) matching the cases' DQC_ID label
    column, its precision/recall against the historical flags."""
    from fastapi import HTTPException

    _require_xlsx(data_file, what="data_file")
    raw = await data_file.read()
    try:
        cases = react.load_cases(raw, table_name)
    except Exception as exc:  # noqa: BLE001 — unreadable workbook is a user error
        raise HTTPException(
            status_code=400,
            detail=_workbook_read_error(data_file.filename, exc))

    conn = _db()
    try:
        checks = checks_db.list_checks(conn, status=status or None, visible=True)
    finally:
        conn.close()

    resultados: list[EvalResult] = []
    precisions: list[float] = []
    recalls: list[float] = []
    for c in checks:
        run = react.run_query(cases, c["sql"], table_name)
        if not run["ok"]:
            resultados.append(EvalResult(
                check_id=c["check_id"], name=c["name"],
                prev_id=c.get("rule_id"), ok=False, error=run["error"]))
            continue
        result = EvalResult(
            check_id=c["check_id"], name=c["name"],
            prev_id=c.get("rule_id"), ok=True,
            n_casos=run["n_casos"], columnas=run["columnas"],
            ejemplos=run["ejemplos"])
        m = react.metrics(cases, run["casos"], c.get("rule_id"))
        if m:
            result.precision = m["precision"]
            result.recall = m["recall"]
            result.esperados = m["esperados"]
            precisions.append(m["precision"])
            recalls.append(m["recall"])
        resultados.append(result)

    evaluados = sum(1 for r in resultados if r.ok)
    return EvaluateResponse(
        casos=cases.n_rows,
        resultados=resultados,
        evaluados=evaluados,
        fallidos=len(resultados) - evaluados,
        precision_media=round(sum(precisions) / len(precisions), 3)
        if precisions else None,
        recall_medio=round(sum(recalls) / len(recalls), 3)
        if recalls else None,
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

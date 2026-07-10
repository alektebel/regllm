"""DQC (Data Quality Check) generator router.

Orchestrates existing SAS lineage tools + regulation search + local LLM
to produce structured DQC checks for a given variable.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import uuid
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agent.tools import (
    _t_backtrace_sas_field,
    _t_get_field_definition,
    _t_get_sas_formula,
    _t_search_docs,
    _t_search_regulation,
    _t_search_regulation_semantic,
    _t_trace_dependencies,
)
from src.knowledge import get_client
from training.dq import checks_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dqc", tags=["dqc"])


# ── Validation pipeline models ──────────────────────────────────────────────

class CheckRecord(BaseModel):
    """One row of the validation table — the union of DQCItem + RL fields."""
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


def _db() -> sqlite3.Connection:
    """Per-call connection — SQLite objects are thread-bound."""
    conn = checks_db.connect()
    return conn


# Severity/category vocab translation: the LLM emits Spanish tokens from
# DQC_SYSTEM_PROMPT; normalise to the canonical English set the table CHECK
# constraints accept.
_SEV_MAP = {"bloqueante": "HIGH", "advertencia": "MED", "informativo": "LOW"}
_CAT_MAP = {
    "formula": "formula", "consistencia": "consistencia",
    "referencial": "referencial", "rango": "rango", "completitud": "completitud",
    # Coherence pipeline canonicals pass through unchanged:
    "cross_field": "cross_field", "cross_table": "cross_table",
    "cardinality": "cardinality",
}


def _persist_dqc_items(items: list[DQCItem], visible: bool = True) -> list[str]:
    """Save generated DQCs to the validation store as ``pending``."""
    ids: list[str] = []
    for it in items:
        sev = _SEV_MAP.get(it.severidad, it.severidad or "MED")
        cat = _CAT_MAP.get(it.tipo, it.tipo or "consistencia")
        try:
            cid = checks_db.insert_check(
                _db(),
                name=(it.dqc_id or "dqc").lower(),
                description=it.descripcion,
                severity=sev,
                category=cat,
                sql=it.regla_sql,
                visible=visible,
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
            logger.warning("DQC persist failed (check_id clash?): %s", exc)
    return ids

# ── Request / Response models ───────────────────────────────────────────────

class DQCRequest(BaseModel):
    message: str
    session_id: str = "default"


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


class RAGSource(BaseModel):
    document: str = ""
    heading: str = ""
    snippet: str = ""
    source_type: str = ""  # "regulation" | "regulation_semantic" | "definition" | "docs"


class DQCResponse(BaseModel):
    variable: str
    dqcs: list[DQCItem]
    context_summary: str = ""
    sources: list[RAGSource] = []


class SimpleDQCRequest(BaseModel):
    """Lighter-weight generation: skip the SAS-lineage / graph / semantic RAG
    context gathering entirely, and ground the model in ONE regulatory
    article the caller supplies directly (by paragraph number from the
    ingested EBA GL/2017/16 corpus, or by pasting the text). Built for
    local experimentation with a self-hosted model (e.g. REGLLM_LLM=gguf):
    a short, single-article prompt is far more tractable for a small local
    model than the full multi-source RAG context the main /generate
    endpoint assembles, and needs no embedding index to be built first.
    """
    expressions: str          # one or more NL data-quality rules, free text
    article_paragraph: int | None = None  # pull §N from data/regulation/eba_gl_2017_16_articles.json
    article_text: str | None = None       # or supply the article text directly
    session_id: str = "default"


class SimpleDQCResponse(BaseModel):
    dqcs: list[DQCItem]
    article_citation: str = ""
    article_text_used: str = ""
    context_summary: str = ""


# ── System prompt ───────────────────────────────────────────────────────────

DQC_SYSTEM_PROMPT = """\
Eres un experto en calidad de datos para reporting regulatorio bancario IRB. \
Generas controles DQC exhaustivos que cubren TODAS las dimensiones de validación \
relevantes para cada campo.

REGLAS OBLIGATORIAS:
- Genera AL MENOS un DQC por cada dimensión aplicable (ver lista abajo).
- Cada DQC debe involucrar RELACIONES entre campos — no generes checks de un \
  solo campo como "campo IS NULL" o "campo >= 0" salvo que el contexto \
  regulatorio lo exija explícitamente con un umbral concreto.
- Cada DQC debe estar FUNDAMENTADO en el contexto regulatorio proporcionado. \
  Cita artículos exactos. Si NO hay contexto regulatorio, indica \
  "Sin referencia regulatoria disponible" y NO inventes normas.
- Las reglas SQL usan la tabla `mylib.ciclos_recuperacion` y nombres de campo \
  exactos del contexto.

DIMENSIONES DE VALIDACIÓN — genera al menos un DQC por cada dimensión \
aplicable al campo objetivo:

1. **formula**: ¿Se calcula según la fórmula regulatoria? \
   (ej: ECL = PD_ESTIMADA × LGD_efectiva × EAD; \
   LGD_efectiva = max(LGD_ESTIMADA × (1+MOC), LGD_FLOOR_APLICADO))
2. **consistencia_cruzada**: ¿Es coherente con otros campos? \
   (ej: STAGE_IFRS9=3 → PD_ESTIMADA=1.0; CURE_FLAG=1 → STAGE_IFRS9 ≤ 2)
3. **suelo_regulatorio**: ¿Respeta los suelos mínimos por segmento/colateral? \
   (ej: PD ≥ 0.0005 si PROVISION_PERIOD_MONTHS>0; \
   LGD_ESTIMADA ≥ LGD_FLOOR_APLICADO según segmento+colateral)
4. **condicional_segmento**: Reglas que varían por SEGMENTO/COLATERAL_TIPO \
   (ej: PROVISION_PERIOD_MONTHS mínimo depende de SEGMENTO + fase de ciclo)
5. **coherencia_temporal**: ¿Es consistente con las fases del ciclo? \
   (ej: FASE_EXPANSION=1 → sólo compatible con STAGE_IFRS9=1; \
   VENTANA_CALIBRACION ≤ VENTANA_OBSERVACION)
6. **dependencia_condicional**: Campos que deben existir/tener valor bajo \
   ciertas condiciones (ej: CURE_FLAG=1 requiere DPDS=0 y PROVISION_PERIOD ≥ mínimo)
7. **integridad_liberacion**: Requisitos cruzados para liberar provisiones \
   (Art.23: CURE_FLAG=1 AND STAGE_IFRS9 ≤ 2 AND PROVISION_PERIOD ≥ mínimo \
   AND suelos históricos cumplidos)
8. **anomalia_logica**: Combinaciones de valores que son lógicamente imposibles \
   (ej: FASE_CRISIS=1 con DPDS=0; LGD_ESTIMADA > 1.0)

Responde con JSON: {"dqcs": [...]}. Cada objeto DQC:
{
  "dqc_id": "DQC_<VARIABLE>_<DIMENSION>_NNN",
  "variable": "<nombre>",
  "descripcion": "<qué verifica — sé específico, menciona los campos involucrados>",
  "tipo": "formula|consistencia|referencial|rango|completitud",
  "severidad": "bloqueante|advertencia|informativo",
  "regla_sql": "<SQL sobre mylib.ciclos_recuperacion — debe involucrar 2+ campos>",
  "condicion_error": "<cuándo falla — sé específico con los valores>",
  "campos_entrada": ["campo1", "campo2", "..."],
  "referencia_regulatoria": "<artículo exacto del contexto o 'Sin referencia'>",
  "umbral": "<valor si aplica>",
  "periodicidad": "diaria|mensual|trimestral",
  "justificacion": "<por qué este control importa, citando el contexto>"
}

Genera entre 5 y 12 DQCs por campo, cubriendo el máximo de dimensiones. \
Responde SOLO con JSON válido.
"""

# Leaner prompt for /dqc/generate/simple — no RAG context, just the caller's
# expressions plus (at most) one article. Deliberately short: a small local
# model has a much better shot at this than at the full multi-source prompt.
SIMPLE_DQC_SYSTEM_PROMPT = """\
Eres un experto en calidad de datos para reporting regulatorio bancario IRB.

Se te da:
1. Una o varias expresiones en lenguaje natural, cada una describiendo una \
regla de calidad de datos que debe cumplirse (p.ej. "la PD nunca debe superar \
el 100%", "toda hipoteca debe tener colateral tipo HIPOTECA").
2. Opcionalmente, el texto de UN ÚNICO artículo o párrafo regulatorio que \
puede (o no) fundamentar esas reglas.

Para CADA expresión, genera al menos un DQC (control de calidad de datos) \
que la implemente como una consulta SQL:
- La tabla es `mylib.ciclos_recuperacion`.
- Usa nombres de campo en MAYÚSCULAS, tal como aparecen en la expresión o se \
  infieran razonablemente de ella (p.ej. "la PD" → PD_ESTIMADA).
- Si se proporcionó un artículo y fundamenta la regla, cita EXACTAMENTE su \
  referencia en `referencia_regulatoria`. Si no se proporcionó artículo, o no \
  fundamenta la regla, escribe "Sin referencia regulatoria disponible" — \
  NO inventes artículos ni cites ninguno distinto del proporcionado.
- No omitas ninguna expresión: genera al menos un DQC por cada una.

Responde con JSON: {"dqcs": [...]}. Cada objeto DQC:
{
  "dqc_id": "DQC_<CAMPO>_<NNN>",
  "variable": "<campo principal>",
  "descripcion": "<qué verifica — sé específico>",
  "tipo": "formula|consistencia|referencial|rango|completitud",
  "severidad": "bloqueante|advertencia|informativo",
  "regla_sql": "<SQL sobre mylib.ciclos_recuperacion>",
  "condicion_error": "<cuándo falla>",
  "campos_entrada": ["campo1", "campo2", "..."],
  "referencia_regulatoria": "<cita exacta o 'Sin referencia regulatoria disponible'>",
  "umbral": "<valor si aplica>",
  "periodicidad": "diaria|mensual|trimestral",
  "justificacion": "<por qué, citando el artículo si aplica>"
}

Responde SOLO con JSON válido.
"""


# ── Helpers ─────────────────────────────────────────────────────────────────

_VAR_FIELD_RE = re.compile(r"\b([A-Z][A-Z0-9]*_[A-Z0-9_]+)\b")


def _load_known_fields() -> set[str]:
    """Load known field names from the regulation graph."""
    try:
        with open("data/regulation/graph.json") as f:
            g = json.load(f)
        return {
            n["id"].replace("field:", "")
            for n in g.get("nodes", [])
            if n.get("type") == "Field"
        }
    except Exception:
        return set()


_KNOWN_FIELDS: set[str] = _load_known_fields()


def _build_cross_field_context(variable: str) -> str:
    """Build cross-field relationships and regulatory section snippets for a field."""
    try:
        with open("data/regulation/graph.json") as f:
            g = json.load(f)
    except Exception:
        return ""

    field_id = f"field:{variable}"
    sections_node = {n["id"]: n for n in g["nodes"] if n["type"] == "Section"}

    # Find sections that mention this field
    relevant_sections: list[str] = []
    section_fields: dict[str, set[str]] = {}
    for e in g["edges"]:
        rel = e.get("relation") or e.get("type") or ""
        if rel in ("MENTIONS_FIELD", "JUSTIFIES"):
            sec = e["source"]
            fld = e["target"].replace("field:", "")
            section_fields.setdefault(sec, set()).add(fld)
            if e["target"] == field_id:
                relevant_sections.append(sec)

    # Co-mentioned fields (fields that appear in same regulatory sections)
    skip = {"CORP", "RETAIL", "MORTGAGE", "HIPOTECA", "PERSONAL", "FINANCIERO",
            "NINGUNA", "IFRS9", "STAGE"}
    co_fields: set[str] = set()
    for sec in relevant_sections:
        co_fields.update(section_fields.get(sec, set()) - {variable} - skip)

    parts: list[str] = []

    if co_fields:
        parts.append(
            f"CAMPOS RELACIONADOS con {variable} (aparecen en las mismas "
            f"secciones regulatorias): {', '.join(sorted(co_fields))}\n"
            f"Genera DQCs que verifiquen las relaciones entre {variable} y "
            f"estos campos."
        )

    # Include relevant section bodies (trimmed)
    seen_titles: set[str] = set()
    for sec_id in relevant_sections:
        sec = sections_node.get(sec_id, {})
        title = sec.get("title", "")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        body = sec.get("body", "")
        if body:
            parts.append(f"--- {title} ---\n{body[:600]}")

    return "\n\n".join(parts)

_STOP_WORDS = {"DQC", "DQCS", "PARA", "QUE", "CREA", "GENERA", "CREAR", "GENERAR",
               "COMPROBAR", "VERIFICAR", "ESTA", "ESTE", "SQL", "SAS", "BIEN",
               "CON", "CALCULA", "CALCULADA", "CALCULADO", "CORRECTAMENTE"}


def _fuzzy_match_field(text: str) -> str | None:
    """Match natural language to a known field name.

    Converts "provision period" → PROVISION_PERIOD → matches PROVISION_PERIOD_MONTHS.
    """
    normalized = re.sub(r"\s+", "_", text.strip()).upper()
    if len(normalized) < 3:
        return None
    # Exact match
    if normalized in _KNOWN_FIELDS:
        return normalized
    # Prefix match — score by coverage ratio (len(input)/len(field))
    # "PROVISION_PERIOD" → PROVISION_PERIOD_MONTHS (16/24=0.67) beats
    # PROVISION_PERIODO (16/18=0.89) because we want the canonical field
    candidates = [f for f in _KNOWN_FIELDS if f.startswith(normalized)]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        # Prefer the field whose remaining suffix looks like a unit/qualifier
        # (e.g. _MONTHS, _YEARS) over a translation variant (_PERIODO vs _PERIOD)
        return max(candidates, key=len)
    # Substring match
    candidates = [f for f in _KNOWN_FIELDS if normalized in f]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return max(candidates, key=len)
    return None


def _extract_variable(message: str) -> str | None:
    # 1. Exact underscore field names in the message (LGD_ESTIMADA, PD_ESTIMADA)
    for m in _VAR_FIELD_RE.finditer(message):
        candidate = m.group(1)
        if candidate in _KNOWN_FIELDS:
            return candidate

    # 2. Try to fuzzy-match multi-word spans against known fields
    #    Extract candidate spans: 2-4 consecutive words, skip stop words
    words = re.findall(r"[A-Za-záéíóúñÁÉÍÓÚÑ]+", message)
    for length in (3, 2, 4):
        for i in range(len(words) - length + 1):
            span = " ".join(words[i:i + length])
            match = _fuzzy_match_field(span)
            if match:
                return match
    # Single-word match
    for w in words:
        match = _fuzzy_match_field(w)
        if match and w.upper() not in _STOP_WORDS:
            return match

    # 3. Fallback: any uppercase token in message, skip stop words
    for m in _VAR_FIELD_RE.finditer(message):
        if m.group(1) not in _STOP_WORDS:
            return m.group(1)

    return None


def _gather_context(variable: str, session_id: str) -> dict[str, Any]:
    """Gather grounding context along two RAG channels:

    - **context RAG** (SAS lineage/formula/backtrace + ``docs`` BM25 over
      ``data/docs/**/*.md``): what the field *is* and where it comes from.
    - **regulation RAG**, in two complementary forms: ``regulation`` (graph
      traversal — precise but only covers fields already linked into the
      regulation knowledge graph) and ``regulation_semantic`` (embedding
      search over the chunked EBA GL/2017/16 PD & LGD guidelines — lower
      precision, much higher recall, works for any field or free-form
      question, no graph-linking step required).
    """
    ctx: dict[str, Any] = {}
    for label, fn in [
        ("formula", lambda: _t_get_sas_formula(variable, session_id)),
        ("dependencies", lambda: _t_trace_dependencies(variable, session_id, max_depth=3)),
        ("definition", lambda: _t_get_field_definition(variable)),
        ("regulation", lambda: _t_search_regulation(variable, k=3)),
        ("regulation_semantic", lambda: _t_search_regulation_semantic(variable, k=5)),
        ("backtrace", lambda: _t_backtrace_sas_field(variable, session_id)),
        ("docs", lambda: _t_search_docs(variable, k=3)),
    ]:
        try:
            ctx[label] = fn()
        except Exception as exc:
            logger.warning("DQC context gather %s failed: %s", label, exc)
            ctx[label] = {"error": str(exc)}
    return ctx


def _extract_sources(ctx: dict[str, Any]) -> list[RAGSource]:
    """Pull retrieved article/doc references from the gathered context."""
    sources: list[RAGSource] = []
    seen: set[str] = set()

    # Regulation evidence
    reg = ctx.get("regulation", {})
    for ev in reg.get("evidence", []) if isinstance(reg, dict) else []:
        key = ev.get("section_id", "")
        if key in seen:
            continue
        seen.add(key)
        sources.append(RAGSource(
            document=ev.get("document", ""),
            heading=ev.get("heading", ""),
            snippet=ev.get("snippet", "")[:300],
            source_type="regulation",
        ))

    # Regulation — semantic (embedding) hits over the chunked EBA GL/2017/16
    reg_sem = ctx.get("regulation_semantic", {})
    for hit in reg_sem.get("hits", []) if isinstance(reg_sem, dict) else []:
        key = f"semantic:{hit.get('chunk_id', '')}"
        if key in seen:
            continue
        seen.add(key)
        sources.append(RAGSource(
            document=hit.get("citation", hit.get("source_doc", "")),
            heading=f"{hit.get('section', '')} {hit.get('subsection', '')}".strip(),
            snippet=hit.get("text", "")[:300],
            source_type="regulation_semantic",
        ))

    # Field definition
    defn = ctx.get("definition", {})
    if isinstance(defn, dict):
        res = defn.get("result", {})
        if isinstance(res, dict) and res.get("found"):
            sources.append(RAGSource(
                document=res.get("path", ""),
                heading=res.get("heading", ""),
                snippet=res.get("body", "")[:300],
                source_type="definition",
            ))

    # Doc search hits
    docs = ctx.get("docs", {})
    for hit in docs.get("hits", []) if isinstance(docs, dict) else []:
        key = f"{hit.get('path', '')}:{hit.get('line_start', '')}"
        if key in seen:
            continue
        seen.add(key)
        sources.append(RAGSource(
            document=hit.get("path", ""),
            heading=hit.get("heading", ""),
            snippet=hit.get("snippet", "")[:300],
            source_type="docs",
        ))

    return sources


# ── Endpoint ────────────────────────────────────────────────────────────────

@router.post("/generate", response_model=DQCResponse)
def generate_dqc(req: DQCRequest) -> DQCResponse:
    variable = _extract_variable(req.message)
    if not variable:
        return DQCResponse(
            variable="(no se pudo extraer)",
            dqcs=[],
            context_summary="No se encontró un nombre de variable en el mensaje.",
        )

    ctx = _gather_context(variable, req.session_id)
    sources = _extract_sources(ctx)
    context_str = json.dumps(ctx, ensure_ascii=False, default=str)[:10000]

    source_note = ""
    if not sources:
        source_note = (
            "\n\nAVISO: No se encontró contexto regulatorio ni documentación "
            "para esta variable. NO inventes referencias regulatorias. "
            "Indica 'Sin referencia regulatoria disponible' en cada DQC."
        )

    cross_field_ctx = _build_cross_field_context(variable)

    user_prompt = (
        f"Variable objetivo: {variable}\n\n"
        f"Mensaje del usuario: {req.message}\n\n"
        f"{cross_field_ctx}\n\n"
        f"Contexto recopilado (fórmula SAS, dependencias, regulación):\n{context_str}"
        f"{source_note}"
    )

    try:
        result = get_client().chat_json(
            system=DQC_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=8192,
        )
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return DQCResponse(
            variable=variable,
            dqcs=[],
            context_summary=f"Error al generar DQCs: {exc}",
        )

    logger.info("LLM result keys: %s", list(result.keys()) if isinstance(result, dict) else type(result))
    if isinstance(result, dict) and "error" in result:
        logger.warning("LLM JSON parse issue: %s", result.get("raw", result.get("error", ""))[:500])
    # Accept alternative keys the LLM may produce (case-insensitive)
    raw_dqcs = _extract_dqc_list(result)
    dqcs = []
    for d in raw_dqcs:
        cleaned = {}
        for k, v in d.items():
            if k not in DQCItem.model_fields:
                continue
            if v is None:
                continue
            # Coerce to expected type — LLM may return numbers for string fields
            field_type = DQCItem.model_fields[k].annotation
            if field_type is str and not isinstance(v, str):
                v = str(v)
            cleaned[k] = v
        dqcs.append(DQCItem(**cleaned))

    # Auto-persist generated DQCs as `pending` so the validation UI can
    # surface them. The user then validates/rejects from the UI; the
    # dashboard unlocks once all visible pending checks are resolved.
    try:
        saved_ids = _persist_dqc_items(dqcs)
        logger.info("persisted %d/%d DQCs as pending", len(saved_ids), len(dqcs))
    except Exception as exc:
        # Persistence is best-effort — generation still succeeds even if
        # the validation store is unavailable.
        logger.warning("DQC persist failed: %s", exc)

    return DQCResponse(
        variable=variable,
        dqcs=dqcs,
        context_summary=f"Se generaron {len(dqcs)} DQCs para {variable} usando contexto SAS + regulación.",
        sources=sources,
    )


# ── Simple generation (no RAG — natural-language expressions + one article) ─

_EBA_GL_PATH = "data/regulation/eba_gl_2017_16_articles.json"


def _load_article_by_paragraph(n: int) -> tuple[str, str] | None:
    """Return (citation, text) for paragraph ``n`` of the ingested EBA
    GL/2017/16 corpus, or None if not found / the corpus is missing."""
    try:
        with open(_EBA_GL_PATH, encoding="utf-8") as f:
            records = json.load(f)
    except Exception:
        return None
    for rec in records:
        if rec.get("paragraph") == n:
            return f"EBA GL 2017/16 §{n}", rec.get("text", "")
    return None


@router.post("/generate/simple", response_model=SimpleDQCResponse)
def generate_dqc_simple(req: SimpleDQCRequest) -> SimpleDQCResponse:
    """Turn a batch of natural-language DQC expressions into SQL checks,
    grounded in at most one caller-supplied regulatory article — no
    SAS-lineage/graph/semantic RAG context gathering. See
    ``SimpleDQCRequest`` for why this exists."""
    article_citation = ""
    article_text = ""
    if req.article_text:
        article_text = req.article_text.strip()
        article_citation = "Artículo proporcionado por el usuario"
    elif req.article_paragraph is not None:
        found = _load_article_by_paragraph(req.article_paragraph)
        if found is None:
            return SimpleDQCResponse(
                dqcs=[],
                context_summary=(
                    f"No se encontró el párrafo {req.article_paragraph} en "
                    f"{_EBA_GL_PATH}."
                ),
            )
        article_citation, article_text = found

    user_prompt = (
        "Expresiones de calidad de datos (lenguaje natural, una o varias):\n"
        f"{req.expressions.strip()}\n\n"
    )
    if article_text:
        user_prompt += f"Artículo regulatorio ({article_citation}):\n{article_text}\n"
    else:
        user_prompt += (
            "No se proporcionó ningún artículo regulatorio. Indica "
            "'Sin referencia regulatoria disponible' en cada DQC.\n"
        )

    try:
        result = get_client().chat_json(
            system=SIMPLE_DQC_SYSTEM_PROMPT, user=user_prompt, max_tokens=4096,
        )
    except Exception as exc:
        logger.error("Simple DQC LLM call failed: %s", exc)
        return SimpleDQCResponse(
            dqcs=[], article_citation=article_citation, article_text_used=article_text,
            context_summary=f"Error al generar DQCs: {exc}",
        )

    raw_dqcs = _extract_dqc_list(result)
    dqcs: list[DQCItem] = []
    for d in raw_dqcs:
        cleaned = {}
        for k, v in d.items():
            if k not in DQCItem.model_fields or v is None:
                continue
            field_type = DQCItem.model_fields[k].annotation
            if field_type is str and not isinstance(v, str):
                v = str(v)
            cleaned[k] = v
        dqcs.append(DQCItem(**cleaned))

    try:
        saved_ids = _persist_dqc_items(dqcs)
        logger.info("persisted %d/%d simple DQCs as pending", len(saved_ids), len(dqcs))
    except Exception as exc:
        logger.warning("Simple DQC persist failed: %s", exc)

    n_expr = len([e for e in req.expressions.splitlines() if e.strip()]) or 1
    return SimpleDQCResponse(
        dqcs=dqcs, article_citation=article_citation, article_text_used=article_text,
        context_summary=f"Se generaron {len(dqcs)} DQCs a partir de {n_expr} expresión(es).",
    )


# ── Batch generation (SSE streaming) ───────────────────────────────────────


def _get_batch_fields() -> list[str]:
    """Return the fields from the regulation graph that have regulatory edges."""
    try:
        with open("data/regulation/graph.json") as f:
            g = json.load(f)
    except Exception:
        return sorted(_KNOWN_FIELDS)

    mentioned: set[str] = set()
    field_ids = {n["id"] for n in g.get("nodes", []) if n.get("type") == "Field"}
    for e in g.get("edges", []):
        rel = e.get("relation") or e.get("type") or ""
        if rel == "MENTIONS_FIELD" and e.get("target") in field_ids:
            mentioned.add(e["target"].replace("field:", ""))
    skip = {"CORP", "RETAIL", "MORTGAGE", "HIPOTECA", "PERSONAL", "FINANCIERO",
            "NINGUNA", "IFRS9", "STAGE"}
    return sorted(mentioned - skip)


def _extract_dqc_list(result: Any) -> list[dict]:
    """Extract the DQC list from LLM output, tolerating varied key names."""
    if not isinstance(result, dict):
        return []
    # Try common keys case-insensitively
    lower_map = {k.lower(): k for k in result}
    for candidate in ("dqcs", "dqc_list", "checks", "controles", "dqc"):
        real_key = lower_map.get(candidate)
        if real_key and isinstance(result[real_key], list):
            return result[real_key]
    # Fallback: find the first list value in the dict
    for v in result.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    return []


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/generate/batch/stream")
def batch_stream(session_id: str = "default"):
    """SSE endpoint: generates DQCs for all regulated fields with real-time LLM output."""
    fields = _get_batch_fields()

    def event_stream():
        yield _sse("start", {"total": len(fields), "fields": fields})
        total_generated = 0
        client = get_client()

        for i, field in enumerate(fields):
            yield _sse("field_start", {
                "field": field, "index": i, "total": len(fields),
            })

            try:
                # Gather context
                ctx = _gather_context(field, session_id)
                context_str = json.dumps(ctx, ensure_ascii=False, default=str)[:10000]
                sources = _extract_sources(ctx)
                source_note = ""
                if not sources:
                    source_note = (
                        "\n\nAVISO: No se encontró contexto regulatorio. "
                        "Indica 'Sin referencia regulatoria disponible' en cada DQC."
                    )

                cross_field_ctx = _build_cross_field_context(field)

                user_prompt = (
                    f"Variable objetivo: {field}\n\n"
                    f"Mensaje del usuario: Genera DQCs exhaustivos para {field} "
                    f"cubriendo todas las dimensiones de validación aplicables.\n\n"
                    f"{cross_field_ctx}\n\n"
                    f"Contexto recopilado:\n{context_str}{source_note}"
                )

                # Stream LLM tokens
                result = None
                for chunk in client.chat_json_stream(
                    system=DQC_SYSTEM_PROMPT,
                    user=user_prompt,
                    max_tokens=8192,
                ):
                    if isinstance(chunk, dict):
                        result = chunk
                    else:
                        yield _sse("token", {"field": field, "token": chunk})

                # Parse and persist
                count = 0
                if result and isinstance(result, dict) and "error" not in result:
                    raw_dqcs = _extract_dqc_list(result)
                    dqcs = []
                    for d in raw_dqcs:
                        cleaned = {}
                        for k, v in d.items():
                            if k not in DQCItem.model_fields or v is None:
                                continue
                            ft = DQCItem.model_fields[k].annotation
                            if ft is str and not isinstance(v, str):
                                v = str(v)
                            cleaned[k] = v
                        dqcs.append(DQCItem(**cleaned))
                    try:
                        _persist_dqc_items(dqcs)
                    except Exception as exc:
                        logger.warning("persist failed for %s: %s", field, exc)
                    count = len(dqcs)

                total_generated += count
                yield _sse("field_done", {
                    "field": field, "index": i,
                    "count": count, "total_generated": total_generated,
                })
            except Exception as exc:
                logger.error("Batch field %s failed: %s", field, exc)
                yield _sse("field_error", {"field": field, "error": str(exc)})

        yield _sse("done", {"total_generated": total_generated})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Validation pipeline endpoints ───────────────────────────────────────────

@router.get("/checks", response_model=list[CheckRecord])
def list_checks(
    status: str | None = None,
    visible: bool | None = True,
) -> list[CheckRecord]:
    """List checks for the validation UI.

    Default: visible only (ocultos never surface here). Pass ``visible=null``
    via the querystring (i.e. omit) to include them.
    """
    rows = checks_db.list_checks(_db(), status=status, visible=visible)
    return [CheckRecord(**r) for r in rows]


@router.get("/checks/counts")
def checks_counts() -> dict:
    """Status breakdown + ``dashboard_ready`` flag for the UI badge."""
    return checks_db.counts(_db())


@router.post("/checks/{check_id}/status", response_model=CheckRecord)
def update_check_status(check_id: str, body: StatusUpdate) -> CheckRecord:
    """Validate or reject a check. Idempotent."""
    if not checks_db.set_status(_db(), check_id, body.status):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"check_id {check_id} not found")
    row = checks_db.get_check(_db(), check_id)
    return CheckRecord(**row)


@router.delete("/checks/{check_id}")
def delete_check(check_id: str) -> dict:
    """Permanently delete a check."""
    conn = _db()
    cur = conn.execute("DELETE FROM checks WHERE check_id=?", (check_id,))
    conn.commit()
    if cur.rowcount == 0:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"check_id {check_id} not found")
    return {"deleted": check_id}


@router.get("/dashboard", response_model=DashboardResponse)
def dashboard() -> DashboardResponse:
    """Return the UNION-ALL dashboard query.

    Only populated when ``counts.dashboard_ready`` is true (no pending
    visible checks; at least one validated). The UI exposes the Copy
    button only in that state.
    """
    c = checks_db.counts(_db())
    sql = checks_db.build_dashboard_query(_db(), status="validated") if c["dashboard_ready"] else None
    validated = checks_db.export_validated(_db()) if c["dashboard_ready"] else []
    return DashboardResponse(
        ready=c["dashboard_ready"],
        pending_visible=c["pending_visible"],
        validated=c["validated"],
        rejected=c["rejected"],
        oculto=c["oculto"],
        sql=sql,
        checks=[CheckRecord(**r) for r in validated],
    )

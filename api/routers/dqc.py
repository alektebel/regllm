"""DQC (Data Quality Check) generator router.

Orchestrates existing SAS lineage tools + regulation search + local LLM
to produce structured DQC checks for a given variable.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from src.agent.tools import (
    _t_backtrace_sas_field,
    _t_get_field_definition,
    _t_get_sas_formula,
    _t_search_docs,
    _t_search_regulation,
    _t_trace_dependencies,
)
from src.knowledge import get_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dqc", tags=["dqc"])

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
    source_type: str = ""  # "regulation" | "definition" | "docs"


class DQCResponse(BaseModel):
    variable: str
    dqcs: list[DQCItem]
    context_summary: str = ""
    sources: list[RAGSource] = []


# ── System prompt ───────────────────────────────────────────────────────────

DQC_SYSTEM_PROMPT = """\
Eres un experto en calidad de datos para reporting regulatorio bancario \
(COREP/FINREP IRB). Generas controles DQC NO TRIVIALES que verifican la \
coherencia regulatoria y las relaciones entre campos.

IMPORTANTE — Reglas de generación:
- NO generes DQCs triviales como "campo IS NULL" o "campo >= 0" a menos que \
  tengan justificación regulatoria específica del contexto proporcionado.
- PRIORIZA controles que verifiquen RELACIONES entre campos (consistencia \
  cruzada, fórmulas, dependencias condicionales) basándote en el contexto.
- Cada DQC debe estar FUNDAMENTADO en el contexto regulatorio proporcionado. \
  Si el contexto incluye artículos regulatorios, cítalos exactamente. \
  Si NO hay contexto regulatorio, indica "Sin referencia regulatoria disponible" \
  y NO inventes artículos ni normas.
- Las reglas SQL deben usar la tabla `mylib.ciclos_recuperacion` y los nombres \
  de campo exactos del contexto.

Tipos de DQC valiosos (de mayor a menor prioridad):
1. **formula**: verificar que un campo se calcula según la fórmula regulatoria \
   (ej: ECL = PD_ESTIMADA × LGD_efectiva × EAD)
2. **consistencia**: verificar coherencia entre campos relacionados \
   (ej: si STAGE_IFRS9=3 → PD_ESTIMADA=1.0, si PROVISION_PERIOD_MONTHS>0 → suelos)
3. **referencial**: verificar que valores cumplen umbrales regulatorios específicos \
   (ej: PD mínima 0.05% según Circular 4/2022)
4. **rango**: solo si hay umbrales regulatorios explícitos en el contexto
5. **completitud**: solo si el contexto indica que el campo es obligatorio

Responde con JSON: {"dqcs": [...]}. Cada objeto DQC:
{
  "dqc_id": "DQC_<VARIABLE>_NNN",
  "variable": "<nombre>",
  "descripcion": "<qué verifica — sé específico>",
  "tipo": "formula|consistencia|referencial|rango|completitud",
  "severidad": "bloqueante|advertencia|informativo",
  "regla_sql": "<SQL sobre mylib.ciclos_recuperacion>",
  "condicion_error": "<cuándo falla>",
  "campos_entrada": ["campo1", "campo2"],
  "referencia_regulatoria": "<artículo exacto del contexto o 'Sin referencia'>",
  "umbral": "<valor si aplica>",
  "periodicidad": "diaria|mensual|trimestral",
  "justificacion": "<por qué este control importa, citando el contexto>"
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
    ctx: dict[str, Any] = {}
    for label, fn in [
        ("formula", lambda: _t_get_sas_formula(variable, session_id)),
        ("dependencies", lambda: _t_trace_dependencies(variable, session_id, max_depth=3)),
        ("definition", lambda: _t_get_field_definition(variable)),
        ("regulation", lambda: _t_search_regulation(variable, k=3)),
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
    context_str = json.dumps(ctx, ensure_ascii=False, default=str)[:6000]

    source_note = ""
    if not sources:
        source_note = (
            "\n\nAVISO: No se encontró contexto regulatorio ni documentación "
            "para esta variable. NO inventes referencias regulatorias. "
            "Indica 'Sin referencia regulatoria disponible' en cada DQC."
        )

    user_prompt = (
        f"Variable objetivo: {variable}\n\n"
        f"Mensaje del usuario: {req.message}\n\n"
        f"Contexto recopilado (fórmula SAS, dependencias, regulación):\n{context_str}"
        f"{source_note}"
    )

    try:
        result = get_client().chat_json(
            system=DQC_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=2048,
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
    raw_dqcs = result.get("dqcs", []) if isinstance(result, dict) else []
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

    return DQCResponse(
        variable=variable,
        dqcs=dqcs,
        context_summary=f"Se generaron {len(dqcs)} DQCs para {variable} usando contexto SAS + regulación.",
        sources=sources,
    )

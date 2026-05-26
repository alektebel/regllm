"""
Regulation assertion extractor.

Reads regulation text chunks from document_chunks, calls an LLM to parse
each article into discrete testable assertions, and stores them in
regulation_assertions.

The LLM output is structured JSON; we validate and normalise it before
storing so the downstream SQL generator can rely on well-formed input.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable

from src.assertions.models import (
    Assertion, CHECK_TYPES, MATERIALITIES, Threshold,
)

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Eres un analista experto en regulación bancaria. Tu tarea es extraer de textos \
regulatorios AFIRMACIONES COMPROBABLES — reglas que pueden verificarse \
directamente contra datos de un banco.

Una afirmación comprobable implica:
  - Un valor numérico, umbral o categoría que puede consultarse en una base de datos
  - Una condición de alcance (a qué operaciones/contratos se aplica)
  - Un resultado binario: cumple o no cumple

TIPOS válidos (check_type):
  range         → campo debe estar dentro de [min, max]
  existence     → algo debe existir o no existir (NOT NULL, NOT IN, etc.)
  classification → campo debe pertenecer a un conjunto de valores válidos
  time_limit    → acción debe completarse en N días/meses
  ratio         → cociente entre dos campos debe cumplir un umbral
  completeness  → todos los registros en scope deben tener un campo cubierto

MATERIALIDAD:
  critical → afecta capital regulatorio, ratios de solvencia
  high     → afecta parámetros IRB (PD, LGD, EAD), provisiones, clasificaciones
  medium   → afecta exactitud de reporting, límites operacionales
  low      → requisitos documentales, procedimentales menores

Responde SOLO con JSON válido, sin texto adicional:
{
  "assertions": [
    {
      "rule_text": "La PD estimada no puede ser inferior al 0,03% para exposiciones minoristas",
      "check_type": "range",
      "conditions": ["metodo = 'IRB'", "clase_exposicion = 'retail'"],
      "threshold": {"field_hint": "pd_estimada", "op": ">=", "value": 0.0003, "unit": ""},
      "materiality": "high",
      "notes": ""
    }
  ]
}

Si el texto no contiene afirmaciones comprobables devuelve {"assertions": []}.\
"""

_USER_TEMPLATE = """\
REGLAMENTO: {regulation_name}
ARTÍCULO: {article}

TEXTO:
{passage}
"""


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _call_groq(messages: list[dict], model: str | None = None, retries: int = 3) -> str:
    from groq import Groq  # type: ignore
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning("Groq attempt %d failed: %s — retrying in %ds", attempt + 1, e, wait)
            time.sleep(wait)
    return ""


def _call_ollama(messages: list[dict], model: str | None = None) -> str:
    import ollama  # type: ignore
    model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
    resp = ollama.chat(model=model, messages=messages, format="json",
                       options={"temperature": 0.0})
    return resp["message"]["content"]


def build_llm_fn(backend: str = "ollama", model: str | None = None) -> Callable[[list[dict]], str]:
    """Return a callable(messages) → str for the chosen backend."""
    if backend == "groq":
        return lambda msgs: _call_groq(msgs, model=model)
    elif backend == "ollama":
        return lambda msgs: _call_ollama(msgs, model=model)
    else:
        raise ValueError(f"Unknown backend: {backend!r} — use 'groq' or 'ollama'")


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = text.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _parse_llm_output(raw: str) -> list[dict]:
    """Parse LLM JSON output → list of raw assertion dicts."""
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        # Attempt to salvage partial JSON
        m = re.search(r'"assertions"\s*:\s*(\[.*?\])', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        logger.debug("Could not parse LLM output: %r", raw[:200])
        return []
    return data.get("assertions", [])


# ── Validation / normalisation ────────────────────────────────────────────────

def _normalise(
    raw: dict,
    regulation_id: str,
    regulation_name: str,
    article: str,
    passage: str,
) -> Assertion | None:
    rule_text  = (raw.get("rule_text") or "").strip()
    check_type = (raw.get("check_type") or "").strip().lower()
    if not rule_text or len(rule_text) < 10:
        return None
    if check_type not in CHECK_TYPES:
        logger.debug("Unknown check_type %r — skipping", check_type)
        return None

    materiality = (raw.get("materiality") or "medium").strip().lower()
    if materiality not in MATERIALITIES:
        materiality = "medium"

    conditions = raw.get("conditions") or []
    if isinstance(conditions, str):
        conditions = [conditions]
    conditions = [str(c).strip() for c in conditions if c]

    threshold = None
    if raw.get("threshold"):
        t = raw["threshold"]
        if isinstance(t, dict) and t.get("field_hint") and t.get("op") is not None and t.get("value") is not None:
            try:
                threshold = Threshold(
                    field_hint=str(t["field_hint"]),
                    op=str(t["op"]),
                    value=t["value"],
                    unit=str(t.get("unit", "")),
                )
            except Exception:
                pass

    return Assertion(
        regulation_id=regulation_id,
        regulation_name=regulation_name,
        article=article,
        passage=passage,
        rule_text=rule_text,
        check_type=check_type,
        conditions=conditions,
        threshold=threshold,
        materiality=materiality,
        notes=str(raw.get("notes", "")),
    )


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_conn():
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "regllm"),
        user=os.getenv("POSTGRES_USER", "regllm"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme"),
    )


def _store_assertions(assertions: list[Assertion], conn) -> int:
    stored = 0
    with conn.cursor() as cur:
        for a in assertions:
            row = a.to_db_row()
            cur.execute(
                """
                INSERT INTO regulation_assertions
                    (id, regulation_id, regulation_name, article, passage,
                     rule_text, check_type, conditions, threshold,
                     materiality, sql_template, notes, active, regulation_date)
                VALUES
                    (%(id)s, %(regulation_id)s, %(regulation_name)s, %(article)s, %(passage)s,
                     %(rule_text)s, %(check_type)s, %(conditions)s::jsonb, %(threshold)s::jsonb,
                     %(materiality)s, %(sql_template)s, %(notes)s, %(active)s, %(regulation_date)s)
                ON CONFLICT (id) DO NOTHING
                """,
                row,
            )
            if cur.rowcount:
                stored += 1
    conn.commit()
    return stored


def _fetch_chunks(conn, regulation_id: str | None, limit: int) -> list[dict]:
    """Pull regulation chunks from document_chunks, grouped by article."""
    with conn.cursor() as cur:
        if regulation_id:
            cur.execute(
                """
                SELECT texto, metadata
                FROM document_chunks
                WHERE metadata->>'documento_id' = %s
                   OR metadata->>'framework' = %s
                ORDER BY (metadata->>'chunk_index')::int NULLS LAST
                LIMIT %s
                """,
                (regulation_id, regulation_id, limit),
            )
        else:
            cur.execute(
                """
                SELECT texto, metadata
                FROM document_chunks
                WHERE metadata->>'articulo' IS NOT NULL
                ORDER BY metadata->>'documento_id', (metadata->>'chunk_index')::int NULLS LAST
                LIMIT %s
                """,
                (limit,),
            )
        return [{"texto": r[0], "metadata": r[1]} for r in cur.fetchall()]


# ── Public API ────────────────────────────────────────────────────────────────

class AssertionExtractor:
    """
    Extracts testable compliance assertions from regulation chunks.

    Usage:
        extractor = AssertionExtractor(llm_fn=build_llm_fn("groq"))
        assertions = extractor.extract_from_chunk(texto, metadata)
        extractor.run_from_db(regulation_id="crr_575_2013")
    """

    def __init__(self, llm_fn: Callable[[list[dict]], str]):
        self.llm_fn = llm_fn

    def extract_from_chunk(
        self,
        texto: str,
        metadata: dict,
    ) -> list[Assertion]:
        """Extract assertions from a single regulation text chunk."""
        regulation_id   = metadata.get("documento_id", metadata.get("framework", "unknown"))
        regulation_name = metadata.get("source", regulation_id)
        article         = metadata.get("articulo", "")

        if not texto or len(texto.strip()) < 50:
            return []

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(
                regulation_name=regulation_name,
                article=article or "(sin artículo)",
                passage=texto[:3000],
            )},
        ]

        try:
            raw_output = self.llm_fn(messages)
        except Exception as e:
            logger.warning("LLM call failed for %s %s: %s", regulation_id, article, e)
            return []

        raw_list = _parse_llm_output(raw_output)
        assertions = []
        for raw in raw_list:
            a = _normalise(raw, regulation_id, regulation_name, article, texto)
            if a is not None:
                assertions.append(a)

        if assertions:
            logger.debug("%s %s → %d assertions", regulation_id, article, len(assertions))
        return assertions

    def run_from_db(
        self,
        regulation_id: str | None = None,
        limit: int = 500,
        dry_run: bool = False,
        progress_cb: Callable[[int, int, int], None] | None = None,
    ) -> int:
        """
        Pull chunks from document_chunks, extract assertions, store results.

        Args:
            regulation_id: filter to one regulation (e.g. "crr_575_2013"); None = all
            limit:         max chunks to process
            dry_run:       extract but do not write to DB
            progress_cb:   called as (chunk_index, total_chunks, running_total)

        Returns:
            Total assertions stored (or would-store if dry_run).
        """
        conn = _get_conn()
        chunks = _fetch_chunks(conn, regulation_id, limit)
        logger.info("Processing %d chunks (regulation_id=%r)", len(chunks), regulation_id)

        total_stored = 0
        for i, chunk in enumerate(chunks):
            assertions = self.extract_from_chunk(chunk["texto"], chunk["metadata"])
            if not dry_run and assertions:
                total_stored += _store_assertions(assertions, conn)
            elif dry_run:
                for a in assertions:
                    logger.info("[dry-run] %s | %s | %s | %s",
                                a.regulation_id, a.article, a.check_type, a.rule_text[:80])
                total_stored += len(assertions)

            if progress_cb:
                progress_cb(i + 1, len(chunks), total_stored)

        conn.close()
        logger.info("Done — %d assertions %s from %d chunks",
                    total_stored, "would be stored" if dry_run else "stored", len(chunks))
        return total_stored

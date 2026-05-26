"""
SQL template generator for regulation assertions.

Given an assertion from regulation_assertions and schema context from context_chunks,
calls a local Ollama LLM (qwen2.5-coder) to generate a PostgreSQL compliance check
query.

The generated SQL must:
  - Return a scalar `n_failing` (count of rows that VIOLATE the rule, 0 = compliant)
  - Use table/column names drawn from the schema context (context_chunks)
  - Apply scope conditions from assertion.conditions
  - Include a header comment citing the source regulation

Usage:
    gen = SQLGenerator(llm_fn=build_llm_fn("ollama", "qwen2.5-coder:14b"))
    sql = gen.generate(assertion, conn)   # conn = psycopg2 connection to regllm DB
"""

from __future__ import annotations

import logging
import re
from typing import Callable

from src.assertions.models import Assertion

logger = logging.getLogger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Eres un experto en SQL para bases de datos bancarias de riesgo de crédito (IRB/IFRS9).
Tu tarea es generar queries SQL para verificar reglas de cumplimiento regulatorio.

REQUISITOS OBLIGATORIOS:
1. La query DEBE devolver exactamente una fila con la columna n_failing (INTEGER).
   n_failing = número de filas que INCUMPLEN la regla (0 = completamente conforme).
2. Usa SOLO los nombres de tabla y columna que aparecen en el CONTEXTO DE ESQUEMA.
3. Si el contexto es insuficiente, genera un SQL placeholder con /* REQUIERE_AJUSTE */
   indicando qué nombre de tabla o columna falta.
4. Dialecto PostgreSQL estándar.
5. Primera línea: comentario con la referencia regulatoria.

MAPA DE TIPOS → PATRÓN SQL:
  range          → WHERE campo < umbral  (o campo > umbral)
  existence      → WHERE campo IS NULL  (o NOT IN lista)
  classification → WHERE campo NOT IN ('val1', 'val2', ...)
  time_limit     → WHERE DATE_PART('day', col_fecha_fin - col_fecha_ini) > N
  ratio          → WHERE col_num::numeric / NULLIF(col_den::numeric, 0) [op] umbral
  completeness   → WHERE campo IS NULL

Responde SOLO con SQL válido, sin bloques markdown ni texto adicional."""

_USER_TEMPLATE = """\
AFIRMACIÓN REGULATORIA:
  Reglamento  : {regulation_name}
  Artículo    : {article}
  Regla       : {rule_text}
  Tipo check  : {check_type}
  Condiciones : {conditions}
  Umbral      : {threshold}

CONTEXTO DE ESQUEMA (tablas y campos disponibles en la BD del banco):
{schema_context}

Genera el SQL que devuelve n_failing = número de filas que INCUMPLEN esta regla.\
"""

_PLACEHOLDER_SQL = """\
-- {regulation_name} {article}: {rule_text_short}
-- REQUIERE_AJUSTE: no se encontró contexto de esquema suficiente para esta afirmación.
-- Ajusta <tabla> y <campo> con los nombres reales de la base de datos del banco.
SELECT COUNT(*) AS n_failing
FROM <tabla>           -- tabla con el campo relevante
WHERE <campo> {op} {value}  -- condición de incumplimiento
  -- condiciones de alcance: {conditions}
;
"""


# ── Schema context fetcher ─────────────────────────────────────────────────────

def _extract_field_hints(assertion: Assertion) -> list[str]:
    """Extract candidate column name keywords from an assertion."""
    hints = []

    if assertion.threshold and assertion.threshold.field_hint:
        hints.append(assertion.threshold.field_hint.upper())

    # Split conditions on common delimiters and grab identifiers
    for cond in assertion.conditions:
        for token in re.split(r"[\s=<>!(),]+", cond):
            if token and len(token) >= 2 and token.upper() == token:
                hints.append(token)
            elif token and len(token) >= 3:
                hints.append(token.upper())

    # Mine rule_text for obvious column name patterns (SNAKE_CASE words)
    for m in re.finditer(r"\b([A-Z][A-Z0-9_]{2,})\b", assertion.rule_text.upper()):
        hints.append(m.group(1))

    # Keep unique, preserve order
    seen: set[str] = set()
    out = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out[:8]


def fetch_schema_context(conn, assertion: Assertion, n: int = 8) -> str:
    """Search context_chunks for schema entries relevant to *assertion*.

    Returns a formatted string of matching column definitions.
    Falls back to empty string if context_chunks is empty or unavailable.
    """
    hints = _extract_field_hints(assertion)

    # Build a search query from hints + rule_text keywords
    search_text = " ".join(hints[:4]) + " " + assertion.rule_text[:100]

    rows: list[tuple[str, dict]] = []

    try:
        with conn.cursor() as cur:
            # 1. Direct metadata match on column_name (fast, highest precision)
            if hints:
                placeholders = ", ".join(["%s"] * len(hints))
                cur.execute(
                    f"""
                    SELECT content, metadata
                    FROM context_chunks
                    WHERE upper(metadata->>'column_name') IN ({placeholders})
                    ORDER BY (metadata->>'confidence') DESC
                    LIMIT %s
                    """,
                    [*hints, n],
                )
                rows = cur.fetchall()

            # 2. Full-text content search as fallback / complement
            if len(rows) < n // 2:
                cur.execute(
                    """
                    SELECT content, metadata
                    FROM context_chunks
                    WHERE content ILIKE %s
                    LIMIT %s
                    """,
                    (f"%{hints[0]}%" if hints else f"%{assertion.threshold.field_hint if assertion.threshold else ''}%", n),
                )
                extra = cur.fetchall()
                # deduplicate
                existing_content = {r[0] for r in rows}
                for row in extra:
                    if row[0] not in existing_content:
                        rows.append(row)
                        existing_content.add(row[0])

    except Exception as e:
        logger.warning("Schema context fetch failed: %s", e)
        return "(sin contexto de esquema disponible)"

    if not rows:
        return "(sin contexto de esquema — ejecuta ingest_db_context.py primero)"

    parts = []
    for content, _meta in rows[:n]:
        parts.append(content)

    return "\n\n".join(parts)


# ── SQL generation ─────────────────────────────────────────────────────────────

def _clean_sql(raw: str) -> str:
    """Strip markdown fences and leading/trailing whitespace."""
    text = raw.strip()
    for fence in ("```sql", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _is_plausible_sql(sql: str) -> bool:
    """Basic sanity check: must contain SELECT and n_failing."""
    upper = sql.upper()
    return "SELECT" in upper and "N_FAILING" in upper


def generate_sql(
    assertion: Assertion,
    schema_context: str,
    llm_fn: Callable[[list[dict]], str],
) -> str:
    """Call the LLM to generate a SQL template for *assertion*.

    Returns a SQL string.  Falls back to a placeholder if generation fails
    or the LLM output is unparseable.
    """
    threshold_str = ""
    if assertion.threshold:
        t = assertion.threshold
        threshold_str = f"campo={t.field_hint!r} | op={t.op} | valor={t.value} {t.unit}"

    user_msg = _USER_TEMPLATE.format(
        regulation_name=assertion.regulation_name,
        article=assertion.article,
        rule_text=assertion.rule_text,
        check_type=assertion.check_type,
        conditions=assertion.conditions or "(ninguna)",
        threshold=threshold_str or "(sin umbral numérico)",
        schema_context=schema_context or "(sin contexto)",
    )

    try:
        raw = llm_fn([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ])
        sql = _clean_sql(raw)
        if _is_plausible_sql(sql):
            logger.debug("Generated SQL for %s %s (%d chars)", assertion.regulation_id, assertion.article, len(sql))
            return sql
        logger.warning("LLM output did not look like valid SQL — using placeholder")
    except Exception as e:
        logger.warning("LLM call failed for %s %s: %s", assertion.regulation_id, assertion.article, e)

    # Fallback placeholder
    op = assertion.threshold.op if assertion.threshold else "?"
    value = assertion.threshold.value if assertion.threshold else "?"
    return _PLACEHOLDER_SQL.format(
        regulation_name=assertion.regulation_name,
        article=assertion.article,
        rule_text_short=assertion.rule_text[:60],
        op=op,
        value=value,
        conditions=", ".join(assertion.conditions) if assertion.conditions else "ninguna",
    )


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _store_sql_template(conn, assertion_id: str, sql: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE regulation_assertions SET sql_template = %s WHERE id = %s",
            (sql, assertion_id),
        )
    conn.commit()


def _fetch_assertions_without_sql(conn, regulation_id: str | None, limit: int) -> list[dict]:
    with conn.cursor() as cur:
        if regulation_id:
            cur.execute(
                """
                SELECT id, regulation_id, regulation_name, article, passage,
                       rule_text, check_type, conditions, threshold, materiality, notes
                FROM regulation_assertions
                WHERE sql_template IS NULL
                  AND active = true
                  AND regulation_id = %s
                ORDER BY materiality DESC, article
                LIMIT %s
                """,
                (regulation_id, limit),
            )
        else:
            cur.execute(
                """
                SELECT id, regulation_id, regulation_name, article, passage,
                       rule_text, check_type, conditions, threshold, materiality, notes
                FROM regulation_assertions
                WHERE sql_template IS NULL
                  AND active = true
                ORDER BY materiality DESC, article
                LIMIT %s
                """,
                (limit,),
            )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ── Main class ─────────────────────────────────────────────────────────────────

class SQLGenerator:
    """
    Generates SQL compliance check templates for regulation assertions.

    Usage:
        gen = SQLGenerator(llm_fn=build_llm_fn("ollama", "qwen2.5-coder:14b"))
        n = gen.run(conn, regulation_id="crr_575_2013", limit=50)
        print(f"Generated {n} SQL templates")
    """

    def __init__(self, llm_fn: Callable[[list[dict]], str]):
        self.llm_fn = llm_fn

    def generate(self, assertion: Assertion, conn) -> str:
        """Generate SQL for a single assertion, using schema context from conn."""
        schema_context = fetch_schema_context(conn, assertion)
        return generate_sql(assertion, schema_context, self.llm_fn)

    def run(
        self,
        conn,
        regulation_id: str | None = None,
        limit: int = 200,
        dry_run: bool = False,
        force: bool = False,
    ) -> int:
        """
        Generate SQL templates for assertions where sql_template IS NULL.

        Args:
            conn:          psycopg2 connection to the regllm database
            regulation_id: filter to one regulation; None = all
            limit:         max assertions to process
            dry_run:       print SQL but don't write to DB
            force:         also regenerate existing sql_templates (not yet implemented)

        Returns:
            Number of SQL templates generated (or would generate in dry_run).
        """
        rows = _fetch_assertions_without_sql(conn, regulation_id, limit)
        logger.info(
            "Generating SQL for %d assertion(s) (regulation_id=%r)",
            len(rows), regulation_id,
        )

        count = 0
        for row in rows:
            from src.assertions.models import Assertion, Threshold
            threshold = None
            if row.get("threshold"):
                t = row["threshold"]
                if isinstance(t, str):
                    import json
                    t = json.loads(t)
                if isinstance(t, dict) and t.get("field_hint"):
                    threshold = Threshold(
                        field_hint=t["field_hint"],
                        op=t.get("op", "="),
                        value=t.get("value", 0),
                        unit=t.get("unit", ""),
                    )
            conditions = row.get("conditions") or []
            if isinstance(conditions, str):
                import json
                conditions = json.loads(conditions)

            assertion = Assertion(
                regulation_id=row["regulation_id"],
                regulation_name=row["regulation_name"],
                article=row["article"],
                passage=row["passage"],
                rule_text=row["rule_text"],
                check_type=row["check_type"],
                conditions=conditions,
                threshold=threshold,
                materiality=row.get("materiality", "medium"),
                notes=row.get("notes", ""),
            )
            assertion._id = row["id"]

            schema_context = fetch_schema_context(conn, assertion)
            sql = generate_sql(assertion, schema_context, self.llm_fn)

            if dry_run:
                print(f"\n{'='*70}")
                print(f"[DRY-RUN] {assertion.regulation_id} | {assertion.article}")
                print(f"Rule: {assertion.rule_text[:80]}")
                print(f"SQL:\n{sql[:400]}")
            else:
                _store_sql_template(conn, assertion.id, sql)
                logger.info(
                    "  [%d/%d] %s %s → SQL stored (%d chars)",
                    count + 1, len(rows), assertion.regulation_id, assertion.article, len(sql),
                )

            count += 1

        return count

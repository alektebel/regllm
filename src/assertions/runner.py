"""
Assertion runner — executes sql_templates from regulation_assertions against
a target database (the bank's own data warehouse) and produces structured findings.

Architecture:
    regulation_assertions (sql_template filled) + target DB
        → execute SQL → AssertionResult
        → LLM finding writer → Finding dict

The target database is SEPARATE from the regllm metadata DB.
Configure via env vars TARGET_DB_* or pass a DSN string.

Usage:
    runner = AssertionRunner(
        target_dsn="postgresql://user:pass@host:5432/bank_dw",
        llm_fn=build_llm_fn("ollama", "llama3.3"),
    )
    findings = runner.run_all(assertions)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from src.assertions.models import Assertion

logger = logging.getLogger(__name__)

# ── Finding writer prompt ──────────────────────────────────────────────────────

_FINDING_SYSTEM = """\
Eres un auditor bancario especializado en validación de modelos IRB/IFRS9.
Redacta hallazgos de auditoría en español con voz técnica y directa.
Responde SOLO en JSON válido (sin markdown):
{
  "titulo": "Título breve del hallazgo (máx 80 chars)",
  "descripcion": "Descripción concisa del incumplimiento detectado con cifras concretas.",
  "impacto": "Impacto regulatorio o de capital (2-3 frases).",
  "recomendacion": "Acción correctiva recomendada (2-3 frases)."
}"""

_FINDING_USER = """\
REGLA REGULATORIA:
  Reglamento  : {regulation_name}
  Artículo    : {article}
  Regla       : {rule_text}
  Materialidad: {materiality}

RESULTADO DEL CHECK SQL:
  Filas incumplidoras: {n_failing:,}
  Total filas analizadas: {n_total:,}
  Tasa de incumplimiento: {rate:.2%}
  Muestra de IDs afectados: {sample_ids}

Redacta el hallazgo de auditoría.\
"""


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class AssertionResult:
    """Raw result from executing one sql_template."""
    assertion_id: str
    n_failing: int
    n_total: int          # may be 0 if query doesn't return a total
    sample_ids: list      # up to 20 sample failing IDs
    sql_used: str
    execution_time_s: float = 0.0
    error: str | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @property
    def rate(self) -> float:
        return self.n_failing / self.n_total if self.n_total else 0.0

    @property
    def compliant(self) -> bool:
        return not self.is_error and self.n_failing == 0


@dataclass
class Finding:
    assertion: Assertion
    result: AssertionResult
    titulo: str = ""
    descripcion: str = ""
    impacto: str = ""
    recomendacion: str = ""
    run_date: str = field(default_factory=lambda: date.today().isoformat())

    def to_dict(self) -> dict:
        return {
            "assertion_id":    self.assertion.id,
            "regulation_id":   self.assertion.regulation_id,
            "regulation_name": self.assertion.regulation_name,
            "article":         self.assertion.article,
            "rule_text":       self.assertion.rule_text,
            "check_type":      self.assertion.check_type,
            "materiality":     self.assertion.materiality,
            "n_failing":       self.result.n_failing,
            "n_total":         self.result.n_total,
            "rate":            round(self.result.rate, 6),
            "sample_ids":      self.result.sample_ids,
            "titulo":          self.titulo,
            "descripcion":     self.descripcion,
            "impacto":         self.impacto,
            "recomendacion":   self.recomendacion,
            "sql_used":        self.result.sql_used,
            "run_date":        self.run_date,
        }


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _target_conn(dsn: str | None):
    import psycopg2
    if dsn:
        return psycopg2.connect(dsn)
    return psycopg2.connect(
        host=os.getenv("TARGET_DB_HOST", os.getenv("POSTGRES_HOST", "localhost")),
        port=int(os.getenv("TARGET_DB_PORT", os.getenv("POSTGRES_PORT", "5432"))),
        dbname=os.getenv("TARGET_DB_NAME", os.getenv("POSTGRES_DB", "regllm")),
        user=os.getenv("TARGET_DB_USER", os.getenv("POSTGRES_USER", "regllm")),
        password=os.getenv("TARGET_DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", "changeme")),
    )


def _fetch_assertions_with_sql(conn, regulation_id: str | None, materiality: str | None, limit: int) -> list[dict]:
    clauses = ["sql_template IS NOT NULL", "active = true"]
    params: list = []

    if regulation_id:
        clauses.append("regulation_id = %s")
        params.append(regulation_id)
    if materiality:
        valid = {"low", "medium", "high", "critical"}
        mats = [m.strip() for m in materiality.split(",") if m.strip() in valid]
        if mats:
            clauses.append(f"materiality IN ({', '.join(['%s']*len(mats))})")
            params.extend(mats)

    where = " AND ".join(clauses)
    params.append(limit)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, regulation_id, regulation_name, article, passage,
                   rule_text, check_type, conditions, threshold, materiality,
                   notes, sql_template
            FROM regulation_assertions
            WHERE {where}
            ORDER BY CASE materiality
                WHEN 'critical' THEN 1 WHEN 'high' THEN 2
                WHEN 'medium'   THEN 3 WHEN 'low'  THEN 4
                ELSE 5 END, article
            LIMIT %s
            """,
            params,
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _row_to_assertion(row: dict) -> Assertion:
    from src.assertions.models import Assertion, Threshold
    threshold = None
    if row.get("threshold"):
        t = row["threshold"]
        if isinstance(t, str):
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
        conditions = json.loads(conditions)

    a = Assertion(
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
        sql_template=row.get("sql_template"),
    )
    a._id = row["id"]
    return a


# ── SQL execution ──────────────────────────────────────────────────────────────

def execute_assertion(assertion: Assertion, target_conn) -> AssertionResult:
    """Execute assertion.sql_template against target_conn.

    Expects the query to return a row with at least `n_failing`.
    Optionally also reads `n_total` and `sample_ids` if the query provides them.
    """
    sql = assertion.sql_template
    if not sql:
        return AssertionResult(
            assertion_id=assertion.id,
            n_failing=0, n_total=0, sample_ids=[],
            sql_used="(no sql_template)", error="sql_template is empty",
        )

    t0 = time.perf_counter()
    try:
        with target_conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            if row is None:
                raise ValueError("Query returned no rows")

            col_names = [d[0].lower() for d in cur.description]
            row_dict = dict(zip(col_names, row))

            n_failing = int(row_dict.get("n_failing", 0))
            n_total   = int(row_dict.get("n_total",   0))
            sample_ids = list(row_dict.get("sample_ids") or [])

        return AssertionResult(
            assertion_id=assertion.id,
            n_failing=n_failing,
            n_total=n_total,
            sample_ids=sample_ids[:20],
            sql_used=sql,
            execution_time_s=time.perf_counter() - t0,
        )
    except Exception as e:
        logger.warning("SQL execution failed for %s: %s", assertion.id, e)
        return AssertionResult(
            assertion_id=assertion.id,
            n_failing=0, n_total=0, sample_ids=[],
            sql_used=sql,
            execution_time_s=time.perf_counter() - t0,
            error=str(e),
        )


# ── Finding writer ─────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    text = raw.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {}


def write_finding(
    assertion: Assertion,
    result: AssertionResult,
    llm_fn: Callable[[list[dict]], str] | None,
) -> Finding:
    """Generate an auditor-voice finding for a failed assertion check."""
    finding = Finding(assertion=assertion, result=result)

    if llm_fn is None or result.n_total == 0:
        finding.titulo = f"[{assertion.materiality.upper()}] {assertion.regulation_name} {assertion.article}"
        finding.descripcion = (
            f"{result.n_failing:,} filas incumplen: {assertion.rule_text}"
        )
        finding.impacto = "Revisar con el equipo de riesgo."
        finding.recomendacion = "Corregir los parámetros afectados según la normativa citada."
        return finding

    try:
        raw = llm_fn([
            {"role": "system", "content": _FINDING_SYSTEM},
            {"role": "user",   "content": _FINDING_USER.format(
                regulation_name=assertion.regulation_name,
                article=assertion.article,
                rule_text=assertion.rule_text,
                materiality=assertion.materiality,
                n_failing=result.n_failing,
                n_total=result.n_total,
                rate=result.rate,
                sample_ids=result.sample_ids[:10] or "(no disponibles)",
            )},
        ])
        parsed = _parse_json(raw)
        finding.titulo         = parsed.get("titulo", "")
        finding.descripcion    = parsed.get("descripcion", "")
        finding.impacto        = parsed.get("impacto", "")
        finding.recomendacion  = parsed.get("recomendacion", "")
    except Exception as e:
        logger.warning("Finding writer failed for %s: %s", assertion.id, e)
        finding.titulo = f"{assertion.regulation_name} {assertion.article} — incumplimiento"
        finding.descripcion = f"{result.n_failing:,} filas incumplen: {assertion.rule_text}"

    return finding


# ── Main class ─────────────────────────────────────────────────────────────────

class AssertionRunner:
    """
    Executes SQL compliance checks from regulation_assertions against a target DB.

    Usage:
        runner = AssertionRunner(
            target_dsn="postgresql://...",
            llm_fn=build_llm_fn("ollama", "llama3.3"),
        )
        findings = runner.run_all(conn, regulation_id="crr_575_2013")
    """

    def __init__(
        self,
        target_dsn: str | None = None,
        llm_fn: Callable[[list[dict]], str] | None = None,
        materiality_filter: str | None = None,
    ):
        self.target_dsn = target_dsn
        self.llm_fn = llm_fn
        self.materiality_filter = materiality_filter

    def run_all(
        self,
        meta_conn,          # psycopg2 connection to regllm DB (reads assertions)
        regulation_id: str | None = None,
        limit: int = 500,
        dry_run: bool = False,
    ) -> list[Finding]:
        """
        Fetch assertions with sql_templates, execute them, write findings.

        Args:
            meta_conn:     connection to the regllm metadata DB (regulation_assertions)
            regulation_id: filter to one regulation; None = all
            limit:         max assertions to run
            dry_run:       show SQL without executing

        Returns:
            list of Finding objects (only for assertions with n_failing > 0)
        """
        rows = _fetch_assertions_with_sql(
            meta_conn, regulation_id, self.materiality_filter, limit
        )
        logger.info(
            "Running %d assertion(s) against target DB (regulation=%r)",
            len(rows), regulation_id,
        )

        if dry_run:
            for row in rows:
                print(f"\n{'='*70}")
                print(f"[DRY-RUN] {row['regulation_id']} | {row['article']}")
                print(f"SQL:\n{row['sql_template'][:500]}")
            return []

        target_conn = _target_conn(self.target_dsn)
        findings: list[Finding] = []
        errors = 0

        for i, row in enumerate(rows):
            assertion = _row_to_assertion(row)

            result = execute_assertion(assertion, target_conn)

            status = "ERROR" if result.is_error else ("FAIL" if result.n_failing else "OK")
            logger.info(
                "  [%d/%d] %s %s → %s (n_failing=%d, %.1fs)",
                i + 1, len(rows),
                assertion.regulation_id, assertion.article,
                status, result.n_failing, result.execution_time_s,
            )

            if result.is_error:
                errors += 1
                continue

            if result.n_failing > 0:
                finding = write_finding(assertion, result, self.llm_fn)
                findings.append(finding)

        target_conn.close()
        logger.info(
            "Done — %d assertion(s) checked | %d finding(s) | %d error(s)",
            len(rows), len(findings), errors,
        )
        return findings

    def build_report(self, findings: list[Finding], total_checked: int) -> dict:
        """Build a JSON-serialisable report from findings."""
        flagged = len(findings)
        return {
            "run_date":          date.today().isoformat(),
            "assertions_checked": total_checked,
            "assertions_flagged": flagged,
            "compliance_rate":   round(1 - flagged / total_checked, 4) if total_checked else 1.0,
            "findings":          [f.to_dict() for f in findings],
        }

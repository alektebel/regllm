"""Persistence for generated DQCs — the only module that talks to SQLite.

Split out of ``dqc.py`` so the route handlers stay transport-level: they call
``persist_dqc_items`` / ``save_failed_item`` / ``save_check_cases`` and never
open a connection or write SQL themselves.

Two tables are involved:
  * ``checks``           — the checks store proper (``training.dq.checks_db``).
  * ``check_eval_cases`` — the latest detected-cases payload and decision trace
    per check, feeding the UI's detail panel. Best-effort cache: a failure here
    is logged and swallowed, never surfaced to the caller.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone

from training.dq import checks_db

from .dqc_models import DQCItem

logger = logging.getLogger(__name__)

# The LLM emits Spanish severities/categories; the store speaks the canonical
# English set. Unmapped values fall back to a safe default rather than raising,
# so one odd label from the model never loses a whole generation run.
_SEV_MAP = {"bloqueante": "HIGH", "advertencia": "MED", "informativo": "LOW"}
_CAT_MAP = {
    "formula": "formula", "consistencia": "consistencia",
    "referencial": "referencial", "rango": "rango", "completitud": "completitud",
}

_EVAL_CASES_SCHEMA = """\
CREATE TABLE IF NOT EXISTS check_eval_cases (
    check_id     TEXT PRIMARY KEY,
    payload      TEXT NOT NULL,
    evaluated_at TEXT NOT NULL
)"""

_CASE_KEYS = ("n_casos", "columnas", "ejemplos", "precision", "recall",
              "esperados", "trace")


def connect() -> sqlite3.Connection:
    """Open the checks store (schema created/migrated on connect)."""
    return checks_db.connect()


def persist_dqc_items(items: list[DQCItem]) -> list[tuple[DQCItem, str]]:
    """Insert generated checks; returns the (item, check_id) pairs stored.

    A per-item IntegrityError is logged and skipped so one bad row cannot
    discard the rest of a generation run.
    """
    ids: list[tuple[DQCItem, str]] = []
    conn = connect()
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
                ids.append((it, cid))
            except sqlite3.IntegrityError as exc:
                logger.warning("DQC persist clash: %s", exc)
    finally:
        conn.close()
    return ids


def save_failed_item(*, rule: str, estado: str, motivo: str,
                     prev_id: str = "", eid: object = "",
                     campos: list[str] | None = None,
                     trace: list[dict] | None = None) -> str | None:
    """Store a rule that never produced a check (ambiguous or errored).

    These rows carry ``sql=None`` and a ``motivo``; the UI lists them in red so
    the user can read why generation failed and retry. Returns the check_id, or
    None when persisting failed (best-effort — never breaks a generation run).
    """
    try:
        conn = connect()
        try:
            cid = checks_db.insert_check(
                conn,
                rule_id=prev_id or None,
                name=(prev_id or f"regla_{eid}").lower(),
                description=rule,
                severity="informativo",
                category="consistencia",
                sql=None,
                status=estado,
                motivo=motivo,
                campos_entrada=campos or None,
            )
        finally:
            conn.close()
        save_check_cases([(cid, {"trace": trace or []})])
        return cid
    except Exception as exc:  # noqa: BLE001 — never break generation over this
        logger.warning("failed-item persist failed: %s", exc)
        return None


def save_check_cases(entries: list[tuple[str, dict]]) -> None:
    """Upsert the latest detected-cases payload (and trace) for each check_id."""
    if not entries:
        return
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn = connect()
    try:
        conn.execute(_EVAL_CASES_SCHEMA)
        for check_id, data in entries:
            payload = {k: data[k] for k in _CASE_KEYS if data.get(k) is not None}
            conn.execute(
                "INSERT OR REPLACE INTO check_eval_cases VALUES (?, ?, ?)",
                (check_id, json.dumps(payload, ensure_ascii=False), now))
        conn.commit()
    except Exception as exc:  # noqa: BLE001 — cases cache is best-effort
        logger.warning("saving eval cases failed: %s", exc)
    finally:
        conn.close()


def load_check_cases(check_id: str) -> dict:
    """Latest cases payload for one check; {"available": False} when absent."""
    conn = connect()
    try:
        conn.execute(_EVAL_CASES_SCHEMA)
        row = conn.execute(
            "SELECT payload, evaluated_at FROM check_eval_cases "
            "WHERE check_id = ?", (check_id,)).fetchone()
    finally:
        conn.close()
    if not row:
        return {"available": False}
    return {"available": True, "evaluated_at": row["evaluated_at"],
            **json.loads(row["payload"])}

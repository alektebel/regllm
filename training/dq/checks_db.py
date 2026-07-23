"""SQLite persistence layer for DQ checks (the validation pipeline).

Two producer paths feed this store:
  * ``/dqc/generate`` (LLM + RAG) — emits rich :class:`DQCItem`-shaped rows.
  * RL inference on coherence rules (``training/dq/*``) — emits leaner rows
    carrying a ``reward`` signal.

Both share one table. ``visible=False`` rows are the "ocultos" — they are
inserted straight into ``validated`` status and never surface in the
review UI, but they DO contribute to the final dashboard UNION ALL.

The dashboard contract: every check's ``sql`` must ``SELECT`` the column
named by :data:`PK_COLUMN` (``ID_CONTR_CICLO_LGD``) so the UNION ALL can
project a normalised violation rowset.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from training.dq.coherence_rules import PK_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = Path(os.getenv("REGLLM_CHECKS_DB", PROJECT_ROOT / "data" / "dq" / "checks.db"))

# Single source of truth for the checks table. Adding columns here is the
# only change needed to extend the schema (CREATE IF NOT EXISTS is
# idempotent on the column set we ship; for production migrations use
# Alembic — out of scope here, see README "Out of scope").
_SCHEMA = """
CREATE TABLE IF NOT EXISTS checks (
    check_id              TEXT PRIMARY KEY,
    rule_id               TEXT,
    name                  TEXT NOT NULL,
    description           TEXT,
    severity              TEXT NOT NULL CHECK (severity IN ('HIGH','MED','LOW','bloqueante','advertencia','informativo')),
    category              TEXT NOT NULL CHECK (category IN ('cross_field','cross_table','cardinality','formula','consistencia','referencial','rango','completitud')),
    sql                   TEXT,           -- null for failed/ambiguous items
    visible               INTEGER NOT NULL DEFAULT 1,
    status                TEXT NOT NULL DEFAULT 'pending'
                          CHECK (status IN ('pending','validated','rejected','ambigua','error')),
    reward                REAL,
    motivo                TEXT,            -- why generation failed (ambigua/error)
    -- Rich DQC metadata (nullable — populated by /dqc/generate, empty for RL-only)
    variable              TEXT,
    tipo                  TEXT,
    condicion_error       TEXT,
    campos_entrada        TEXT,           -- JSON array
    referencia_regulatoria TEXT,
    umbral                TEXT,
    periodicidad          TEXT,
    justificacion         TEXT,
    created_at            TEXT NOT NULL,
    validated_at          TEXT
);
CREATE INDEX IF NOT EXISTS idx_checks_status  ON checks(status);
CREATE INDEX IF NOT EXISTS idx_checks_visible ON checks(visible);
"""


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    # Lightweight migration for DBs created before the failed-item columns
    # existed (adding a column is safe; CHECK/nullable changes only apply to
    # freshly created tables — the demo uses an ephemeral DB, so it gets them).
    cols = {r[1] for r in conn.execute("PRAGMA table_info(checks)")}
    if "motivo" not in cols:
        conn.execute("ALTER TABLE checks ADD COLUMN motivo TEXT")
        conn.commit()
    return conn


# Severity / category normalisation — the LLM emits Spanish
# ("bloqueante"/"advertencia") and the RL pipeline emits English
# ("HIGH"/"MED"). Both are accepted by the CHECK constraints; we normalise
# to the English canonical form on read for the dashboard.
_SEVERITY_CANON = {
    "bloqueante": "HIGH", "HIGH": "HIGH",
    "advertencia": "MED", "MED": "MED",
    "informativo": "LOW", "LOW": "LOW",
}
_CATEGORY_CANON = {
    "cross_field": "cross_field", "cross_table": "cross_table",
    "cardinality": "cardinality",
    "formula": "formula", "consistencia": "consistencia",
    "referencial": "referencial", "rango": "rango",
    "completitud": "completitud",
}


def insert_check(
    conn: sqlite3.Connection,
    *,
    name: str,
    description: str,
    severity: str,
    category: str,
    sql: str | None = None,
    check_id: str | None = None,
    rule_id: str | None = None,
    visible: bool = True,
    reward: float | None = None,
    status: str = "pending",
    motivo: str | None = None,
    variable: str | None = None,
    tipo: str | None = None,
    condicion_error: str | None = None,
    campos_entrada: list[str] | None = None,
    referencia_regulatoria: str | None = None,
    umbral: str | None = None,
    periodicidad: str | None = None,
    justificacion: str | None = None,
) -> str:
    """Insert one check; returns its id. Idempotent on (rule_id, sql)
    when ``rule_id`` is set — re-inserting the same trained-check row is
    a no-op so re-running inference doesn't bloat the table."""
    cid = check_id or f"chk_{uuid.uuid4().hex[:12]}"
    if rule_id:
        existing = conn.execute(
            "SELECT check_id FROM checks WHERE rule_id=? AND sql=?",
            (rule_id, sql),
        ).fetchone()
        if existing:
            return existing["check_id"]
    campos_json = json.dumps(campos_entrada, ensure_ascii=False) if campos_entrada else None
    conn.execute(
        """INSERT INTO checks
           (check_id, rule_id, name, description, severity, category, sql,
            visible, status, reward, motivo, variable, tipo, condicion_error,
            campos_entrada, referencia_regulatoria, umbral, periodicidad,
            justificacion, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cid, rule_id, name, description, severity, category, sql,
         int(visible), status, reward, motivo, variable, tipo, condicion_error,
         campos_json, referencia_regulatoria, umbral, periodicidad,
         justificacion, _now()),
    )
    conn.commit()
    return cid


def set_status(conn: sqlite3.Connection, check_id: str, status: str) -> bool:
    """Returns True if a row was actually updated.

    A rejection is terminal. Reviewers must create a new proposal rather than
    validating a DQC they have rejected.
    """
    if status not in ("pending", "validated", "rejected"):
        raise ValueError(f"invalid status: {status}")
    validated_at = _now() if status == "validated" else None
    cur = conn.execute(
        "UPDATE checks SET status=?, validated_at=COALESCE(?, validated_at) "
        "WHERE check_id=? AND NOT (status='rejected' AND ?='validated')",
        (status, validated_at, check_id, status),
    )
    conn.commit()
    return cur.rowcount > 0


def list_checks(
    conn: sqlite3.Connection,
    *,
    status: str | None = None,
    visible: bool | None = None,
) -> list[dict]:
    q = "SELECT * FROM checks WHERE 1=1"
    params: list[Any] = []
    if status is not None:
        q += " AND status=?"; params.append(status)
    if visible is not None:
        q += " AND visible=?"; params.append(int(visible))
    q += " ORDER BY severity DESC, created_at DESC"
    return [_row_to_dict(r) for r in conn.execute(q, params).fetchall()]


def get_check(conn: sqlite3.Connection, check_id: str) -> dict | None:
    r = conn.execute("SELECT * FROM checks WHERE check_id=?", (check_id,)).fetchone()
    return _row_to_dict(r) if r else None


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Breakdown by status × visibility — drives the dashboard-ready flag."""
    out = {"pending_visible": 0, "validated": 0, "rejected": 0, "oculto": 0}
    for r in conn.execute(
        "SELECT status, visible, COUNT(*) AS n FROM checks GROUP BY status, visible"
    ):
        s, v, n = r["status"], r["visible"], r["n"]
        if s == "pending" and v:
            out["pending_visible"] += n
        if s == "validated":
            out["validated"] += n
            if not v:
                out["oculto"] += n
        if s == "rejected":
            out["rejected"] += n
    out["dashboard_ready"] = out["pending_visible"] == 0 and out["validated"] > 0
    return out


def build_dashboard_query(
    conn: sqlite3.Connection,
    *,
    status: str = "validated",
) -> str | None:
    """UNION ALL of every validated check's per-PK violations.

    Contract: each check's ``sql`` must ``SELECT`` :data:`PK_COLUMN`
    (``ID_CONTR_CICLO_LGD``). The wrapper projects that column as ``pk``
    so the dashboard rowset is uniform regardless of what each check
    selects alongside it.

    Returns ``None`` when there are no checks of the requested status.
    """
    rows = list_checks(conn, status=status)
    if not rows:
        return None
    parts: list[str] = []
    for r in rows:
        inner = r["sql"].strip().rstrip(";").strip()
        alias = "_" + re.sub(r"[^0-9A-Za-z_]", "_", r["check_id"])
        sev = _SEVERITY_CANON.get(r["severity"], r["severity"])
        name_lit = r["name"].replace("'", "''")
        check_id_lit = r["check_id"].replace("'", "''")
        parts.append(
            f"SELECT '{check_id_lit}' AS check_id, "
            f"'{name_lit}' AS check_name, "
            f"'{sev}' AS severity, "
            f"{alias}.\"{PK_COLUMN}\" AS pk "
            f"FROM ({inner}) AS {alias}"
        )
    return "\nUNION ALL\n".join(parts)


def export_validated(conn: sqlite3.Connection) -> list[dict]:
    """All validated checks as plain dicts — fuels the UI's copy-all button."""
    return list_checks(conn, status="validated")


def delete_check(conn: sqlite3.Connection, check_id: str) -> bool:
    cur = conn.execute("DELETE FROM checks WHERE check_id=?", (check_id,))
    conn.commit()
    return cur.rowcount > 0


# ── Helpers ─────────────────────────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    # Decode campos_entrada JSON for the API layer.
    raw = d.get("campos_entrada")
    if isinstance(raw, str) and raw:
        try:
            d["campos_entrada"] = json.loads(raw)
        except json.JSONDecodeError:
            d["campos_entrada"] = []
    elif d.get("campos_entrada") is None:
        d["campos_entrada"] = []
    return d

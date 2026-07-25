"""Unit tests for api/routers/dqc_store.py — the persistence layer.

Each test runs against its own temporary SQLite file (REGLLM_CHECKS_DB), so
they are isolated and leave nothing behind.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """dqc_store bound to a throwaway database."""
    monkeypatch.setenv("REGLLM_CHECKS_DB", str(tmp_path / "checks.db"))
    from training.dq import checks_db
    importlib.reload(checks_db)                 # re-read DEFAULT_DB_PATH
    from api.routers import dqc_store
    importlib.reload(dqc_store)
    yield dqc_store
    # restore module state for tests that follow
    monkeypatch.delenv("REGLLM_CHECKS_DB", raising=False)
    importlib.reload(checks_db)
    importlib.reload(dqc_store)


def _item(**kw):
    from api.routers.dqc_models import DQCItem
    base = dict(dqc_id="DQC_PD_001", descripcion="PD entre 0 y 1",
                severidad="bloqueante", tipo="rango",
                regla_sql="SELECT * FROM t WHERE PD > 1",
                campos_entrada=["PD"])
    base.update(kw)
    return DQCItem(**base)


# ── persisting generated checks ─────────────────────────────────────────────

def test_persist_returns_ids_and_maps_severity_and_category(store):
    from training.dq import checks_db
    saved = store.persist_dqc_items([_item()])
    assert len(saved) == 1
    _, cid = saved[0]

    conn = store.connect()
    try:
        row = checks_db.get_check(conn, cid)
    finally:
        conn.close()
    assert row["severity"] == "HIGH"        # bloqueante → HIGH
    assert row["category"] == "rango"
    assert row["status"] == "pending"
    assert row["campos_entrada"] == ["PD"]  # JSON round-trip


def test_persist_falls_back_for_unknown_severity(store):
    from training.dq import checks_db
    _, cid = store.persist_dqc_items([_item(severidad="", tipo="")])[0]
    conn = store.connect()
    try:
        row = checks_db.get_check(conn, cid)
    finally:
        conn.close()
    assert row["severity"] == "MED" and row["category"] == "consistencia"


def test_persist_empty_list_is_a_noop(store):
    assert store.persist_dqc_items([]) == []


# ── failed / ambiguous items ────────────────────────────────────────────────

@pytest.mark.parametrize("estado", ["ambigua", "error"])
def test_failed_item_is_stored_with_reason_and_no_sql(store, estado):
    from training.dq import checks_db
    cid = store.save_failed_item(
        rule="El colateral debe estar correctamente informado",
        estado=estado, motivo="Falta el criterio concreto",
        prev_id="", eid=3, campos=["COLATERAL"], trace=[{"paso": "suficiencia"}])
    assert cid

    conn = store.connect()
    try:
        row = checks_db.get_check(conn, cid)
    finally:
        conn.close()
    assert row["status"] == estado
    assert row["sql"] is None                       # nothing was generated
    assert row["motivo"] == "Falta el criterio concreto"
    assert row["description"].startswith("El colateral")


def test_failed_item_trace_is_retrievable_for_the_tree(store):
    """The detail panel renders its decision tree from this payload."""
    trace = [{"paso": "suficiencia", "resultado": "no", "detalle": "ambigua"}]
    cid = store.save_failed_item(rule="r", estado="ambigua", motivo="m",
                                 trace=trace)
    cases = store.load_check_cases(cid)
    assert cases["available"] is True
    assert cases["trace"] == trace


def test_failed_item_names_itself_from_the_rule_id_when_present(store):
    from training.dq import checks_db
    cid = store.save_failed_item(rule="r", estado="error", motivo="m",
                                 prev_id="DQC_ECL_003")
    conn = store.connect()
    try:
        row = checks_db.get_check(conn, cid)
    finally:
        conn.close()
    assert row["name"] == "dqc_ecl_003"


def test_failed_items_appear_in_list_checks(store):
    """They must be listable — that is what puts them (red) in the panel."""
    from training.dq import checks_db
    store.persist_dqc_items([_item()])
    store.save_failed_item(rule="r", estado="ambigua", motivo="m")
    conn = store.connect()
    try:
        statuses = {c["status"] for c in checks_db.list_checks(conn)}
    finally:
        conn.close()
    assert {"pending", "ambigua"} <= statuses


# ── cases / trace cache ─────────────────────────────────────────────────────

def test_load_check_cases_absent_returns_unavailable(store):
    assert store.load_check_cases("chk_does_not_exist") == {"available": False}


def test_save_check_cases_upserts_the_latest_payload(store):
    store.save_check_cases([("chk_1", {"n_casos": 1, "columnas": ["PD"]})])
    store.save_check_cases([("chk_1", {"n_casos": 7, "columnas": ["PD", "EAD"]})])
    got = store.load_check_cases("chk_1")
    assert got["n_casos"] == 7 and got["columnas"] == ["PD", "EAD"]
    assert got["evaluated_at"]


def test_save_check_cases_drops_none_values(store):
    store.save_check_cases([("chk_2", {"n_casos": 2, "precision": None})])
    assert "precision" not in store.load_check_cases("chk_2")


def test_save_check_cases_empty_is_a_noop(store):
    store.save_check_cases([])          # must not raise


# ── schema migration ────────────────────────────────────────────────────────

def test_motivo_column_is_added_to_a_preexisting_database(tmp_path, monkeypatch):
    """A DB created before the failed-item work must gain `motivo` on connect,
    because CREATE TABLE IF NOT EXISTS alone would silently skip it."""
    import sqlite3

    db = tmp_path / "old.db"
    con = sqlite3.connect(str(db))
    con.execute("""CREATE TABLE checks (
        check_id TEXT PRIMARY KEY, rule_id TEXT, name TEXT NOT NULL,
        description TEXT, severity TEXT NOT NULL, category TEXT NOT NULL,
        sql TEXT NOT NULL, visible INTEGER NOT NULL DEFAULT 1,
        status TEXT NOT NULL DEFAULT 'pending', reward REAL,
        variable TEXT, tipo TEXT, condicion_error TEXT, campos_entrada TEXT,
        referencia_regulatoria TEXT, umbral TEXT, periodicidad TEXT,
        justificacion TEXT, created_at TEXT NOT NULL, validated_at TEXT)""")
    con.commit()
    con.close()

    monkeypatch.setenv("REGLLM_CHECKS_DB", str(db))
    from training.dq import checks_db
    importlib.reload(checks_db)
    try:
        conn = checks_db.connect()
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(checks)")}
        finally:
            conn.close()
        assert "motivo" in cols
    finally:
        monkeypatch.delenv("REGLLM_CHECKS_DB", raising=False)
        importlib.reload(checks_db)

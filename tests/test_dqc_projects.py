"""Per-project scoping of stored DQCs (project_id column + API filters)."""

from __future__ import annotations

import os
import sqlite3

import pytest

os.environ.setdefault("REGLLM_LLM", "stub")

from fastapi.testclient import TestClient  # noqa: E402

import api.routers.dqc as dqc_router  # noqa: E402
from api.main import app  # noqa: E402
from training.dq import checks_db  # noqa: E402


@pytest.fixture()
def isolated_checks_db(tmp_path, monkeypatch):
    db_path = tmp_path / "checks.db"
    real_connect = checks_db.connect

    def _connect(path=None):
        return real_connect(db_path if path is None else path)

    monkeypatch.setattr(dqc_router.checks_db, "connect", _connect)
    return db_path


@pytest.fixture()
def client(isolated_checks_db):
    return TestClient(app)


def _insert(name, project_id=None, status="pending"):
    conn = checks_db.connect()
    try:
        return checks_db.insert_check(
            conn, name=name, description="d", severity="HIGH", category="rango",
            sql="SELECT 1", status=status, project_id=project_id)
    finally:
        conn.close()


def test_list_checks_filters_by_project(client, isolated_checks_db):
    _insert("a", project_id="prj_1")
    _insert("b", project_id="prj_2")
    _insert("legacy")                      # pre-projects row

    all_names = [c["name"] for c in client.get("/dqc/checks").json()]
    assert sorted(all_names) == ["a", "b", "legacy"]

    p1 = [c["name"] for c in client.get("/dqc/checks?project_id=prj_1").json()]
    assert p1 == ["a"]
    p2 = [c["name"] for c in client.get("/dqc/checks?project_id=prj_2").json()]
    assert p2 == ["b"]


def test_unscoped_checks_are_not_leaked_into_a_project(client, isolated_checks_db):
    _insert("legacy")
    assert client.get("/dqc/checks?project_id=prj_1").json() == []


def test_migration_adds_project_id_to_an_existing_db(tmp_path):
    """A DB created before the column existed upgrades on connect()."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE checks (check_id TEXT PRIMARY KEY, rule_id TEXT,"
        " name TEXT NOT NULL, description TEXT, severity TEXT NOT NULL,"
        " category TEXT NOT NULL, sql TEXT NOT NULL,"
        " visible INTEGER NOT NULL DEFAULT 1,"
        " status TEXT NOT NULL DEFAULT 'pending', reward REAL, variable TEXT,"
        " tipo TEXT, condicion_error TEXT, campos_entrada TEXT,"
        " referencia_regulatoria TEXT, umbral TEXT, periodicidad TEXT,"
        " justificacion TEXT, created_at TEXT NOT NULL, validated_at TEXT)")
    conn.commit()
    conn.close()

    upgraded = checks_db.connect(db)
    cols = {r[1] for r in upgraded.execute("PRAGMA table_info(checks)")}
    assert "project_id" in cols
    # and it is usable straight away
    checks_db.insert_check(upgraded, name="x", description="", severity="LOW",
                           category="rango", sql="SELECT 1", project_id="prj_9")
    assert [c["name"] for c in checks_db.list_checks(upgraded, project_id="prj_9")] == ["x"]


def test_evaluate_can_be_scoped_to_a_project(client, isolated_checks_db):
    import io
    import openpyxl

    _insert("in_project", project_id="prj_1")
    _insert("other", project_id="prj_2")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["PD_ESTIMADA"])
    ws.append([1.5])
    buf = io.BytesIO()
    wb.save(buf)

    resp = client.post("/dqc/evaluate", data={"project_id": "prj_1"}, files={
        "data_file": ("casos.xlsx", buf.getvalue(),
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert resp.status_code == 200
    assert [r["name"] for r in resp.json()["resultados"]] == ["in_project"]

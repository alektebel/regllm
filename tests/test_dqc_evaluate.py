"""Tests for POST /dqc/evaluate — run stored DQCs against a cases Excel."""

from __future__ import annotations

import io
import os

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


def _insert(sql, rule_id=None, name="dqc_pd_001", status="pending"):
    conn = checks_db.connect()
    try:
        checks_db.insert_check(
            conn, name=name, description="d", severity="HIGH",
            category="rango", sql=sql, rule_id=rule_id, status=status)
    finally:
        conn.close()


def _cases_xlsx() -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["PD_ESTIMADA", "EAD_TOTAL", "DQC_ID"])
    ws.append([1.5, 100, "DQC_PD_001"])
    ws.append([0.5, 200, ""])
    ws.append([2.0, -1, "DQC_PD_001;DQC_EAD_001"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _post(client, status=None):
    data = {"table_name": "mylib.ciclos_recuperacion"}
    if status:
        data["status"] = status
    return client.post("/dqc/evaluate", data=data, files={
        "data_file": ("casos.xlsx", _cases_xlsx(),
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})


def test_evaluate_runs_checks_with_metrics(client, isolated_checks_db):
    _insert("SELECT * FROM mylib.ciclos_recuperacion WHERE PD_ESTIMADA > 1",
            rule_id="DQC_PD_001")
    resp = _post(client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["casos"] == 3
    assert body["evaluados"] == 1 and body["fallidos"] == 0
    r = body["resultados"][0]
    assert r["ok"] and r["n_casos"] == 2
    assert r["prev_id"] == "DQC_PD_001"
    assert r["precision"] == 1.0 and r["recall"] == 1.0
    assert len(r["ejemplos"]) == 2
    assert body["precision_media"] == 1.0


def test_evaluate_without_prev_id_gives_cases_but_no_metrics(
        client, isolated_checks_db):
    _insert("SELECT * FROM mylib.ciclos_recuperacion WHERE EAD_TOTAL < 0")
    body = _post(client).json()
    r = body["resultados"][0]
    assert r["ok"] and r["n_casos"] == 1
    assert r["precision"] is None
    assert body["precision_media"] is None


def test_evaluate_reports_broken_query(client, isolated_checks_db):
    _insert("SELECT * FROM mylib.ciclos_recuperacion WHERE NO_EXISTE > 1")
    body = _post(client).json()
    assert body["fallidos"] == 1
    assert not body["resultados"][0]["ok"]
    assert "falla al ejecutarse" in body["resultados"][0]["error"]


def test_evaluate_status_filter(client, isolated_checks_db):
    _insert("SELECT * FROM mylib.ciclos_recuperacion WHERE PD_ESTIMADA > 1",
            name="a", status="pending")
    _insert("SELECT * FROM mylib.ciclos_recuperacion WHERE EAD_TOTAL < 0",
            name="b", status="validated")
    body = _post(client, status="validated").json()
    assert [r["name"] for r in body["resultados"]] == ["b"]


def test_evaluate_no_checks_is_empty_not_error(client, isolated_checks_db):
    body = _post(client).json()
    assert body["resultados"] == [] and body["evaluados"] == 0


def test_evaluate_persists_cases_for_sidebar(client, isolated_checks_db):
    _insert("SELECT * FROM mylib.ciclos_recuperacion WHERE PD_ESTIMADA > 1",
            rule_id="DQC_PD_001")
    check_id = _post(client).json()["resultados"][0]["check_id"]

    resp = client.get(f"/dqc/checks/{check_id}/cases")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["n_casos"] == 2
    assert body["precision"] == 1.0
    assert len(body["ejemplos"]) == 2
    assert body["evaluated_at"]


def test_check_cases_unavailable_before_any_evaluation(client,
                                                       isolated_checks_db):
    assert client.get("/dqc/checks/chk_nope/cases").json() == {
        "available": False}


def test_evaluate_corrupt_data_file_is_400(client, isolated_checks_db):
    resp = client.post("/dqc/evaluate", files={
        "data_file": ("casos.xlsx", b"not a zip",
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert resp.status_code == 400
    assert "No se pudo leer el Excel" in resp.json()["detail"]

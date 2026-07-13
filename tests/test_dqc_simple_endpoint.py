"""Tests for POST /dqc/generate — Excel dictionary + NL instructions."""

from __future__ import annotations

import io
import os
import sqlite3

import pytest

os.environ.setdefault("REGLLM_LLM", "stub")

from fastapi.testclient import TestClient  # noqa: E402

import api.routers.dqc as dqc_router  # noqa: E402
from api.main import app  # noqa: E402
from training.dq import checks_db  # noqa: E402


class _FakeClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.last_user = None

    def chat_json(self, system, user, **kwargs):
        self.last_user = user
        return self.payload


def _make_xlsx() -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Field", "Type", "Description", "Null", "Formula"])
    ws.append(["PD_ESTIMADA", "REAL", "Estimated PD", "no", ""])
    ws.append(["EAD_TOTAL", "REAL", "Total EAD", "no", "EAD_BALANCE + EAD_FUERA_BALANCE"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


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


def _stub_dqcs():
    return {
        "dqcs": [
            {
                "dqc_id": "DQC_PD_ESTIMADA_001",
                "variable": "PD_ESTIMADA",
                "descripcion": "PD no puede superar 1.0",
                "tipo": "rango",
                "severidad": "bloqueante",
                "regla_sql": "SELECT * FROM mylib.ciclos_recuperacion WHERE PD_ESTIMADA > 1.0",
                "condicion_error": "PD_ESTIMADA > 1.0",
                "campos_entrada": ["PD_ESTIMADA"],
                "referencia_regulatoria": "Sin referencia en diccionario",
                "umbral": "1.0",
                "periodicidad": "mensual",
                "justificacion": "La PD es una probabilidad",
            },
        ]
    }


def test_generate_from_excel_and_instructions(client, monkeypatch):
    fake = _FakeClient(_stub_dqcs())
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)
    monkeypatch.setattr(dqc_router, "get_inspect_client", lambda: fake)

    resp = client.post(
        "/dqc/generate",
        data={"instructions": "La PD nunca debe superar el 100%"},
        files={"dictionary": ("dict.xlsx", _make_xlsx(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dictionary_fields"] == 2
    assert len(body["dqcs"]) == 1
    assert "PD_ESTIMADA" in fake.last_user
    assert "La PD nunca debe superar" in fake.last_user


def test_rejects_empty_dictionary(client):
    import openpyxl

    wb = openpyxl.Workbook()
    buf = io.BytesIO()
    wb.save(buf)

    resp = client.post(
        "/dqc/generate",
        data={"instructions": "algo"},
        files={"dictionary": ("empty.xlsx", buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert resp.status_code == 400


def test_persists_generated_dqcs(client, monkeypatch, isolated_checks_db):
    fake = _FakeClient(_stub_dqcs())
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)
    monkeypatch.setattr(dqc_router, "get_inspect_client", lambda: fake)

    resp = client.post(
        "/dqc/generate",
        data={"instructions": "regla 1"},
        files={"dictionary": ("dict.xlsx", _make_xlsx(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert resp.status_code == 200

    conn = sqlite3.connect(isolated_checks_db)
    rows = conn.execute("SELECT sql, status FROM checks").fetchall()
    assert len(rows) == 1
    assert rows[0][1] == "pending"


def test_llm_error_degrades_gracefully(client, monkeypatch):
    class _BrokenClient:
        def chat_json(self, *a, **k):
            raise RuntimeError("model not loaded")

    broken = _BrokenClient()
    monkeypatch.setattr(dqc_router, "get_client", lambda: broken)
    monkeypatch.setattr(dqc_router, "get_inspect_client", lambda: broken)
    resp = client.post(
        "/dqc/generate",
        data={"instructions": "x"},
        files={"dictionary": ("dict.xlsx", _make_xlsx(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dqcs"] == []
    assert "Error al generar DQCs" in body["context_summary"]

"""Tests for POST /dqc/generate/simple — natural-language DQC expressions
+ at most one caller-supplied regulatory article, no RAG context gathering.
Built for local experimentation with a self-hosted model (REGLLM_LLM=gguf):
a short, single-article prompt instead of the full multi-source RAG prompt.
"""

from __future__ import annotations

import json
import os
import sqlite3

import pytest

os.environ.setdefault("REGLLM_LLM", "stub")

from fastapi.testclient import TestClient  # noqa: E402

import api.routers.dqc as dqc_router  # noqa: E402
from api.main import app  # noqa: E402
from training.dq import checks_db  # noqa: E402


class _FakeClient:
    """Returns a fixed DQCItem-shaped payload, capturing the prompt sent."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.last_system = None
        self.last_user = None

    def chat_json(self, system, user, **kwargs):
        self.last_system = system
        self.last_user = user
        return self.payload


@pytest.fixture()
def isolated_checks_db(tmp_path, monkeypatch):
    """Redirect the validation store to a throwaway sqlite file so tests
    don't pollute (or depend on) the real data/dq/checks.db."""
    db_path = tmp_path / "checks.db"
    real_connect = checks_db.connect  # capture before patching to avoid recursion

    monkeypatch.setattr(dqc_router.checks_db, "connect", lambda: real_connect(db_path))
    return db_path


@pytest.fixture()
def client(isolated_checks_db):
    return TestClient(app)


def _stub_two_dqcs():
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
                "referencia_regulatoria": "EBA GL 2017/16 §73",
                "umbral": "1.0",
                "periodicidad": "mensual",
                "justificacion": "La PD es una probabilidad",
            },
            {
                "dqc_id": "DQC_COLATERAL_TIPO_001",
                "variable": "COLATERAL_TIPO",
                "descripcion": "Hipotecas requieren colateral HIPOTECA",
                "tipo": "referencial",
                "severidad": "bloqueante",
                "regla_sql": "SELECT * FROM mylib.ciclos_recuperacion WHERE SEGMENTO='RETAIL_HIP' AND COLATERAL_TIPO<>'HIPOTECA'",
                "condicion_error": "segmento hipotecario sin colateral hipoteca",
                "campos_entrada": ["SEGMENTO", "COLATERAL_TIPO"],
                "referencia_regulatoria": "Sin referencia regulatoria disponible",
                "umbral": "",
                "periodicidad": "mensual",
                "justificacion": "coherencia de negocio",
            },
        ]
    }


def test_paragraph_mode_loads_real_article_and_generates(client, monkeypatch):
    fake = _FakeClient(_stub_two_dqcs())
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)

    resp = client.post("/dqc/generate/simple", json={
        "expressions": "La PD nunca debe superar el 100%.\nToda hipoteca debe tener colateral HIPOTECA.",
        "article_paragraph": 73,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["article_citation"] == "EBA GL 2017/16 §73"
    assert "tasa de default" in body["article_text_used"]
    assert len(body["dqcs"]) == 2
    assert body["dqcs"][0]["variable"] == "PD_ESTIMADA"

    # the article text and expressions actually reached the prompt
    assert "PD nunca debe superar" in fake.last_user
    assert "tasa de default" in fake.last_user


def test_raw_article_text_mode(client, monkeypatch):
    fake = _FakeClient(_stub_two_dqcs())
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)

    resp = client.post("/dqc/generate/simple", json={
        "expressions": "La LGD debe respetar el suelo del 30% en hipotecas.",
        "article_text": "Las entidades aplicarán un suelo del 30% a la LGD hipotecaria.",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["article_citation"] == "Artículo proporcionado por el usuario"
    assert body["article_text_used"] == "Las entidades aplicarán un suelo del 30% a la LGD hipotecaria."
    assert "suelo del 30%" in fake.last_user


def test_no_article_mode_instructs_no_invented_citation(client, monkeypatch):
    fake = _FakeClient(_stub_two_dqcs())
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)

    resp = client.post("/dqc/generate/simple", json={
        "expressions": "El EAD total debe ser la suma de balance y fuera de balance.",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["article_citation"] == ""
    assert "No se proporcionó ningún artículo" in fake.last_user


def test_unknown_paragraph_degrades_gracefully_without_calling_llm(client, monkeypatch):
    calls = []
    fake = _FakeClient(_stub_two_dqcs())
    monkeypatch.setattr(fake, "chat_json", lambda *a, **k: calls.append(1) or _stub_two_dqcs())
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)

    resp = client.post("/dqc/generate/simple", json={
        "expressions": "algo", "article_paragraph": 999999,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["dqcs"] == []
    assert "No se encontró el párrafo" in body["context_summary"]
    assert calls == [], "must not call the LLM when the paragraph lookup fails"


def test_persists_generated_dqcs_into_checks_db(client, monkeypatch, isolated_checks_db):
    fake = _FakeClient(_stub_two_dqcs())
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)

    resp = client.post("/dqc/generate/simple", json={
        "expressions": "regla 1\nregla 2", "article_paragraph": 73,
    })
    assert resp.status_code == 200

    conn = sqlite3.connect(isolated_checks_db)
    rows = conn.execute("SELECT sql, status FROM checks").fetchall()
    assert len(rows) == 2
    assert all(status == "pending" for _, status in rows)


def test_llm_error_degrades_to_empty_dqcs(client, monkeypatch):
    class _BrokenClient:
        def chat_json(self, *a, **k):
            raise RuntimeError("model not loaded")

    monkeypatch.setattr(dqc_router, "get_client", lambda: _BrokenClient())
    resp = client.post("/dqc/generate/simple", json={"expressions": "x"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dqcs"] == []
    assert "Error al generar DQCs" in body["context_summary"]

"""Tests for POST /dqc/generate_stream — plan-mode SSE generation.

One planner agent splits the rules into a JSON action plan; then one
generation agent runs per plan item, with per-item SSE status events the
chat UI renders as a live checklist.
"""

from __future__ import annotations

import io
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
    """Pops one canned chat_json payload per call, recording prompts."""

    def __init__(self, payloads: list[dict]):
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def chat_json(self, system, user, **kwargs):
        self.calls.append({"system": system, "user": user})
        return self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]


def _make_xlsx() -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Field", "Type", "Description"])
    ws.append(["PD_ESTIMADA", "REAL", "Estimated PD"])
    ws.append(["EAD_TOTAL", "REAL", "Total EAD"])
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


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for frame in body.split("\n\n"):
        if not frame.strip():
            continue
        event, data = "message", []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if data:
            events.append((event, json.loads("".join(data))))
    return events


def _dqc(n: int) -> dict:
    return {"dqcs": [{
        "dqc_id": f"DQC_TEST_{n:03d}", "variable": "PD_ESTIMADA",
        "descripcion": f"regla {n}", "tipo": "rango",
        "severidad": "bloqueante", "regla_sql": "SELECT 1",
    }]}


def test_stream_emits_plan_then_per_item_events(client, monkeypatch,
                                                isolated_checks_db):
    fake = _FakeClient([
        # 1. planner: splits/keeps 2 rules with an action each
        {"plan": [
            {"id": 1, "regla": "PD <= 1", "accion": "check de rango sobre PD_ESTIMADA"},
            {"id": 2, "regla": "EAD >= 0", "accion": "check de rango sobre EAD_TOTAL"},
        ]},
        # 2-3. one generation agent per plan item
        _dqc(1),
        _dqc(2),
    ])
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)
    monkeypatch.setattr(dqc_router, "get_inspect_client", lambda: fake)

    resp = client.post(
        "/dqc/generate_stream",
        data={"instructions": "La PD no supera 1 y el EAD no es negativo",
              "sheet": "Sheet",
              "column_mapping": json.dumps({"field": "Field", "type": "Type"})},
        files={"dictionary": ("d.xlsx", _make_xlsx(),
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    kinds = [e for e, _ in events]
    assert kinds == ["meta", "plan",
                     "item", "item",     # rule 1: en_curso → completado
                     "item", "item",     # rule 2
                     "done"]

    plan = dict(events)["plan"]["items"]
    assert [p["regla"] for p in plan] == ["PD <= 1", "EAD >= 0"]
    assert all(p["estado"] == "pendiente" for p in plan)

    item_events = [d for e, d in events if e == "item"]
    assert [i["estado"] for i in item_events] == [
        "en_curso", "completado", "en_curso", "completado"]
    assert len(item_events[1]["dqcs"]) == 1

    done = events[-1][1]
    assert len(done["dqcs"]) == 2
    # planner + 2 generation agents
    assert done["agents_used"] == 3

    # generated checks were persisted
    conn = sqlite3.connect(isolated_checks_db)
    assert conn.execute("SELECT COUNT(*) FROM checks").fetchone()[0] == 2


def test_stream_item_error_does_not_kill_run(client, monkeypatch,
                                             isolated_checks_db):
    class _FlakyClient(_FakeClient):
        def chat_json(self, system, user, **kwargs):
            if len(self.calls) == 1:  # second call = first generation agent
                self.calls.append({})
                raise RuntimeError("model exploded")
            return super().chat_json(system, user, **kwargs)

    fake = _FlakyClient([
        {"plan": [{"id": 1, "regla": "PD <= 1", "accion": "rango"},
                  {"id": 2, "regla": "EAD >= 0", "accion": "rango"}]},
        _dqc(2),
    ])
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)
    monkeypatch.setattr(dqc_router, "get_inspect_client", lambda: fake)

    resp = client.post(
        "/dqc/generate_stream",
        data={"instructions": "dos reglas", "sheet": "Sheet",
              "column_mapping": json.dumps({"field": "Field", "type": "Type"})},
        files={"dictionary": ("d.xlsx", _make_xlsx(),
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    events = _parse_sse(resp.text)
    item_events = [d for e, d in events if e == "item"]
    assert [i["estado"] for i in item_events] == [
        "en_curso", "error", "en_curso", "completado"]
    assert "model exploded" in item_events[1]["error"]
    done = events[-1][1]
    assert len(done["dqcs"]) == 1  # the surviving rule


def test_stream_planner_failure_falls_back_to_line_split(client, monkeypatch,
                                                         isolated_checks_db):
    class _PlannerDown(_FakeClient):
        def chat_json(self, system, user, **kwargs):
            if not self.calls:  # first call = planner
                self.calls.append({})
                raise RuntimeError("planner down")
            return super().chat_json(system, user, **kwargs)

    fake = _PlannerDown([_dqc(1)])
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)
    monkeypatch.setattr(dqc_router, "get_inspect_client", lambda: fake)

    resp = client.post(
        "/dqc/generate_stream",
        data={"instructions": "PD <= 1", "sheet": "Sheet",
              "column_mapping": json.dumps({"field": "Field", "type": "Type"})},
        files={"dictionary": ("d.xlsx", _make_xlsx(),
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    events = _parse_sse(resp.text)
    plan = dict(events)["plan"]["items"]
    # fallback: one item per input line, generation still ran
    assert plan[0]["regla"] == "PD <= 1"
    assert len(events[-1][1]["dqcs"]) == 1


def test_stream_validation_errors_stay_http(client, monkeypatch):
    fake = _FakeClient([{}])
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)
    monkeypatch.setattr(dqc_router, "get_inspect_client", lambda: fake)
    # no instructions at all → same 400 as /generate, not a broken stream
    resp = client.post(
        "/dqc/generate_stream",
        data={"instructions": "", "sheet": "Sheet"},
        files={"dictionary": ("d.xlsx", _make_xlsx(),
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert resp.status_code == 400

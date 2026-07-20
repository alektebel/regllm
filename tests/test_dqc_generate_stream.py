"""Tests for POST /dqc/generate_stream — plan-mode ReAct generation (SSE).

Per plan item: sufficiency check → SAS generation → static/dynamic
validation → correction loop, with per-item SSE status events the chat UI
renders as a live checklist. A cases Excel enables execution, example
violating rows, and precision/recall against a DQC_ID label column.
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
from api.routers import dqc_react  # noqa: E402
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


def _make_dict_xlsx() -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Field", "Type", "Description"])
    ws.append(["PD_ESTIMADA", "REAL", "Estimated PD"])
    ws.append(["EAD_TOTAL", "REAL", "Total EAD"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_cases_xlsx() -> bytes:
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


def _post(client, instructions, with_data=False, files_extra=None, extra=None):
    files = {"dictionary": ("d.xlsx", _make_dict_xlsx(),
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    if with_data:
        files["data_file"] = ("casos.xlsx", _make_cases_xlsx(),
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if files_extra:
        files.update(files_extra)
    data = {"instructions": instructions, "sheet": "Sheet",
            "column_mapping": json.dumps({"field": "Field", "type": "Type"})}
    if extra:
        data.update(extra)
    return client.post("/dqc/generate_stream", data=data, files=files)


_GOOD_SQL = "SELECT * FROM mylib.ciclos_recuperacion WHERE PD_ESTIMADA > 1"


def _dqc(sql=_GOOD_SQL):
    return {"dqcs": [{
        "dqc_id": "DQC_PD_001", "variable": "PD_ESTIMADA",
        "descripcion": "PD en rango", "tipo": "rango",
        "severidad": "bloqueante", "regla_sql": sql,
    }]}


def _suf(campos=("PD_ESTIMADA",), ok=True, falta=""):
    return {"suficiente": ok, "campos": list(campos),
            "interpretacion": "PD como probabilidad 0-1", "falta": falta}


def _wire(monkeypatch, fake):
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)
    monkeypatch.setattr(dqc_router, "get_inspect_client", lambda: fake)


# ── happy path ───────────────────────────────────────────────────────────────

def test_react_pipeline_full_sequence(client, monkeypatch, isolated_checks_db):
    fake = _FakeClient([
        {"plan": [{"id": 1, "regla": "PD <= 1", "accion": "rango"}]},
        _suf(),                       # sufficiency: everything identified
        _dqc(),                       # SAS generation
    ])
    _wire(monkeypatch, fake)

    resp = _post(client, "La PD no supera 1")
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    kinds = [e for e, _ in events]
    assert kinds == ["meta", "plan", "item", "item", "item", "item", "done"]

    fases = [(d.get("estado"), d.get("fase")) for e, d in events if e == "item"]
    assert fases == [("en_curso", "suficiencia"), ("en_curso", "generacion"),
                     ("en_curso", "validacion"), ("completado", None)]

    done = events[-1][1]
    assert len(done["dqcs"]) == 1
    # planner + sufficiency + generation
    assert done["agents_used"] == 3
    assert done["evaluacion"] is None  # no cases Excel uploaded


# ── ambiguity flagging ───────────────────────────────────────────────────────

def test_insufficient_info_flags_rule_ambigua(client, monkeypatch,
                                              isolated_checks_db):
    fake = _FakeClient([
        {"plan": [{"id": 1, "regla": "el colateral debe ser válido",
                   "accion": "?"}]},
        _suf(campos=(), ok=False,
             falta="No existe ningún campo de colateral en el diccionario"),
    ])
    _wire(monkeypatch, fake)

    events = _parse_sse(_post(client, "el colateral debe ser válido").text)
    item_events = [d for e, d in events if e == "item"]
    assert item_events[-1]["estado"] == "ambigua"
    assert "colateral" in item_events[-1]["falta"]
    assert len(events[-1][1]["dqcs"]) == 0  # nothing generated for it


def test_hallucinated_fields_force_ambigua(client, monkeypatch,
                                           isolated_checks_db):
    # the model SAYS it has enough info but references a non-existent field —
    # the deterministic check must catch it and flag the rule
    fake = _FakeClient([
        {"plan": [{"id": 1, "regla": "regla X", "accion": "?"}]},
        _suf(campos=("PD_ESTIMADA", "CAMPO_INVENTADO"), ok=True),
    ])
    _wire(monkeypatch, fake)

    events = _parse_sse(_post(client, "regla X").text)
    last_item = [d for e, d in events if e == "item"][-1]
    assert last_item["estado"] == "ambigua"
    assert "CAMPO_INVENTADO" in last_item["falta"]


# ── correction loop ──────────────────────────────────────────────────────────

def test_invalid_query_is_corrected_in_loop(client, monkeypatch,
                                            isolated_checks_db):
    bad = _dqc("SELECT * FROM mylib.ciclos_recuperacion WHERE CAMPO_FALSO > 1")
    fake = _FakeClient([
        {"plan": [{"id": 1, "regla": "PD <= 1", "accion": "rango"}]},
        _suf(),
        bad,        # attempt 1: hallucinated field → static validation error
        _dqc(),     # attempt 2: corrected
    ])
    _wire(monkeypatch, fake)

    events = _parse_sse(_post(client, "PD <= 1").text)
    item_events = [d for e, d in events if e == "item"]
    intentos = [(d.get("fase"), d.get("intento")) for d in item_events
                if d["estado"] == "en_curso"]
    assert ("generacion", 1) in intentos and ("generacion", 2) in intentos
    assert item_events[-1]["estado"] == "completado"
    # the correction prompt carried the validation error back to the model
    assert "CAMPO_FALSO" in fake.calls[-1]["user"]


def test_exhausted_attempts_yield_item_error(client, monkeypatch,
                                             isolated_checks_db):
    bad = _dqc("SELECT * FROM mylib.ciclos_recuperacion WHERE CAMPO_FALSO > 1")
    fake = _FakeClient([
        {"plan": [{"id": 1, "regla": "PD <= 1", "accion": "rango"}]},
        _suf(),
        bad, bad, bad,   # never corrected
    ])
    _wire(monkeypatch, fake)

    events = _parse_sse(_post(client, "PD <= 1").text)
    last_item = [d for e, d in events if e == "item"][-1]
    assert last_item["estado"] == "error"
    assert "CAMPO_FALSO" in last_item["error"]
    assert len(events[-1][1]["dqcs"]) == 0


# ── cases Excel: execution, examples, precision/recall ───────────────────────

def test_cases_excel_gives_examples_and_metrics(client, monkeypatch,
                                                isolated_checks_db):
    fake = _FakeClient([
        {"plan": [{"id": 1, "regla": "La PD no puede superar 1",
                   "accion": "rango sobre PD_ESTIMADA"}]},
        _suf(),
        _dqc(),
    ])
    _wire(monkeypatch, fake)

    resp = _post(client, "DQC_PD_001: La PD no puede superar 1",
                 with_data=True)
    events = _parse_sse(resp.text)

    meta = events[0][1]
    assert meta["casos"] == 3

    plan = events[1][1]["items"]
    assert plan[0]["prev_id"] == "DQC_PD_001"
    assert plan[0]["regla"] == "La PD no puede superar 1"

    completed = [d for e, d in events if e == "item"
                 and d["estado"] == "completado"][0]
    val = completed["validacion"]
    assert val["ejecutada"] is True
    assert val["n_casos"] == 2                      # rows with PD 1.5 and 2.0
    assert len(val["ejemplos"]) == 2
    assert "PD_ESTIMADA" in val["columnas"]
    # both flagged rows are labelled DQC_PD_001 → perfect precision/recall
    assert val["precision"] == 1.0
    assert val["recall"] == 1.0
    assert val["esperados"] == 2

    done = events[-1][1]
    assert done["evaluacion"]["tests_comprobados"] == 1
    assert done["evaluacion"]["precision_media"] == 1.0
    assert done["dqcs"][0]["prev_id"] == "DQC_PD_001"

    # previous id persisted as rule_id
    conn = sqlite3.connect(isolated_checks_db)
    rows = conn.execute("SELECT rule_id, check_id FROM checks").fetchall()
    assert [r[0] for r in rows] == ["DQC_PD_001"]

    # executed validation persisted → sidebar cases endpoint serves it
    cases = client.get(f"/dqc/checks/{rows[0][1]}/cases").json()
    assert cases["available"] is True and cases["n_casos"] == 2
    assert cases["precision"] == 1.0


def test_execution_error_feeds_correction_loop(client, monkeypatch,
                                               isolated_checks_db):
    # passes static validation (field exists) but fails at execution time
    broken = _dqc("SELECT * FROM mylib.ciclos_recuperacion WHERE "
                  "PD_ESTIMADA > 1 GROUP BY")
    fake = _FakeClient([
        {"plan": [{"id": 1, "regla": "PD <= 1", "accion": "rango"}]},
        _suf(),
        broken,
        _dqc(),
    ])
    _wire(monkeypatch, fake)

    events = _parse_sse(_post(client, "PD <= 1", with_data=True).text)
    last_item = [d for e, d in events if e == "item"][-1]
    assert last_item["estado"] == "completado"
    assert "falla al ejecutarse" in fake.calls[-1]["user"]


# ── optional features: value grounding + semantic judge (off by default) ────

def test_grounding_injects_real_values_into_prompt(client, monkeypatch,
                                                   isolated_checks_db):
    fake = _FakeClient([
        {"plan": [{"id": 1, "regla": "PD <= 1", "accion": "rango"}]},
        _suf(),
        _dqc(),
    ])
    _wire(monkeypatch, fake)

    events = _parse_sse(_post(client, "PD <= 1", with_data=True,
                              extra={"value_grounding": "true"}).text)
    fases = [d.get("fase") for e, d in events if e == "item"
             and d["estado"] == "en_curso"]
    assert "grounding" in fases
    gen_call = fake.calls[-1]["user"]           # the generation prompt
    assert "VALORES REALES OBSERVADOS" in gen_call
    assert "'1.5'" in gen_call                  # a real PD value from the Excel
    assert [d for e, d in events if e == "item"][-1]["estado"] == "completado"


def test_grounding_off_by_default_no_prompt_change(client, monkeypatch,
                                                   isolated_checks_db):
    fake = _FakeClient([
        {"plan": [{"id": 1, "regla": "PD <= 1", "accion": "rango"}]},
        _suf(),
        _dqc(),
    ])
    _wire(monkeypatch, fake)
    events = _parse_sse(_post(client, "PD <= 1", with_data=True).text)
    assert all(d.get("fase") != "grounding" for e, d in events if e == "item")
    assert "VALORES REALES" not in fake.calls[-1]["user"]


def test_judge_approves_and_annotates(client, monkeypatch, isolated_checks_db):
    fake = _FakeClient([
        {"plan": [{"id": 1, "regla": "PD <= 1", "accion": "rango"}]},
        _suf(),
        _dqc(),
        {"correcto": True, "confianza": 0.9, "motivo": "implementa la regla"},
    ])
    _wire(monkeypatch, fake)

    events = _parse_sse(_post(client, "PD <= 1",
                              extra={"semantic_judge": "true"}).text)
    fases = [d.get("fase") for e, d in events if e == "item"
             and d["estado"] == "en_curso"]
    assert "juicio" in fases
    completed = [d for e, d in events if e == "item"][-1]
    assert completed["estado"] == "completado"
    assert completed["validacion"]["juez_ok"] is True
    assert completed["validacion"]["juez_motivo"] == "implementa la regla"
    # planner + sufficiency + generation + judge
    assert events[-1][1]["agents_used"] == 4


def test_judge_rejection_feeds_correction_loop(client, monkeypatch,
                                               isolated_checks_db):
    fake = _FakeClient([
        {"plan": [{"id": 1, "regla": "PD <= 1", "accion": "rango"}]},
        _suf(),
        _dqc("SELECT * FROM mylib.ciclos_recuperacion WHERE PD_ESTIMADA < 1"),
        {"correcto": False, "confianza": 0.8,
         "motivo": "el sentido está invertido: selecciona las que cumplen"},
        _dqc(),
        {"correcto": True, "confianza": 0.9, "motivo": "ok"},
    ])
    _wire(monkeypatch, fake)

    events = _parse_sse(_post(client, "PD <= 1",
                              extra={"semantic_judge": "true"}).text)
    item_events = [d for e, d in events if e == "item"]
    gen_intentos = [d["intento"] for d in item_events
                    if d.get("fase") == "generacion"]
    assert gen_intentos == [1, 2]
    assert item_events[-1]["estado"] == "completado"
    # calls: 0 planner, 1 sufficiency, 2 gen#1, 3 judge#1, 4 gen#2 (corrective)
    corrective_call = fake.calls[4]["user"]
    assert "juez semántico" in corrective_call and "invertido" in corrective_call


def test_default_run_unaffected_by_new_features(client, monkeypatch,
                                                isolated_checks_db):
    # exact same event sequence and agent count as before the flags existed
    fake = _FakeClient([
        {"plan": [{"id": 1, "regla": "PD <= 1", "accion": "rango"}]},
        _suf(),
        _dqc(),
    ])
    _wire(monkeypatch, fake)
    events = _parse_sse(_post(client, "PD <= 1").text)
    kinds = [e for e, _ in events]
    assert kinds == ["meta", "plan", "item", "item", "item", "item", "done"]
    assert events[-1][1]["agents_used"] == 3


# ── decision trace (feeds the binary-tree view; persisted for audit) ────────

def test_completed_item_carries_decision_trace(client, monkeypatch,
                                               isolated_checks_db):
    fake = _FakeClient([
        {"plan": [{"id": 1, "regla": "PD <= 1", "accion": "rango"}]},
        _suf(),
        _dqc(),
    ])
    _wire(monkeypatch, fake)
    events = _parse_sse(_post(client, "DQC_PD_001: PD <= 1",
                              with_data=True).text)
    completed = [d for e, d in events if e == "item"
                 and d["estado"] == "completado"][0]
    pasos = [t["paso"] for t in completed["trace"]]
    assert pasos == ["suficiencia", "generacion", "validacion", "resultado"]
    assert completed["trace"][0]["resultado"] == "si"
    assert completed["trace"][-1]["n_casos"] == 2

    # trace persisted with the check → served by the cases endpoint
    import sqlite3 as _sql
    conn = _sql.connect(isolated_checks_db)
    check_id = conn.execute("SELECT check_id FROM checks").fetchone()[0]
    cases = client.get(f"/dqc/checks/{check_id}/cases").json()
    assert [t["paso"] for t in cases["trace"]] == pasos


def test_ambiguous_item_trace_records_the_no_branch(client, monkeypatch,
                                                    isolated_checks_db):
    fake = _FakeClient([
        {"plan": [{"id": 1, "regla": "colateral válido", "accion": "?"}]},
        _suf(campos=(), ok=False, falta="no hay campo de colateral"),
    ])
    _wire(monkeypatch, fake)
    events = _parse_sse(_post(client, "colateral válido").text)
    amb = [d for e, d in events if e == "item" and d["estado"] == "ambigua"][0]
    assert [t["paso"] for t in amb["trace"]] == ["suficiencia", "resultado"]
    assert amb["trace"][0]["resultado"] == "no"
    assert amb["trace"][1]["estado"] == "ambigua"


def test_correction_loop_trace_records_failed_validation(client, monkeypatch,
                                                         isolated_checks_db):
    bad = _dqc("SELECT * FROM mylib.ciclos_recuperacion WHERE CAMPO_FALSO > 1")
    fake = _FakeClient([
        {"plan": [{"id": 1, "regla": "PD <= 1", "accion": "rango"}]},
        _suf(),
        bad,
        _dqc(),
    ])
    _wire(monkeypatch, fake)
    events = _parse_sse(_post(client, "PD <= 1").text)
    completed = [d for e, d in events if e == "item"
                 and d["estado"] == "completado"][0]
    pasos = [(t["paso"], t.get("resultado")) for t in completed["trace"]]
    assert pasos == [("suficiencia", "si"),
                     ("generacion", None), ("validacion", "no"),
                     ("generacion", None), ("validacion", "si"),
                     ("resultado", None)]


# ── unit: previous-id extraction ─────────────────────────────────────────────

def test_split_prev_id_formats():
    assert dqc_react.split_prev_id("DQC_PD_001: la PD <= 1") == \
        ("DQC_PD_001", "la PD <= 1")
    assert dqc_react.split_prev_id("[CTRL-7] EAD >= 0") == \
        ("CTRL-7", "EAD >= 0")
    assert dqc_react.split_prev_id("la PD <= 1") == (None, "la PD <= 1")


def test_stream_validation_errors_stay_http(client, monkeypatch):
    fake = _FakeClient([{}])
    _wire(monkeypatch, fake)
    resp = client.post(
        "/dqc/generate_stream",
        data={"instructions": "", "sheet": "Sheet"},
        files={"dictionary": ("d.xlsx", _make_dict_xlsx(),
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert resp.status_code == 400

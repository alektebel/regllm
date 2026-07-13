"""Tests for the dictionary intelligence layer (sheet selection, column
mapping, format inference, context-window batching) and the endpoints
that expose it."""

from __future__ import annotations

import io
import json
import os

import pytest

os.environ.setdefault("REGLLM_LLM", "stub")

from fastapi.testclient import TestClient  # noqa: E402

import api.routers.dqc as dqc_router  # noqa: E402
from api.main import app  # noqa: E402
from api.routers import dqc_dictionary as dict_ai  # noqa: E402
from training.dq import checks_db  # noqa: E402


class _FakeClient:
    """Stateless fake: records every call (agent) and returns payloads."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def chat_json(self, system, user, **kwargs):
        self.calls.append({"system": system, "user": user})
        if not self.payloads:
            return {}
        return self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]


def _workbook(sheets: dict[str, list[list]]) -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(title=name)
        for row in rows:
            ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


DICT_ROWS = [
    ["Field", "Type", "Description", "Null", "Formula", "Reg ref"],
    ["PD_ESTIMADA", "REAL", "PD estimada del ciclo", "no", "", "GL §161"],
    ["LGD_ESTIMADA", "", "LGD estimada", "no", "", ""],
    ["EAD_TOTAL", "REAL", "Exposición total", "no",
     "EAD_BALANCE + EAD_FUERA_BALANCE", "GL §140"],
    ["ECL", "", "Pérdida esperada", "no",
     "PD_ESTIMADA * LGD_ESTIMADA * EAD_TOTAL", "Anejo IX"],
    ["SEGMENTO", "TEXT", "Segmento regulatorio", "no", "", "GL §21"],
]

NOTES_ROWS = [["Notas de la versión"], ["v3 2026-05: se añade ECL"]]


# ── inspection & heuristics ──────────────────────────────────────────────────

def test_inspection_scores_dictionary_sheet_highest():
    data = _workbook({"Notas": NOTES_ROWS, "DICCIONARIO": DICT_ROWS})
    inspection = dict_ai.inspect_workbook(data)
    assert {s.name for s in inspection.sheets} == {"Notas", "DICCIONARIO"}
    assert inspection.best.name == "DICCIONARIO"
    assert not inspection.ambiguous


def test_two_plausible_sheets_are_ambiguous():
    data = _workbook({"HOJA1": DICT_ROWS, "HOJA2": DICT_ROWS})
    inspection = dict_ai.inspect_workbook(data)
    assert inspection.ambiguous


def test_propose_mapping_llm_answer_validated():
    data = _workbook({"Notas": NOTES_ROWS, "DICCIONARIO": DICT_ROWS})
    inspection = dict_ai.inspect_workbook(data)
    fake = _FakeClient([{
        "sheet": "DICCIONARIO",
        "column_mapping": {"field": "Field", "type": "Type",
                           "description": "Description", "nullable": "Null",
                           "formula": "Formula", "reg_ref": "Reg ref"},
        "confidence": 0.95,
    }])
    proposal = dict_ai.propose_mapping(inspection, fake)
    assert proposal["source"] == "llm"
    assert proposal["sheet"] == "DICCIONARIO"
    assert proposal["column_mapping"]["field"] == "Field"
    assert proposal["question"] is None
    assert len(fake.calls) == 1                    # exactly one agent


def test_propose_mapping_falls_back_on_garbage():
    data = _workbook({"DICCIONARIO": DICT_ROWS})
    inspection = dict_ai.inspect_workbook(data)
    fake = _FakeClient([{"sheet": "NO_EXISTE", "column_mapping": {}}])
    proposal = dict_ai.propose_mapping(inspection, fake)
    assert proposal["source"] == "heuristic"
    assert proposal["sheet"] == "DICCIONARIO"


def test_parse_dictionary_with_explicit_mapping():
    # headers the heuristics do NOT know — only the mapping can resolve them
    rows = [["Col_A", "Col_B", "Col_C"],
            ["PD_ESTIMADA", "REAL", "PD estimada"],
            ["EAD_TOTAL", "REAL", "Exposición"]]
    data = _workbook({"X": rows})
    mapping = {"field": "Col_A", "type": "Col_B", "description": "Col_C"}
    fields = dict_ai.parse_dictionary(data, sheet="X", mapping=mapping)
    assert [f.name for f in fields] == ["PD_ESTIMADA", "EAD_TOTAL"]
    assert fields[0].type == "REAL"
    assert fields[0].description == "PD estimada"


# ── format inference ─────────────────────────────────────────────────────────

def test_infer_missing_formats_marks_inferred():
    data = _workbook({"D": DICT_ROWS})
    fields = dict_ai.parse_dictionary(data, sheet="D")
    fake = _FakeClient([{
        "formats": {"LGD_ESTIMADA": {"tipo": "real", "nulo": "no"},
                    "ECL": {"tipo": "REAL", "nulo": "no"}},
    }])
    inferred = dict_ai.infer_missing_formats(fields, fake)
    assert inferred == 2
    lgd = next(f for f in fields if f.name == "LGD_ESTIMADA")
    assert lgd.type == "REAL" and lgd.type_inferred
    assert "(inferido)" in lgd.to_line()
    # typed fields untouched
    pd = next(f for f in fields if f.name == "PD_ESTIMADA")
    assert not pd.type_inferred


# ── context-window discipline ────────────────────────────────────────────────

def test_select_relevant_fields_prefers_named_and_caps():
    fields = [dict_ai.FieldEntry(name=f"CAMPO_{i:03d}", description="relleno")
              for i in range(200)]
    fields.append(dict_ai.FieldEntry(name="ECL", description="pérdida esperada"))
    chosen = dict_ai.select_relevant_fields(
        fields, ["ECL debe igualar PD por LGD por EAD"], cap=10)
    assert len(chosen) == 10
    assert any(f.name == "ECL" for f in chosen)


def test_fields_to_text_respects_budget():
    fields = [dict_ai.FieldEntry(name=f"F{i}", description="x" * 200)
              for i in range(200)]
    text, used = dict_ai.fields_to_text(fields, budget=2000)
    assert len(text) <= 2000
    assert used < 200


def test_plan_batches():
    assert dict_ai.plan_batches(list("abcdefg"), 3) == [
        ["a", "b", "c"], ["d", "e", "f"], ["g"]]


# ── endpoints ────────────────────────────────────────────────────────────────

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


def _upload(data: bytes):
    return {"dictionary": (
        "dic.xlsx", data,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}


def test_inspect_endpoint_proposes_sheet(client, monkeypatch):
    fake = _FakeClient([{
        "sheet": "DICCIONARIO",
        "column_mapping": {"field": "Field", "type": "Type",
                           "description": "Description"},
        "confidence": 0.9,
    }])
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)
    monkeypatch.setattr(dqc_router, "get_inspect_client", lambda: fake)
    data = _workbook({"Notas": NOTES_ROWS, "DICCIONARIO": DICT_ROWS})
    resp = client.post("/dqc/inspect_dictionary", files=_upload(data))
    assert resp.status_code == 200
    body = resp.json()
    assert body["proposed_sheet"] == "DICCIONARIO"
    assert body["source"] == "llm"
    assert body["question"] is None
    assert {s["name"] for s in body["sheets"]} == {"Notas", "DICCIONARIO"}


def test_generate_asks_when_sheets_ambiguous(client, monkeypatch):
    fake = _FakeClient([{"sheet": "HOJA1", "column_mapping": {"field": "Field"},
                         "confidence": 0.4,
                         "question": "¿HOJA1 o HOJA2?"}])
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)
    monkeypatch.setattr(dqc_router, "get_inspect_client", lambda: fake)
    data = _workbook({"HOJA1": DICT_ROWS, "HOJA2": DICT_ROWS})
    resp = client.post("/dqc/generate", data={"instructions": "ECL correcto"},
                       files=_upload(data))
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["needs_sheet_selection"] is True
    assert set(detail["options"]) == {"HOJA1", "HOJA2"}


def test_generate_with_explicit_sheet_batches_agents(client, monkeypatch):
    gen_payload = {"dqcs": [{
        "dqc_id": "DQC_ECL_001", "variable": "ECL",
        "descripcion": "ECL recalculado", "tipo": "formula",
        "severidad": "bloqueante",
        "regla_sql": "SELECT 1", "condicion_error": "difiere",
        "campos_entrada": ["ECL"], "referencia_regulatoria": "Anejo IX",
    }]}
    # payload list with >1 entries pops per call: 1 format call + 2 batches
    fake = _FakeClient([
        {"formats": {"LGD_ESTIMADA": {"tipo": "REAL"},
                     "ECL": {"tipo": "REAL"}}},
        gen_payload,
        gen_payload,
    ])
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)
    monkeypatch.setattr(dqc_router, "get_inspect_client", lambda: fake)
    data = _workbook({"Notas": NOTES_ROWS, "DICCIONARIO": DICT_ROWS})
    instructions = "\n".join(["ECL = PD*LGD*EAD", "PD <= 1", "EAD >= 0"])
    resp = client.post(
        "/dqc/generate",
        data={"instructions": instructions, "sheet": "DICCIONARIO",
              "batch_size": 2,
              "column_mapping": json.dumps({"field": "Field", "type": "Type"})},
        files=_upload(data))
    assert resp.status_code == 200
    body = resp.json()
    assert body["sheet_used"] == "DICCIONARIO"
    assert body["mapping_source"] == "user"
    assert body["formats_inferred"] == 2
    # 3 instructions / batch_size 2 = 2 generation agents + 1 format agent
    assert len(fake.calls) == 3
    # duplicate dqc_ids across batches got de-duplicated
    ids = [d["dqc_id"] for d in body["dqcs"]]
    assert len(ids) == len(set(ids)) == 2


def test_read_instructions_upload_txt_strips_bullets():
    text = "1. La PD nunca supera 1\n- ECL = PD*LGD*EAD\n\n• EAD >= 0\n"
    rules = dict_ai.read_instructions_upload("tests.txt", text.encode())
    assert rules == ["La PD nunca supera 1", "ECL = PD*LGD*EAD", "EAD >= 0"]


def test_read_instructions_upload_xlsx_skips_header():
    data = _workbook({"Tests": [["Regla"], ["PD <= 1"], ["EAD >= 0"]]})
    rules = dict_ai.read_instructions_upload("tests.xlsx", data)
    assert rules == ["PD <= 1", "EAD >= 0"]


def test_read_instructions_upload_csv_first_cell():
    csv = 'Test,Owner\n"PD <= 1",riesgos\nEAD >= 0,riesgos\n'
    rules = dict_ai.read_instructions_upload("tests.csv", csv.encode())
    assert rules == ["PD <= 1", "EAD >= 0"]


def test_generate_accepts_uploaded_test_list(client, monkeypatch):
    fake = _FakeClient([
        {"formats": {"LGD_ESTIMADA": {"tipo": "REAL"}, "ECL": {"tipo": "REAL"}}},
        {"dqcs": [{"dqc_id": "DQC_PD_001", "variable": "PD_ESTIMADA",
                   "descripcion": "PD en rango", "tipo": "rango",
                   "severidad": "bloqueante", "regla_sql": "SELECT 1"}]},
    ])
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)
    monkeypatch.setattr(dqc_router, "get_inspect_client", lambda: fake)
    data = _workbook({"DICCIONARIO": DICT_ROWS})
    tests_txt = b"1. PD_ESTIMADA <= 1\n2. EAD_TOTAL >= 0\n"
    resp = client.post(
        "/dqc/generate",
        data={"instructions": "", "sheet": "DICCIONARIO",
              "column_mapping": json.dumps({"field": "Field", "type": "Type"})},
        files={**_upload(data),
               "instructions_file": ("tests.txt", tests_txt, "text/plain")})
    assert resp.status_code == 200
    # both uploaded rules reached the generation agent
    gen_call = fake.calls[-1]["user"]
    assert "PD_ESTIMADA <= 1" in gen_call and "EAD_TOTAL >= 0" in gen_call


def test_generate_rejects_no_rules_at_all(client, monkeypatch):
    fake = _FakeClient([{}])
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)
    monkeypatch.setattr(dqc_router, "get_inspect_client", lambda: fake)
    data = _workbook({"DICCIONARIO": DICT_ROWS})
    resp = client.post("/dqc/generate",
                       data={"instructions": "", "sheet": "DICCIONARIO"},
                       files=_upload(data))
    assert resp.status_code == 400
    assert "test list" in resp.json()["detail"] or "instructions" in resp.json()["detail"]


def test_generate_single_sheet_no_question(client, monkeypatch):
    fake = _FakeClient([
        {"sheet": "DICCIONARIO",
         "column_mapping": {"field": "Field", "type": "Type"},
         "confidence": 0.9},
        {"formats": {"LGD_ESTIMADA": {"tipo": "REAL"}, "ECL": {"tipo": "REAL"}}},
        {"dqcs": []},
    ])
    monkeypatch.setattr(dqc_router, "get_client", lambda: fake)
    monkeypatch.setattr(dqc_router, "get_inspect_client", lambda: fake)
    data = _workbook({"DICCIONARIO": DICT_ROWS})
    resp = client.post("/dqc/generate",
                       data={"instructions": "PD <= 1"}, files=_upload(data))
    assert resp.status_code == 200
    assert resp.json()["sheet_used"] == "DICCIONARIO"

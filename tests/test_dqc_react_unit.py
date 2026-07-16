"""Unit tests for the ReAct pipeline pieces in api.routers.dqc_react."""

from __future__ import annotations

import io

import pytest

from api.routers import dqc_react as react
from api.routers.dqc_dictionary import FieldEntry

FIELDS = [FieldEntry(name="PD_ESTIMADA", type="REAL"),
          FieldEntry(name="EAD_TOTAL", type="REAL"),
          FieldEntry(name="LGD_ESTIMADA", type="REAL")]
TABLE = "mylib.ciclos_recuperacion"


# ── static validation ────────────────────────────────────────────────────────

def test_static_validate_accepts_known_fields_functions_aliases():
    sql = ("SELECT * FROM mylib.ciclos_recuperacion AS ciclos "
           "WHERE ABS(PD_ESTIMADA - 1) > 0.01 AND COALESCE(EAD_TOTAL, 0) < 0")
    assert react.static_validate(sql, FIELDS, TABLE) == []


def test_static_validate_flags_hallucinated_fields():
    sql = "SELECT * FROM mylib.ciclos_recuperacion WHERE CAMPO_FANTASMA IS NULL"
    errors = react.static_validate(sql, FIELDS, TABLE)
    assert len(errors) == 1 and "CAMPO_FANTASMA" in errors[0]


def test_static_validate_ignores_string_literals():
    sql = ("SELECT * FROM mylib.ciclos_recuperacion "
           "WHERE PD_ESTIMADA = 'TEXTO_QUE_NO_ES_CAMPO'")
    assert react.static_validate(sql, FIELDS, TABLE) == []


def test_static_validate_rejects_non_select():
    assert react.static_validate("DROP TABLE x", FIELDS, TABLE)
    assert react.static_validate("", FIELDS, TABLE)


# ── cases loading + execution ────────────────────────────────────────────────

def _cases(rows, headers=("PD_ESTIMADA", "EAD_TOTAL", "DQC_ID")):
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(list(headers))
    for r in rows:
        ws.append(list(r))
    buf = io.BytesIO()
    wb.save(buf)
    return react.load_cases(buf.getvalue(), TABLE)


def test_load_cases_parses_labels_multi_id():
    ctx = _cases([(1.5, 100, "DQC_A"), (0.5, 200, None),
                  (2.0, -1, "DQC_A; DQC_B")])
    assert ctx.n_rows == 3
    assert ctx.labels == {1: {"DQC_A"}, 3: {"DQC_A", "DQC_B"}}


def test_load_cases_without_label_column():
    ctx = _cases([(1.5, 100)], headers=("PD_ESTIMADA", "EAD_TOTAL"))
    assert ctx.label_idx is None and ctx.labels == {}


def test_load_cases_empty_workbook_raises():
    import openpyxl

    wb = openpyxl.Workbook()
    buf = io.BytesIO()
    wb.save(buf)
    with pytest.raises(ValueError):
        react.load_cases(buf.getvalue(), TABLE)


def test_run_query_specific_columns_counts_rows_without_metrics():
    ctx = _cases([(1.5, 100, "DQC_A"), (2.0, -1, "DQC_A")])
    out = react.run_query(
        ctx, f"SELECT PD_ESTIMADA FROM {TABLE} WHERE PD_ESTIMADA > 1", TABLE)
    assert out["ok"] and out["n_casos"] == 2
    assert out["casos"] == set()          # no _CASO_ column selected
    assert react.metrics(ctx, out["casos"], "DQC_A") is None


def test_run_query_strips_proc_sql_wrappers():
    ctx = _cases([(1.5, 100, "DQC_A")])
    sql = f"proc sql;\nSELECT * FROM {TABLE} WHERE PD_ESTIMADA > 1;\nquit;"
    out = react.run_query(ctx, sql, TABLE)
    assert out["ok"] and out["n_casos"] == 1


def test_metrics_partial_overlap():
    ctx = _cases([(1.5, 0, "DQC_A"),     # flagged, expected     → TP
                  (2.0, 0, ""),          # flagged, not expected → FP
                  (0.5, 0, "DQC_A")])    # not flagged, expected → FN
    out = react.run_query(
        ctx, f"SELECT * FROM {TABLE} WHERE PD_ESTIMADA > 1", TABLE)
    m = react.metrics(ctx, out["casos"], "DQC_A")
    assert m == {"precision": 0.5, "recall": 0.5, "esperados": 2, "aciertos": 1}


def test_metrics_requires_prev_id_and_labels():
    ctx = _cases([(1.5, 0, "DQC_A")])
    out = react.run_query(
        ctx, f"SELECT * FROM {TABLE} WHERE PD_ESTIMADA > 1", TABLE)
    assert react.metrics(ctx, out["casos"], None) is None
    assert react.metrics(ctx, out["casos"], "DQC_DESCONOCIDO") is None


# ── previous-id extraction ───────────────────────────────────────────────────

def test_split_prev_id_edge_cases():
    assert react.split_prev_id("DQC_X_001 - la regla") == ("DQC_X_001", "la regla")
    # a bare id with no rule text is NOT an id
    assert react.split_prev_id("DQC_X_001") == (None, "DQC_X_001")
    # ids must look like ids, not arbitrary words
    assert react.split_prev_id("La PD: nunca supera 1") == (None, "La PD: nunca supera 1")

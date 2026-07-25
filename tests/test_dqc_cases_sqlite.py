"""Unit tests for the cases-Excel → SQLite layer in api/routers/dqc_react.py.

Two things are covered, both of which turn a *correct* DQC into a spurious
failure when they go wrong:

  * header sanitation — SQLite rejects blank/duplicate column names, so an
    Excel with an empty trailing column or two columns differing only in case
    used to blow up with "duplicate column name";
  * SAS → SQLite translation — the generated check is SAS PROC SQL but is
    validated by executing it against the cases in SQLite, so SAS-only idioms
    must be rewritten first.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import openpyxl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.routers import dqc_react as react  # noqa: E402


def _cases_xlsx(rows: list[list]) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "CASOS"
    for r in rows:
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ── header sanitation ───────────────────────────────────────────────────────

def test_dedupe_keeps_distinct_names_untouched():
    """Real column names must survive verbatim — generated SQL references them."""
    assert react._dedupe_headers(["PD", "EAD", "LGD"], "_CASO_") == ["PD", "EAD", "LGD"]


def test_dedupe_renames_blank_headers():
    out = react._dedupe_headers(["PD", "", "  "], "_CASO_")
    assert out[0] == "PD"
    assert out[1] != "" and out[2] != "" and out[1] != out[2]


def test_dedupe_resolves_case_insensitive_collision():
    """SQLite column names are case-insensitive: PD and pd collide."""
    out = react._dedupe_headers(["PD", "pd"], "_CASO_")
    assert out[0] == "PD"
    assert out[1].lower() != "pd"
    assert len({c.lower() for c in out}) == 2


def test_dedupe_avoids_the_reserved_synthetic_column():
    out = react._dedupe_headers(["_CASO_"], "_CASO_")
    assert out[0].lower() != "_caso_"


def test_dedupe_preserves_length_and_order():
    """Row values are positional — the count must never change."""
    headers = ["A", "", "a", "B", "", "_CASO_"]
    out = react._dedupe_headers(headers, "_CASO_")
    assert len(out) == len(headers)
    assert out[0] == "A" and out[3] == "B"
    assert len({c.lower() for c in out}) == len(out)   # all unique


def test_load_cases_survives_blank_and_duplicate_headers():
    """The regression this fixed: an Excel that looks fine but has an empty
    trailing column plus a case-duplicate used to raise 'duplicate column name'."""
    data = _cases_xlsx([
        ["PD_ESTIMADA", "EAD_TOTAL", "pd_estimada", "", "DQC_ID"],
        [1.15, 12000, 0.4, None, "DQC_PD_001"],
        [0.50, -500, 0.4, None, ""],
    ])
    ctx = react.load_cases(data, "mylib.casos")
    assert ctx.n_rows == 2
    assert len({h.lower() for h in ctx.headers}) == len(ctx.headers)
    # the label column is still detected, on the original header name
    assert ctx.labels == {1: {"DQC_PD_001"}}


def test_load_cases_without_label_column():
    ctx = react.load_cases(_cases_xlsx([["PD"], [0.5]]), "mylib.casos")
    assert ctx.label_idx is None and ctx.labels == {}


def test_load_cases_rejects_a_sheet_with_no_headers():
    with pytest.raises(ValueError):
        react.load_cases(_cases_xlsx([]), "mylib.casos")


# ── SAS → SQLite translation ────────────────────────────────────────────────

@pytest.mark.parametrize("sas,expected", [
    ("a ^= b", "a <> b"),                 # SAS not-equal
    ("a ~= b", "a <> b"),                 # the other SAS not-equal
    ("UPCASE(T) = 'X'", "UPPER(T) = 'X'"),
    ("LOWCASE(T) = 'x'", "LOWER(T) = 'x'"),
    ("STRIP(N)", "TRIM(N)"),
    ("INDEX(C, 'A') > 0", "INSTR(C, 'A') > 0"),
    ("CALCULATED ratio > 1", "ratio > 1"),
])
def test_sas_idioms_are_translated(sas, expected):
    assert react._sas_to_sqlite(sas) == expected


def test_missing_dot_comparisons_become_null_checks():
    assert "IS NULL" in react._sas_to_sqlite("PD = .")
    assert "IS NOT NULL" in react._sas_to_sqlite("PD <> .")


def test_missing_function_becomes_is_null():
    assert react._sas_to_sqlite("MISSING(LGD)") == "(LGD IS NULL)"


def test_decimals_are_not_mistaken_for_the_missing_dot():
    """`= 0.5` must survive — only a bare dot means SAS missing."""
    assert react._sas_to_sqlite("PD = 0.5") == "PD = 0.5"


def test_plain_ansi_sql_is_left_alone():
    q = "SELECT * FROM t WHERE PD > 1 AND EAD IS NULL"
    assert react._sas_to_sqlite(q) == q


# ── end-to-end: a SAS-flavoured check validates against the Excel ───────────

def test_sas_style_query_runs_against_the_cases():
    data = _cases_xlsx([
        ["PD_ESTIMADA", "EAD_TOTAL", "DQC_ID"],
        [1.15, 12000, "DQC_PD_001"],     # violates PD <= 1
        [0.50, None, ""],                # EAD missing
    ])
    ctx = react.load_cases(data, "mylib.casos")

    res = react.run_query(
        ctx, "SELECT * FROM mylib.casos WHERE PD_ESTIMADA ^= . AND PD_ESTIMADA > 1",
        "mylib.casos")
    assert res["ok"] is True and res["n_casos"] == 1

    res2 = react.run_query(
        ctx, "SELECT * FROM mylib.casos WHERE MISSING(EAD_TOTAL)", "mylib.casos")
    assert res2["ok"] is True and res2["n_casos"] == 1


def test_proc_sql_wrapper_is_tolerated():
    ctx = react.load_cases(
        _cases_xlsx([["PD"], [1.5], [0.2]]), "mylib.casos")
    res = react.run_query(
        ctx, "PROC SQL;\nSELECT * FROM mylib.casos WHERE PD > 1;\nQUIT;",
        "mylib.casos")
    assert res["ok"] is True and res["n_casos"] == 1


def test_invalid_sql_reports_an_error_instead_of_raising():
    ctx = react.load_cases(_cases_xlsx([["PD"], [1.0]]), "mylib.casos")
    res = react.run_query(ctx, "SELECT * FROM mylib.casos WHERE NOPE >", "mylib.casos")
    assert res["ok"] is False and res["error"]

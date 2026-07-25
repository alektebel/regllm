"""Unit tests for api/routers/dqc_uploads.py — upload validation and the
user-facing workbook error messages.

Pure functions: no database, no LLM, no HTTP client needed.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException  # noqa: E402

from api.routers.dqc_uploads import (  # noqa: E402
    parse_mapping_form,
    split_instructions,
    workbook_read_error,
)


class _Upload:
    """Minimal stand-in for fastapi.UploadFile (only .filename is read)."""

    def __init__(self, filename):
        self.filename = filename


# ── require_xlsx ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["dic.xlsx", "DIC.XLSX", "libro.xls", "a.b.xlsx"])
def test_require_xlsx_accepts_excel_extensions(name):
    from api.routers.dqc_uploads import require_xlsx
    require_xlsx(_Upload(name))  # must not raise


@pytest.mark.parametrize("name", ["datos.csv", "notas.txt", "libro", "x.xlsx.exe", None])
def test_require_xlsx_rejects_non_excel(name):
    from api.routers.dqc_uploads import require_xlsx
    with pytest.raises(HTTPException) as exc:
        require_xlsx(_Upload(name))
    assert exc.value.status_code == 400


def test_require_xlsx_names_the_subject_in_the_error():
    from api.routers.dqc_uploads import require_xlsx
    with pytest.raises(HTTPException) as exc:
        require_xlsx(_Upload("x.csv"), what="cases file")
    assert "cases file" in exc.value.detail


# ── workbook_read_error ─────────────────────────────────────────────────────

def test_read_error_always_names_file_and_technical_detail():
    msg = workbook_read_error("midic.xlsx", ValueError("boom"))
    assert "midic.xlsx" in msg                 # which file failed
    assert "ValueError" in msg and "boom" in msg   # the technical reason
    assert "No se pudo leer el Excel" in msg   # stable prefix the UI/tests use


def test_read_error_legacy_xls_hint():
    msg = workbook_read_error("datos.xls", ValueError("nope"))
    assert ".xls antiguo" in msg
    assert ".xlsx" in msg                      # tells them what to convert to


def test_read_error_not_a_real_xlsx_hint():
    """A CSV/Numbers/Sheets export renamed .xlsx → openpyxl raises BadZipFile."""
    import openpyxl
    with pytest.raises(Exception) as exc:
        openpyxl.load_workbook(io.BytesIO(b"a,b,c\n1,2,3\n"))
    msg = workbook_read_error("renamed.xlsx", exc.value)
    assert "no es un .xlsx real" in msg


def test_read_error_password_protected_hint():
    msg = workbook_read_error("d.xlsx", ValueError("File is encrypted with password"))
    assert "contraseña" in msg


def test_read_error_generic_fallback_still_actionable():
    msg = workbook_read_error("d.xlsx", RuntimeError("something odd"))
    assert "válido" in msg and "RuntimeError" in msg


def test_read_error_handles_missing_filename_and_empty_message():
    msg = workbook_read_error(None, ValueError(""))
    assert "sin nombre" in msg and "(sin mensaje)" in msg


# ── parse_mapping_form ──────────────────────────────────────────────────────

def test_parse_mapping_none_and_empty_return_none():
    assert parse_mapping_form(None) is None
    assert parse_mapping_form("") is None


def test_parse_mapping_valid_json_object():
    assert parse_mapping_form('{"field": "Campo"}') == {"field": "Campo"}


def test_parse_mapping_non_object_json_is_ignored():
    """A JSON array/scalar is valid JSON but not a mapping → treated as absent."""
    assert parse_mapping_form("[1, 2]") is None
    assert parse_mapping_form('"x"') is None


def test_parse_mapping_invalid_json_is_a_400():
    with pytest.raises(HTTPException) as exc:
        parse_mapping_form("{not json")
    assert exc.value.status_code == 400


# ── split_instructions ──────────────────────────────────────────────────────

def test_split_instructions_one_rule_per_line_trimmed():
    assert split_instructions("  a  \n\n b \n") == ["a", "b"]


def test_split_instructions_single_line_and_blank():
    assert split_instructions("solo una regla") == ["solo una regla"]
    assert split_instructions("   ") == []
    assert split_instructions("") == []

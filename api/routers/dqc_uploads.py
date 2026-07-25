"""Upload handling for the DQC endpoints: validating what the user sent and
turning a failed workbook read into a message they can act on.

Split out of ``dqc.py`` because these are pure input-validation concerns with
no database or LLM involvement — which also makes them cheap to unit-test.
"""

from __future__ import annotations

import json
import logging

from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)


def require_xlsx(upload: UploadFile, what: str = "dictionary") -> None:
    """Reject anything that is not an Excel upload before we try to parse it."""
    if not upload.filename or not upload.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400,
                            detail=f"{what} must be an Excel file (.xlsx)")


def workbook_read_error(filename: str | None, exc: Exception) -> str:
    """Verbose, user-facing detail for a workbook openpyxl could not load:
    a symptom-specific hint plus the technical reason. The full error (with
    traceback) is also logged server-side for `aws logs tail` / the console."""
    name = filename or "sin nombre"
    lower = name.lower()
    cls = exc.__class__.__name__
    msg = str(exc).strip() or "(sin mensaje)"
    mlow = msg.lower()
    logger.warning("workbook read failed for %r: %s: %s", name, cls, msg,
                   exc_info=True)

    if lower.endswith(".xls"):
        hint = ("Es un .xls antiguo (formato no soportado). Ábrelo en Excel o "
                "LibreOffice y usa «Guardar como → Libro de Excel (.xlsx)».")
    elif cls == "BadZipFile" or "not a zip" in mlow or "file is not" in mlow:
        hint = ("El fichero no es un .xlsx real por dentro (un .xlsx es un ZIP). "
                "Suele pasar con un .csv, un .xls, o un export de Google Sheets / "
                "Apple Numbers renombrado a .xlsx. Ábrelo y usa «Guardar como → "
                "Libro de Excel (.xlsx)».")
    elif any(k in mlow for k in ("password", "encrypt", "protected", "cifr")):
        hint = ("Parece protegido con contraseña o cifrado. Quita la protección "
                "(Archivo → Información → Quitar contraseña) y guárdalo de nuevo.")
    elif "support" in mlow and ("xls" in mlow or "format" in mlow):
        hint = ("Formato no soportado. Guárdalo como «Libro de Excel (.xlsx)» "
                "(no .xls, .csv ni .ods).")
    else:
        hint = ("Comprueba que es un .xlsx válido, sin contraseña y no corrupto; "
                "si dudas, vuelve a exportarlo desde el origen como .xlsx.")

    return (f"No se pudo leer el Excel «{name}»: {hint} "
            f"[detalle técnico → {cls}: {msg}]")


def parse_mapping_form(column_mapping: str | None) -> dict | None:
    """Decode the optional column-mapping form field (JSON object or nothing)."""
    if not column_mapping:
        return None
    try:
        parsed = json.loads(column_mapping)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="column_mapping must be JSON")
    return parsed if isinstance(parsed, dict) else None


def split_instructions(text: str) -> list[str]:
    """One rule per non-empty line; the whole text as a single rule otherwise."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines
    return [text.strip()] if text.strip() else []

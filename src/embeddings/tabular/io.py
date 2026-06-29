"""Load tabular files (CSV / Excel) for TabBERT embedding."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd


def load_table(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    """Load a CSV or Excel file into a list of row dicts (string values)."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, nrows=max_rows, dtype=str, keep_default_na=False)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, nrows=max_rows, dtype=str, engine="openpyxl")
        df = df.fillna("")
    else:
        raise ValueError(f"Unsupported format {suffix!r} — use .csv or .xlsx")

    rows: list[dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        rows.append({str(k): "" if v is None else str(v) for k, v in rec.items()})
    return rows


def table_columns(path: Path) -> list[str]:
    rows = load_table(path, max_rows=1)
    if not rows:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader.fieldnames or [])
    return list(rows[0].keys())


def suggest_id_column(columns: list[str]) -> str | None:
    """Heuristic: pick a column likely to be a row identifier."""
    upper = {c: c.upper() for c in columns}
    for pattern in ("ID_", "ID", "PK", "KEY", "UUID", "GUID"):
        for col, u in upper.items():
            if pattern in u:
                return col
    return columns[0] if columns else None

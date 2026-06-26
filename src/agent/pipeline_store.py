"""Read pipeline output rows for a SAS session (ground-truth DB values)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SAMPLES = _PROJECT_ROOT / "data" / "samples"
_SESSION_DATA = _SAMPLES / "sessions"
_DEFAULT_CSV = _SAMPLES / "cycles_v3.csv"
_PK = "CICLO_ID"

_NUMERIC_COLS = {
    "PD_ESTIMADA", "LGD_ESTIMADA", "EAD", "DPDS", "STAGE_IFRS9",
    "PROVISION_PERIOD_MONTHS", "VENTANA_OBSERVACION_YEARS",
    "VENTANA_CALIBRACION_YEARS", "RATING_GRADO", "ECL",
    "LGD_FLOOR_APLICADO", "MOC", "CURE_FLAG", "RWA",
    "OR_EAD", "OR_EAD_TIT", "LGD_CON_MOC", "K_IRB", "SW_FUSION",
    "FLAG_NC",
}

_MISSING_TOKENS = frozenset({".", "", "NULL", "NA", "N/A", "NONE"})


def session_data_path(session_id: str) -> Path | None:
    """Return session-specific pipeline output CSV if it exists."""
    for name in ("lgd_output.csv", "pipeline_output.csv"):
        p = _SESSION_DATA / session_id / name
        if p.exists():
            return p
    return None


def resolve_data_path(session_id: str = "default") -> Path:
    p = session_data_path(session_id)
    return p if p else _DEFAULT_CSV


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        ku = k.upper() if isinstance(k, str) else k
        if v is None:
            out[ku] = None
            continue
        sv = str(v).strip()
        if sv.upper() in _MISSING_TOKENS:
            out[ku] = None
            continue
        if ku in _NUMERIC_COLS:
            try:
                out[ku] = float(sv)
                continue
            except ValueError:
                pass
        out[ku] = sv
    return out


def load_rows(session_id: str = "default") -> list[dict[str, Any]]:
    path = resolve_data_path(session_id)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [_coerce_row(r) for r in csv.DictReader(f)]


def _is_missing(row: dict[str, Any], field: str) -> bool:
    v = row.get(field.upper())
    return v is None


def _row_matches_filter(row: dict[str, Any], field: str, value: Any) -> bool:
    fu = field.upper()
    rv = row.get(fu)
    if value is None or (isinstance(value, str) and value.upper() in ("MISSING", ".", "NULL")):
        return _is_missing(row, fu)
    if isinstance(value, (int, float)) and isinstance(rv, (int, float)):
        return abs(float(rv) - float(value)) < 1e-9
    return str(rv).upper() == str(value).upper()


def query_cycle_data(
    session_id: str = "default",
    *,
    ciclo_id: str | None = None,
    fields: list[str] | None = None,
    missing_field: str | None = None,
    filter_field: str | None = None,
    filter_value: Any = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Query actual row values from the session pipeline output table."""
    path = resolve_data_path(session_id)
    rows = load_rows(session_id)
    if not rows:
        return {
            "session_id": session_id,
            "source": str(path),
            "found": False,
            "error": f"no data at {path}",
            "rows": [],
        }

    hits = rows
    if ciclo_id:
        pk_u = ciclo_id.upper()
        hits = [r for r in hits if str(r.get(_PK, "")).upper() == pk_u]

    if missing_field:
        mf = missing_field.upper()
        hits = [r for r in hits if _is_missing(r, mf)]

    if filter_field:
        hits = [r for r in hits if _row_matches_filter(r, filter_field, filter_value)]

    field_u = [f.upper() for f in fields] if fields else None
    out_rows: list[dict[str, Any]] = []
    for r in hits[: max(1, min(limit, 50))]:
        if field_u:
            out_rows.append({k: r.get(k) for k in field_u if k in r or k == _PK})
            if _PK not in out_rows[-1] and _PK in r:
                out_rows[-1][_PK] = r[_PK]
        else:
            out_rows.append(dict(r))

    return {
        "session_id": session_id,
        "source": str(path.relative_to(_PROJECT_ROOT)) if path.is_relative_to(_PROJECT_ROOT) else str(path),
        "found": bool(out_rows),
        "row_count": len(out_rows),
        "total_matching": len(hits),
        "rows": out_rows,
    }


def sample_for_causal_chain(
    session_id: str,
    target: str,
    chain_fields: list[str],
    *,
    limit: int = 5,
) -> dict[str, Any]:
    """Rows where target is missing vs present — with chain field values."""
    target_u = target.upper()
    fields = list(dict.fromkeys([_PK, target_u, *chain_fields]))
    field_set = [f.upper() for f in fields]

    rows = load_rows(session_id)
    missing_rows: list[dict[str, Any]] = []
    present_rows: list[dict[str, Any]] = []
    for r in rows:
        subset = {k: r.get(k) for k in field_set if k in r}
        if _PK in r:
            subset[_PK] = r[_PK]
        if _is_missing(r, target_u):
            if len(missing_rows) < limit:
                missing_rows.append(subset)
        elif len(present_rows) < min(3, limit):
            present_rows.append(subset)

    return {
        "target": target_u,
        "fields_shown": field_set,
        "rows_missing_target": missing_rows,
        "rows_with_target": present_rows,
        "missing_count": sum(1 for r in rows if _is_missing(r, target_u)),
        "present_count": sum(1 for r in rows if not _is_missing(r, target_u)),
        "source": str(resolve_data_path(session_id)),
    }

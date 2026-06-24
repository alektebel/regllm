"""Serialization strategies for converting tabular rows to text.

Each strategy produces a flat text string that captures the column-value
pairs of a row.  The resulting text is then embedded via a text embedding
model (TabBERT-style approach).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


class SerializationStrategy(str, Enum):
    FLAT = "flat"         # "COL_NAME: value | COL_NAME: value | ..."
    COLON = "colon"       # "[COL] COL_NAME [VAL] value [COL] COL_NAME [VAL] value ..."
    JSON = "json"         # JSON string of the row dictionary
    COL_VAL = "col_val"   # "COL_NAME is value . COL_NAME is value . ..."

    @classmethod
    def _missing_(cls, value: object) -> SerializationStrategy | None:
        if isinstance(value, str):
            for member in cls:
                if member.value == value.lower():
                    return member
        return None


# Columns to exclude from serialization (IDs that add noise)
EXCLUDE_COLS = {
    "ID_CONTR_CICLO_LGD",
    "ID_FUSION_FINAL",
}

# Columns whose values should be treated as labels (not numeric)
CATEGORICAL_COLS = {
    "SEGMENTO",
    "CALIBRATION_SEGMENT",
    "PRODUCTO",
    "TERMINACION",
    "CAUSA_DEFAULT",
    "COLATERAL_TIPO",
    "ESTADO_CICLO",
    "ADJUDICACION_TIPO",
    "STAGE_IFRS9",
}

# Rename short cryptic names to more descriptive ones for better embedding
COLUMN_RENAMES: dict[str, str] = {
    "OR_DISPTO": "IMPORTE_ORIGINAL_DISPTO",
    "EAD": "EXPOSICION_AL_IMPAGO",
    "ECL": "PERDIDA_ESPERADA",
    "PD_ESTIMADA": "PROBABILIDAD_IMPAGO",
    "LGD_ESTIMADA": "SEVERIDAD_ESTIMADA",
    "LGD_REALIZADA": "SEVERIDAD_REALIZADA",
    "DPDS": "DIAS_MORA",
    "RWA": "ACTIVOS_PONDERADOS_RIESGO",
    "LTV": "LOAN_TO_VALUE",
}


def _clean_val(value: Any) -> str:
    """Format a single value for serialization."""
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return "NONE"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def serialize_flat(row: dict[str, Any]) -> str:
    """Serialize as 'COL_NAME: value | COL_NAME: value | ...'"""
    parts: list[str] = []
    for k, v in row.items():
        if k in EXCLUDE_COLS:
            continue
        name = COLUMN_RENAMES.get(k, k)
        parts.append(f"{name}: {_clean_val(v)}")
    return " | ".join(parts)


def serialize_colon(row: dict[str, Any]) -> str:
    """Serialize as '[COL] COL_NAME [VAL] value [COL] COL_NAME [VAL] value ...'

    This is the format described in the TabBERT paper.
    """
    parts: list[str] = []
    for k, v in row.items():
        if k in EXCLUDE_COLS:
            continue
        name = COLUMN_RENAMES.get(k, k)
        parts.append(f"[COL] {name} [VAL] {_clean_val(v)}")
    return " ".join(parts)


def serialize_json(row: dict[str, Any]) -> str:
    """Serialize as a JSON object string."""
    import json
    clean = {COLUMN_RENAMES.get(k, k): _clean_val(v)
             for k, v in row.items()
             if k not in EXCLUDE_COLS}
    return json.dumps(clean, ensure_ascii=False)


def serialize_col_val(row: dict[str, Any]) -> str:
    """Serialize as 'COL_NAME is value . COL_NAME is value . ...'"""
    parts: list[str] = []
    for k, v in row.items():
        if k in EXCLUDE_COLS:
            continue
        name = COLUMN_RENAMES.get(k, k)
        parts.append(f"{name} is {_clean_val(v)}")
    return " . ".join(parts) + " ."


_SERIALIZERS = {
    SerializationStrategy.FLAT: serialize_flat,
    SerializationStrategy.COLON: serialize_colon,
    SerializationStrategy.JSON: serialize_json,
    SerializationStrategy.COL_VAL: serialize_col_val,
}


def serialize_row(
    row: dict[str, Any],
    strategy: SerializationStrategy | str = SerializationStrategy.COLON,
) -> str:
    """Serialize a single tabular row to text."""
    if isinstance(strategy, str):
        strategy = SerializationStrategy(strategy)
    serializer = _SERIALIZERS.get(strategy)
    if serializer is None:
        raise ValueError(f"Unknown strategy: {strategy!r}; "
                         f"choose from {list(_SERIALIZERS.keys())}")
    return serializer(row)

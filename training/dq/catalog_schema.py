"""Schema + field universe for LLM-generated DQC catalogs."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.dq.coherence_rules import PK_COLUMN, RULES, TABLE_NAME

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "samples" / "recuperatory_cycles.csv"
DEFAULT_UNIVERSE_PATH = Path(__file__).parent / "data" / "catalog_field_universe.json"

# PK is the violation output column, not an auditable business attribute.
EXCLUDED_FROM_COVERAGE = {PK_COLUMN}

REQUIRED_ENTRY_KEYS = ("dqc_id", "descripcion", "campos_entrada")
MULTI_FIELD_TIPOS = {"consistencia", "formula", "referencial", "cross_field", "cross_table"}
VALID_TIPOS = MULTI_FIELD_TIPOS | {"rango", "completitud", "cardinality"}

_SQL_KEYWORDS = {
    "AND", "OR", "NOT", "NULL", "IS", "IN", "BETWEEN", "LIKE", "SELECT",
    "FROM", "WHERE", "CASE", "WHEN", "THEN", "ELSE", "END", "COALESCE",
    "CAST", "AS", "DISTINCT", "TRUE", "FALSE", "ON", "JOIN", "INNER",
    "LEFT", "RIGHT", "OUTER", "GROUP", "ORDER", "HAVING", "BY", "LIMIT",
    "UNION", "ALL", "EXISTS", "ABS", "SUM", "COUNT", "MAX", "MIN", "AVG",
}


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str = "text"
    group: str = ""
    description: str = ""


@dataclass
class CatalogEntry:
    dqc_id: str
    descripcion: str
    campos_entrada: list[str]
    variable: str = ""
    tipo: str = ""
    severidad: str = ""
    regla_sql: str = ""
    condicion_error: str = ""
    referencia_regulatoria: str = ""
    umbral: str = ""
    periodicidad: str = ""
    justificacion: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CatalogEntry":
        campos = raw.get("campos_entrada") or raw.get("fields") or []
        if isinstance(campos, str):
            campos = [c.strip() for c in campos.split(",") if c.strip()]
        return cls(
            dqc_id=str(raw.get("dqc_id") or raw.get("id") or "").strip(),
            descripcion=str(raw.get("descripcion") or raw.get("description") or "").strip(),
            campos_entrada=[str(c).strip() for c in campos if str(c).strip()],
            variable=str(raw.get("variable") or "").strip(),
            tipo=str(raw.get("tipo") or raw.get("category") or "").strip().lower(),
            severidad=str(raw.get("severidad") or raw.get("severity") or "").strip(),
            regla_sql=str(raw.get("regla_sql") or raw.get("sql") or "").strip(),
            condicion_error=str(raw.get("condicion_error") or "").strip(),
            referencia_regulatoria=str(raw.get("referencia_regulatoria") or "").strip(),
            umbral=str(raw.get("umbral") or "").strip(),
            periodicidad=str(raw.get("periodicidad") or "").strip(),
            justificacion=str(raw.get("justificacion") or "").strip(),
        )


@dataclass
class CatalogDocument:
    table: str
    dqcs: list[CatalogEntry]
    dataset: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CatalogDocument":
        items = raw.get("dqcs") or raw.get("checks") or []
        return cls(
            table=str(raw.get("table") or TABLE_NAME),
            dataset=str(raw.get("dataset") or raw.get("source") or ""),
            notes=str(raw.get("notes") or ""),
            dqcs=[CatalogEntry.from_dict(x) for x in items if isinstance(x, dict)],
        )


@dataclass
class FieldUniverse:
    table: str
    pk_column: str
    auditable_fields: tuple[str, ...]
    all_columns: tuple[str, ...]
    field_specs: dict[str, FieldSpec]
    coherence_groups: tuple[tuple[str, ...], ...]

    @property
    def required_coverage(self) -> set[str]:
        return set(self.auditable_fields)


def _infer_dtype(col: str) -> str:
    from training.dq.coherence_rules import _INT_COLS, _REAL_COLS

    if col in _INT_COLS:
        return "integer"
    if col in _REAL_COLS:
        return "real"
    return "text"


def load_csv_columns(csv_path: Path = DEFAULT_CSV) -> list[str]:
    with csv_path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh).fieldnames or [])


def build_coherence_groups() -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for rule in RULES:
        if len(rule.columns) < 2:
            continue
        key = tuple(sorted(rule.columns))
        if key not in seen:
            seen.add(key)
            groups.append(rule.columns)
    return tuple(groups)


def load_field_universe(
    csv_path: Path = DEFAULT_CSV,
    universe_path: Path | None = DEFAULT_UNIVERSE_PATH,
) -> FieldUniverse:
    columns = load_csv_columns(csv_path)
    specs: dict[str, FieldSpec] = {}
    meta: dict[str, Any] = {}

    if universe_path and universe_path.exists():
        meta = json.loads(universe_path.read_text(encoding="utf-8"))
        for item in meta.get("fields", []):
            name = item["name"]
            specs[name] = FieldSpec(
                name=name,
                dtype=item.get("dtype", _infer_dtype(name)),
                group=item.get("group", ""),
                description=item.get("description", ""),
            )

    for col in columns:
        if col not in specs:
            specs[col] = FieldSpec(name=col, dtype=_infer_dtype(col))

    auditable = tuple(c for c in columns if c not in EXCLUDED_FROM_COVERAGE)
    groups: tuple[tuple[str, ...], ...] = build_coherence_groups()
    for g in meta.get("coherence_groups") or []:
        tup = tuple(g)
        if len(tup) >= 2 and tup not in groups:
            groups = groups + (tup,)

    return FieldUniverse(
        table=TABLE_NAME,
        pk_column=PK_COLUMN,
        auditable_fields=auditable,
        all_columns=tuple(columns),
        field_specs=specs,
        coherence_groups=groups,
    )


def extract_catalog(text: str) -> CatalogDocument | None:
    """Parse a catalog JSON from raw LLM output."""
    text = text.strip()
    if not text:
        return None

    # Direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and ("dqcs" in obj or "checks" in obj):
            return CatalogDocument.from_dict(obj)
    except json.JSONDecodeError:
        pass

    # ```json fence
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return CatalogDocument.from_dict(obj)
        except json.JSONDecodeError:
            continue

    # Largest balanced object containing "dqcs"
    best: dict | None = None
    for m in re.finditer(r"\{", text):
        start = m.start()
        depth = 0
        in_str = False
        esc = False
        quote = ""
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == quote:
                    in_str = False
            else:
                if ch in ("'", '"'):
                    in_str, esc, quote = True, False, ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = text[start : i + 1]
                        if "dqcs" in chunk or "checks" in chunk:
                            try:
                                obj = json.loads(chunk)
                                if isinstance(obj, dict):
                                    if best is None or len(chunk) > len(json.dumps(best)):
                                        best = obj
                            except json.JSONDecodeError:
                                pass
                        break
    if best:
        return CatalogDocument.from_dict(best)
    return None


def sql_column_refs(sql: str) -> set[str]:
    """Extract likely column identifiers from SQL."""
    if not sql:
        return set()
    refs = set(re.findall(r"\b([A-Z_][A-Z0-9_]{2,})\b", sql.upper()))
    return refs - _SQL_KEYWORDS


def catalog_template(universe: FieldUniverse | None = None) -> dict[str, Any]:
    universe = universe or load_field_universe()
    return {
        "dataset": "recuperatory_cycles",
        "table": universe.table,
        "pk_column": universe.pk_column,
        "notes": "Complete catalog — every auditable field must appear in campos_entrada of at least one DQC.",
        "dqcs": [
            {
                "dqc_id": "DQC_<VARIABLE>_001",
                "variable": "<campo_principal>",
                "descripcion": "<qué verifica>",
                "tipo": "consistencia",
                "severidad": "bloqueante",
                "regla_sql": f"SELECT {universe.pk_column} FROM {universe.table} WHERE ...",
                "condicion_error": "<cuándo falla>",
                "campos_entrada": ["CAMPO_A", "CAMPO_B"],
                "referencia_regulatoria": "<artículo o 'Sin referencia'>",
                "umbral": "",
                "periodicidad": "mensual",
                "justificacion": "<por qué importa>",
            }
        ],
        "field_universe": list(universe.auditable_fields),
    }

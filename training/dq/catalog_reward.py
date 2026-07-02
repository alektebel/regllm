"""Verifiable reward for LLM-generated DQC catalogs.

Measures:
  r_parse     — valid catalog JSON extracted
  r_schema    — each entry has dqc_id, descripcion, campos_entrada
  r_coverage  — fraction of auditable fields covered (target: 1.0)
  r_coherence — per-entry field/SQL coherence (multi-field rules, valid fields)
  r_groups    — known cross-field groups from coherence_rules have ≥1 DQC
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from training.dq.catalog_schema import (
    MULTI_FIELD_TIPOS,
    CatalogDocument,
    CatalogEntry,
    FieldUniverse,
    extract_catalog,
    load_field_universe,
    sql_column_refs,
)
from training.dq.coherence_rules import load_toy_db

WEIGHTS: dict[str, float] = {
    "r_parse": 0.10,
    "r_schema": 0.15,
    "r_coverage": 0.35,
    "r_coherence": 0.30,
    "r_groups": 0.10,
}

VALID_SEVERITIES = {"bloqueante", "advertencia", "informativo", "high", "med", "low"}


@dataclass
class EntryCoherenceReport:
    dqc_id: str
    score: float
    issues: list[str] = field(default_factory=list)
    declared_fields: list[str] = field(default_factory=list)
    sql_fields: list[str] = field(default_factory=list)


@dataclass
class CatalogRewardBreakdown:
    total: float
    components: dict[str, float]
    coverage_pct: float
    covered_fields: list[str]
    missing_fields: list[str]
    duplicate_ids: list[str]
    entry_reports: list[EntryCoherenceReport]
    n_dqcs: int
    catalog: CatalogDocument | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "components": self.components,
            "coverage_pct": round(self.coverage_pct, 4),
            "covered_fields": self.covered_fields,
            "missing_fields": self.missing_fields,
            "duplicate_ids": self.duplicate_ids,
            "n_dqcs": self.n_dqcs,
            "error": self.error,
            "entry_reports": [
                {
                    "dqc_id": r.dqc_id,
                    "score": round(r.score, 4),
                    "issues": r.issues,
                    "declared_fields": r.declared_fields,
                    "sql_fields": r.sql_fields,
                }
                for r in self.entry_reports
            ],
        }


def _score_schema(entries: list[CatalogEntry]) -> tuple[float, list[str]]:
    if not entries:
        return 0.0, ["empty catalog"]
    issues: list[str] = []
    ok = 0
    for e in entries:
        if not e.dqc_id:
            issues.append("entry missing dqc_id")
        elif not e.descripcion:
            issues.append(f"{e.dqc_id}: missing descripcion")
        elif not e.campos_entrada:
            issues.append(f"{e.dqc_id}: empty campos_entrada")
        else:
            ok += 1
    return ok / len(entries), issues


def _score_coverage(
    entries: list[CatalogEntry],
    universe: FieldUniverse,
) -> tuple[float, set[str], set[str]]:
    covered: set[str] = set()
    valid = set(universe.auditable_fields)
    for e in entries:
        for f in e.campos_entrada:
            if f in valid:
                covered.add(f)
    required = universe.required_coverage
    missing = required - covered
    pct = len(covered) / len(required) if required else 1.0
    return pct, covered, missing


def _score_entry_coherence(
    entry: CatalogEntry,
    universe: FieldUniverse,
    conn: sqlite3.Connection | None,
) -> EntryCoherenceReport:
    issues: list[str] = []
    score_parts: list[float] = []
    valid = set(universe.all_columns)
    auditable = set(universe.auditable_fields)

    declared = [f for f in entry.campos_entrada if f]
    unknown_declared = [f for f in declared if f not in valid]
    if unknown_declared:
        issues.append(f"unknown campos_entrada: {unknown_declared}")
        score_parts.append(0.0)
    else:
        score_parts.append(1.0)

    # Multi-field types require ≥2 auditable fields
    tipo = entry.tipo.lower()
    aud = [f for f in declared if f in auditable]
    if tipo in MULTI_FIELD_TIPOS and len(aud) < 2:
        issues.append(f"tipo={tipo} requires ≥2 campos_entrada, got {len(aud)}")
        score_parts.append(0.0)
    elif tipo in MULTI_FIELD_TIPOS:
        score_parts.append(1.0)
    else:
        score_parts.append(1.0 if aud else 0.0)

    sql_fields = sorted(sql_column_refs(entry.regla_sql) - {universe.pk_column.upper()})
    declared_upper = {f.upper() for f in declared}
    sql_in_declared = [f for f in sql_fields if f in declared_upper]
    if entry.regla_sql and sql_fields:
        recall = len(sql_in_declared) / len(sql_fields)
        if recall < 0.5:
            issues.append(f"SQL fields {sql_fields} mostly absent from campos_entrada")
        score_parts.append(recall)
    elif entry.regla_sql:
        issues.append("SQL present but no column refs detected")
        score_parts.append(0.3)
    else:
        issues.append("missing regla_sql")
        score_parts.append(0.0)

    if entry.severidad and entry.severidad.lower() not in VALID_SEVERITIES:
        issues.append(f"unknown severidad: {entry.severidad}")
        score_parts.append(0.5)
    else:
        score_parts.append(1.0 if entry.severidad else 0.7)

    if conn and entry.regla_sql.strip().upper().startswith("SELECT"):
        sql = entry.regla_sql.strip().rstrip(";")
        table = universe.table
        if table.upper() not in sql.upper():
            issues.append(f"SQL should reference table {table}")
            score_parts.append(0.5)
        else:
            score_parts.append(1.0)
        try:
            conn.execute(f"EXPLAIN {sql}").fetchall()
            score_parts.append(1.0)
        except sqlite3.Error as exc:
            issues.append(f"SQL parse error: {exc}")
            score_parts.append(0.0)
    elif entry.regla_sql:
        score_parts.append(0.5)

    final = sum(score_parts) / len(score_parts) if score_parts else 0.0
    return EntryCoherenceReport(
        dqc_id=entry.dqc_id or "?",
        score=final,
        issues=issues,
        declared_fields=declared,
        sql_fields=sql_fields,
    )


def _score_groups(entries: list[CatalogEntry], universe: FieldUniverse) -> tuple[float, list[str]]:
    if not universe.coherence_groups:
        return 1.0, []

    covered_groups = 0
    missing_group_reports: list[str] = []
    for group in universe.coherence_groups:
        group_set = set(group)
        hit = False
        for e in entries:
            if group_set.issubset(set(e.campos_entrada)):
                hit = True
                break
        if hit:
            covered_groups += 1
        else:
            missing_group_reports.append("+".join(group))
    return covered_groups / len(universe.coherence_groups), missing_group_reports


def score_catalog(
    catalog: CatalogDocument | None,
    universe: FieldUniverse | None = None,
    *,
    min_coverage: float = 1.0,
) -> CatalogRewardBreakdown:
    universe = universe or load_field_universe()
    components = {k: 0.0 for k in WEIGHTS}
    empty_reports: list[EntryCoherenceReport] = []

    if catalog is None:
        return CatalogRewardBreakdown(
            total=-1.0,
            components=components,
            coverage_pct=0.0,
            covered_fields=[],
            missing_fields=sorted(universe.auditable_fields),
            duplicate_ids=[],
            entry_reports=empty_reports,
            n_dqcs=0,
            error="failed to parse catalog JSON",
        )

    components["r_parse"] = 1.0
    entries = catalog.dqcs

    schema_score, schema_issues = _score_schema(entries)
    components["r_schema"] = schema_score

    coverage_pct, covered, missing = _score_coverage(entries, universe)
    components["r_coverage"] = coverage_pct

    conn = load_toy_db()
    entry_reports = [_score_entry_coherence(e, universe, conn) for e in entries]
    components["r_coherence"] = (
        sum(r.score for r in entry_reports) / len(entry_reports) if entry_reports else 0.0
    )

    group_score, missing_groups = _score_groups(entries, universe)
    components["r_groups"] = group_score

    ids = [e.dqc_id for e in entries if e.dqc_id]
    duplicate_ids = sorted({i for i in ids if ids.count(i) > 1})

    raw = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    total = round(2.0 * raw - 1.0, 4)

    # Hard gate: sub-100% coverage caps total when min_coverage=1.0
    if min_coverage >= 1.0 and coverage_pct < 1.0:
        cap = round(2.0 * (raw - 0.35 * (1.0 - coverage_pct)) - 1.0, 4)
        total = min(total, cap)

    error_parts = schema_issues[:5]
    if missing_groups:
        error_parts.append(f"missing coherence groups: {missing_groups[:5]}")
    if duplicate_ids:
        error_parts.append(f"duplicate dqc_id: {duplicate_ids}")

    return CatalogRewardBreakdown(
        total=total,
        components={k: round(v, 4) for k, v in components.items()},
        coverage_pct=coverage_pct * 100.0,
        covered_fields=sorted(covered),
        missing_fields=sorted(missing),
        duplicate_ids=duplicate_ids,
        entry_reports=entry_reports,
        n_dqcs=len(entries),
        catalog=catalog,
        error="; ".join(error_parts) if error_parts else None,
    )


def score_catalog_text(
    text: str,
    universe: FieldUniverse | None = None,
    **kwargs: Any,
) -> CatalogRewardBreakdown:
    return score_catalog(extract_catalog(text), universe, **kwargs)

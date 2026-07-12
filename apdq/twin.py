"""Manifest-driven synthetic twin generator.

Builds a SQLite database that satisfies *every* invariant the manifest
declares, by construction:

* source columns are sampled from their declared domains;
* foreign keys are sampled from actual parent keys;
* date-ordering chains are sorted into compliance;
* declared constraints are enforced by rejection sampling (the manifest
  author gets a clear error if a constraint is unsatisfiable in practice);
* derived columns are computed by evaluating their documented formulas;
* reconciliation surfaces mirror the facts they must agree with;
* control-total tables are computed from the generated facts.

Because the twin is clean by construction, any oracle that fires on it is
a bug in the oracle — the same contract as ``DQC/eval/generate_db.py``,
generalized from one hand-coded schema to any manifest.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta

from . import expr
from .manifest import Column, Manifest, ManifestError, Table, derivation_order

MAX_ROW_RETRIES = 200
NULL_RATE = 0.05


@dataclass
class Twin:
    conn: sqlite3.Connection
    manifest: Manifest
    rows: dict[str, list[dict]] = field(default_factory=dict)
    seed: int = 0


# ── sampling ─────────────────────────────────────────────────────────────────

def _sample_source(col: Column, rng: random.Random, idx: int,
                   unique_seen: set) -> object:
    dom = col.domain
    if dom.nullable and rng.random() < dom.null_rate:
        return None
    if dom.values:
        return rng.choice(list(dom.values))
    if dom.type == "int":
        lo = int(dom.min) if dom.min is not None else 0
        hi = int(dom.max) if dom.max is not None else 100
        if dom.unique:
            return lo + idx
        return rng.randint(lo, hi)
    if dom.type == "real":
        lo = dom.min if dom.min is not None else 0.0
        hi = dom.max if dom.max is not None else 1.0
        return round(rng.uniform(lo, hi), 6)
    if dom.type == "yyyymm":
        lo = int(dom.min) if dom.min is not None else 201501
        hi = int(dom.max) if dom.max is not None else 202412
        months = _months_between(lo, hi)
        return _month_shift(lo, rng.randint(0, max(months, 0)))
    if dom.type == "date":
        lo = date.fromisoformat(str(dom.min) if dom.min else "2015-01-01")
        hi = date.fromisoformat(str(dom.max) if dom.max else "2024-12-28")
        span = max((hi - lo).days, 0)
        return (lo + timedelta(days=rng.randint(0, span))).isoformat()
    # text
    if dom.unique:
        return f"{col.name[:4]}{idx:07d}"
    return f"{col.name[:4]}{rng.randint(0, 99999):05d}"


def _month_shift(yyyymm: int, months: int) -> int:
    y, m = divmod(yyyymm, 100)
    total = y * 12 + (m - 1) + months
    return (total // 12) * 100 + (total % 12) + 1


def _months_between(lo: int, hi: int) -> int:
    return (hi // 100 - lo // 100) * 12 + (hi % 100 - lo % 100)


def _apply_orderings(row: dict, table: Table) -> None:
    """Sort each declared date chain's non-null values into compliance."""
    for chain in table.date_orderings:
        present = [c for c in chain if row.get(c) is not None]
        vals = sorted(row[c] for c in present)
        for col_name, val in zip(present, vals):
            row[col_name] = val


def _constraints_hold(row: dict, table: Table) -> bool:
    for constraint in table.constraints:
        val = expr.eval_row(constraint.ast, row)
        if val is not None and not val:
            return False
    return True


def _orderings_hold(row: dict, table: Table) -> bool:
    for chain in table.date_orderings:
        present = [row[c] for c in chain if row.get(c) is not None]
        if present != sorted(present):
            return False
    return True


def _derive(row: dict, ordered: list[Column],
            rng: random.Random | None = None, idx: int = 0) -> dict:
    """Apply derivations in order. Conditionally-derived columns (when:)
    take the formula value where the condition holds; elsewhere the value
    is free — sampled when a rng is supplied, kept as-is otherwise (so
    defect planting can re-derive without disturbing free branches)."""
    for col in ordered:
        if col.when_ast is not None:
            cond = expr.eval_row(col.when_ast, row)
            if cond is not None and cond:
                row[col.name] = expr.eval_row(col.formula_ast, row)
            elif rng is not None:
                row[col.name] = _sample_source(col, rng, idx, set())
            # else: leave the existing free-branch value untouched
        else:
            row[col.name] = expr.eval_row(col.formula_ast, row)
        if isinstance(row[col.name], float):
            row[col.name] = round(row[col.name], 6)
    return row


def _make_row(table: Table, ordered_derived: list[Column],
              parents: dict[str, list[dict]], parent_pk: dict[str, str],
              rng: random.Random, idx: int) -> dict:
    fk_cols = {fk.column: fk for fk in table.foreign_keys}
    for _ in range(MAX_ROW_RETRIES):
        row: dict = {}
        for col in table.columns:
            if col.role != "source":
                continue
            if col.name in fk_cols:
                fk = fk_cols[col.name]
                parent_rows = parents[fk.ref_table]
                row[col.name] = rng.choice(parent_rows)[fk.ref_column]
            else:
                row[col.name] = _sample_source(col, rng, idx, set())
        _apply_orderings(row, table)
        row = _derive(row, ordered_derived, rng, idx)
        if _constraints_hold(row, table) and _orderings_hold(row, table):
            return row
    raise ManifestError(
        f"table {table.name}: could not satisfy declared constraints in "
        f"{MAX_ROW_RETRIES} samples — a constraint is (nearly) unsatisfiable "
        f"under the declared domains; tighten domains or relax the constraint")


def _make_series(table: Table, ordered_derived: list[Column],
                 parents: dict[str, list[dict]], parent_pk: dict[str, str],
                 rng: random.Random, series_idx: int, row_idx: int,
                 forced_series_key=None) -> list[dict]:
    """One panel series: `periods` consecutive months for one series key,
    cumulative columns non-decreasing, other sources held constant."""
    panel = table.panel
    base = _make_row(table, ordered_derived, parents, parent_pk,
                     rng, row_idx)
    if forced_series_key is not None:
        base[panel.series_key] = forced_series_key
    period_col = table.column(panel.period_column)
    lo = int(period_col.domain.min or 201501)
    hi = int(period_col.domain.max or 202412)
    start_slack = max(_months_between(lo, hi) - panel.periods + 1, 0)
    start = _month_shift(lo, rng.randint(0, start_slack))

    cums = {c: 0.0 for c in panel.cumulative_columns}
    increments = {}
    for cname in panel.cumulative_columns:
        dom = table.column(cname).domain
        span = float(dom.max if dom and dom.max is not None else 1000.0)
        increments[cname] = span / max(panel.periods, 1)

    rows = []
    for p in range(panel.periods):
        row = dict(base)
        row[table.primary_key] = f"{base[table.primary_key]}_{p:02d}"
        row[panel.period_column] = _month_shift(start, p)
        for cname in panel.cumulative_columns:
            cums[cname] = round(
                cums[cname] + rng.uniform(0, increments[cname]), 6)
            row[cname] = cums[cname]
        # rng passed so when-columns re-sample their free branch per
        # period — cumulative columns move conditions across thresholds
        row = _derive(row, ordered_derived, rng, row_idx + p)
        if not _constraints_hold(row, table):
            # per-period values changed by the series walk must not break
            # declared constraints — that would poison the clean twin; the
            # author should express the rule as a when-derivation instead
            raise ManifestError(
                f"table {table.name}: a constraint fails on generated panel "
                f"rows; per-period rules must be when-derivations, not "
                f"constraints over cumulative columns")
        rows.append(row)
    return rows


# ── materialization ──────────────────────────────────────────────────────────

def _topo_tables(manifest: Manifest) -> list[Table]:
    remaining = {t.name: t for t in manifest.tables}
    done: set[str] = set()
    ordered: list[Table] = []
    while remaining:
        progressed = False
        for name in list(remaining):
            deps = {fk.ref_table for fk in remaining[name].foreign_keys
                    if fk.ref_table != name}
            if deps <= done:
                ordered.append(remaining.pop(name))
                done.add(name)
                progressed = True
        if not progressed:
            raise ManifestError(
                f"circular foreign-key dependency among {sorted(remaining)}")
    return ordered


def _sql_type(col: Column) -> str:
    if col.domain is None:               # fully-derived: numeric by default
        return "REAL"
    return {"int": "INTEGER", "yyyymm": "INTEGER",
            "real": "REAL", "text": "TEXT", "date": "TEXT"}[col.domain.type]


def create_table_sql(table: Table) -> str:
    # No PK/NOT NULL constraints on purpose: mutations must be plantable
    # (duplicates, orphans, nulls) exactly as in production extracts.
    cols = ", ".join(f"{c.name} {_sql_type(c)}" for c in table.columns)
    return f"CREATE TABLE {table.name} ({cols})"


def _insert_rows(conn: sqlite3.Connection, table_name: str,
                 col_names: list[str], rows: list[dict]) -> None:
    placeholders = ", ".join("?" for _ in col_names)
    conn.executemany(
        f"INSERT INTO {table_name} ({', '.join(col_names)}) "
        f"VALUES ({placeholders})",
        [tuple(r.get(c) for c in col_names) for r in rows])


def _surface_plan(table: Table) -> dict[str, dict]:
    """surface name -> {join_column, columns:[(fact_col, surface_col)]}"""
    plan: dict[str, dict] = {}
    for col in table.columns:
        for recon in col.reconcile:
            entry = plan.setdefault(
                recon.surface, {"join_column": recon.join_column, "columns": []})
            if entry["join_column"] != recon.join_column:
                raise ManifestError(
                    f"{table.name}: surface {recon.surface!r} declared with "
                    f"two different join columns")
            entry["columns"].append((col.name, recon.column))
    return plan


def _materialize_surfaces(conn: sqlite3.Connection, table: Table,
                          rows: list[dict]) -> None:
    for surface, spec in _surface_plan(table).items():
        join_col = spec["join_column"]
        surf_cols = [surface_col for _, surface_col in spec["columns"]]
        # typeless columns (BLOB affinity): values stored exactly as the
        # fact table holds them, whether numeric or text
        ddl_cols = ", ".join([join_col] + surf_cols)
        conn.execute(f"CREATE TABLE {surface} ({ddl_cols})")
        out = []
        for row in rows:
            rec = {join_col: row[join_col]}
            for fact_col, surface_col in spec["columns"]:
                rec[surface_col] = row[fact_col]
            out.append(rec)
        _insert_rows(conn, surface, [join_col] + surf_cols, out)


def _materialize_control(conn: sqlite3.Connection, table: Table,
                         rows: list[dict]) -> None:
    if not table.control:
        return
    conn.execute(
        f"CREATE TABLE {table.control.surface} (metric TEXT, value REAL)")
    records = []
    for check in table.control.checks:
        if check.kind == "count":
            records.append(("count", float(len(rows))))
        else:
            total = sum(float(r[check.column] or 0.0) for r in rows)
            records.append((f"sum:{check.column}", round(total, 6)))
    conn.executemany(
        f"INSERT INTO {table.control.surface} VALUES (?, ?)", records)


def build_twin(manifest: Manifest, seed: int = 42,
               conn: sqlite3.Connection | None = None) -> Twin:
    rng = random.Random(seed)
    if conn is None:
        conn = sqlite3.connect(":memory:")
    expr.register_sql_functions(conn)

    parent_pk = {t.name: t.primary_key for t in manifest.tables}
    all_rows: dict[str, list[dict]] = {}

    for table in _topo_tables(manifest):
        conn.execute(create_table_sql(table))
        ordered_derived = derivation_order(table)
        if table.panel:
            n_series = max(table.rows // table.panel.periods, 1)
            # series keys must be distinct or two series would merge and
            # break the duplicate-period invariant on the clean twin
            fk_map = {fk.column: fk for fk in table.foreign_keys}
            if table.panel.series_key in fk_map:
                fk = fk_map[table.panel.series_key]
                pool = [r[fk.ref_column] for r in all_rows[fk.ref_table]]
                if n_series > len(pool):
                    raise ManifestError(
                        f"{table.name}: panel needs {n_series} distinct "
                        f"series keys but {fk.ref_table} has {len(pool)}")
                keys = rng.sample(pool, n_series)
            else:
                keys = [f"SER_{s:06d}" for s in range(n_series)]
            rows = []
            for s in range(n_series):
                rows.extend(_make_series(
                    table, ordered_derived, all_rows, parent_pk, rng,
                    s, len(rows), forced_series_key=keys[s]))
        else:
            rows = [
                _make_row(table, ordered_derived, all_rows, parent_pk,
                          rng, idx)
                for idx in range(table.rows)
            ]
        all_rows[table.name] = rows
        _insert_rows(conn, table.name, table.column_names, rows)
        _materialize_surfaces(conn, table, rows)
        _materialize_control(conn, table, rows)

    conn.commit()
    return Twin(conn=conn, manifest=manifest, rows=all_rows, seed=seed)


def clone_twin(twin: Twin) -> Twin:
    """Fresh connection with identical content (for planting mutations)."""
    clone = sqlite3.connect(":memory:")
    twin.conn.backup(clone)
    expr.register_sql_functions(clone)
    return Twin(conn=clone, manifest=twin.manifest,
                rows={k: [dict(r) for r in v] for k, v in twin.rows.items()},
                seed=twin.seed)

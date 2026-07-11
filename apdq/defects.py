"""Defect-class mutation generators.

Replaces a hand-written defect *catalog* with generators per normative
defect class (docs/AUDITOR_PARITY_STANDARD.md §4). Given a binding
manifest, ``generate_defects`` instantiates, for every applicable
(class × node), a ``GeneratedDefect`` that pairs:

* ``plant(twin, rng, k)`` — plants k violating rows into a twin copy and
  returns the planted primary keys (mutation testing), and
* ``oracle_sql`` — the reference check that must catch exactly them.

Classes implemented generically (MVP): 1 missing value, 2 domain
violation, 3 duplicate key, 4 broken reference, 5 intra-row constraint,
6 derivation error, 7 reconciliation mismatch, 8 population (control
totals), 9 temporal ordering. Class 10 (panel) exists schema-specifically
in DQC/eval and is a documented expansion; class 11 (distributional) is
advisory by design; class 12 (semantic drift) is served by the SAS AST
differ, not by row mutations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from . import expr
from .manifest import Column, Constraint, Manifest, Table, derivation_order
from .twin import Twin

EPS = 0.01

CLASS_SLUGS = {
    1: "missing_value",
    2: "domain_violation",
    3: "duplicate_key",
    4: "broken_reference",
    5: "constraint_violation",
    6: "derivation_error",
    7: "reconciliation_mismatch",
    8: "population",
    9: "temporal_violation",
    10: "panel_inconsistency",
    11: "distributional",
    12: "semantic_drift",
}
GENERIC_CLASSES = (1, 2, 3, 4, 5, 6, 7, 8, 9)


@dataclass
class GeneratedDefect:
    defect_id: str
    dq_class: int
    table: str
    columns: tuple[str, ...]
    description: str
    oracle_sql: str                       # projects the PK, or aggregate
    plant: Callable[[Twin, random.Random, int], list]
    aggregate: bool = False               # no per-row attribution
    regulation_refs: tuple[str, ...] = ()

    @property
    def class_slug(self) -> str:
        return CLASS_SLUGS[self.dq_class]


class PlantingError(RuntimeError):
    """A generator could not produce a violating row (manifest needs a
    ``plant:`` hint or wider domains)."""


# ── helpers ──────────────────────────────────────────────────────────────────

def _fresh_pk(twin: Twin, table: Table, tag: str, i: int) -> object:
    pk_col = table.column(table.primary_key)
    if pk_col.domain and pk_col.domain.type in ("int", "yyyymm"):
        existing = max((int(r[table.primary_key] or 0)
                        for r in twin.rows[table.name]), default=0)
        return existing + 1_000_000 + i
    return f"APDQ_{tag}_{i:04d}"


def _insert_row(twin: Twin, table: Table, row: dict) -> None:
    cols = table.column_names
    placeholders = ", ".join("?" for _ in cols)
    twin.conn.execute(
        f"INSERT INTO {table.name} ({', '.join(cols)}) "
        f"VALUES ({placeholders})",
        tuple(row.get(c) for c in cols))


def _pick_clean(twin: Twin, table: Table, rng: random.Random,
                predicate=None) -> dict:
    rows = twin.rows[table.name]
    candidates = [r for r in rows if predicate(r)] if predicate else rows
    if not candidates:
        raise PlantingError(
            f"{table.name}: no clean row satisfies the planting predicate")
    return dict(rng.choice(candidates))


def _rederive(row: dict, table: Table) -> dict:
    for col in derivation_order(table):
        row[col.name] = expr.eval_row(col.formula_ast, row)
        if isinstance(row[col.name], float):
            row[col.name] = round(row[col.name], 6)
    return row


def _plant_rows(twin: Twin, table: Table, tag: str, k: int,
                rng: random.Random, mutate: Callable[[dict], dict],
                predicate=None) -> list:
    """Standard planter: k mutated copies of distinct clean rows, each
    under a fresh PK. Returns the planted PKs (the attribution keys)."""
    planted = []
    for i in range(k):
        row = _pick_clean(twin, table, rng, predicate)
        row = mutate(row)
        pk = _fresh_pk(twin, table, tag, i)
        row[table.primary_key] = pk
        _insert_row(twin, table, row)
        planted.append(pk)
    twin.conn.commit()
    return planted


def _sql_literal(value: object) -> str:
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(value)


# ── class 1: missing value ───────────────────────────────────────────────────

def _gen_missing(table: Table, col: Column) -> GeneratedDefect:
    is_pk = col.name == table.primary_key

    def plant(twin: Twin, rng: random.Random, k: int) -> list:
        if is_pk:
            # a NULL key cannot be attributed by key — aggregate defect
            for _ in range(k):
                row = _pick_clean(twin, table, rng)
                row[table.primary_key] = None
                _insert_row(twin, table, row)
            twin.conn.commit()
            return []

        def mutate(row: dict) -> dict:
            row[col.name] = None
            return row
        return _plant_rows(twin, table, f"C01_{col.name}", k, rng, mutate)

    return GeneratedDefect(
        defect_id=f"C01:{table.name}.{col.name}",
        dq_class=1, table=table.name, columns=(col.name,),
        description=f"{col.name} ({col.concept}) is mandatory but missing",
        oracle_sql=(f"SELECT {table.primary_key} FROM {table.name} "
                    f"WHERE {col.name} IS NULL"),
        plant=plant, aggregate=is_pk,
        regulation_refs=col.regulation_refs)


# ── class 2: domain violation ────────────────────────────────────────────────

def _domain_predicate_sql(col: Column) -> str:
    dom = col.domain
    if dom.values is not None:
        vals = ", ".join(_sql_literal(v) for v in dom.values)
        return f"{col.name} NOT IN ({vals})"
    clauses = []
    if dom.min is not None:
        clauses.append(f"{col.name} < {dom.min}")
    if dom.max is not None:
        clauses.append(f"{col.name} > {dom.max}")
    return " OR ".join(clauses)


def _violating_value(col: Column, rng: random.Random) -> object:
    dom = col.domain
    if dom.values is not None:
        if dom.type in ("int", "real"):
            numeric = [v for v in dom.values if isinstance(v, (int, float))]
            return (max(numeric) + 7) if numeric else -1
        return "APDQ_INVALID"
    if dom.max is not None:
        return dom.max + (abs(dom.max) or 1) * 0.5 + 3
    if dom.min is not None:
        return dom.min - (abs(dom.min) or 1) * 0.5 - 3
    raise PlantingError(f"{col.name}: domain has no bounds/values to violate")


def _gen_domain(table: Table, col: Column) -> GeneratedDefect:
    def plant(twin: Twin, rng: random.Random, k: int) -> list:
        def mutate(row: dict) -> dict:
            row[col.name] = _violating_value(col, rng)
            return row
        return _plant_rows(twin, table, f"C02_{col.name}", k, rng, mutate)

    return GeneratedDefect(
        defect_id=f"C02:{table.name}.{col.name}",
        dq_class=2, table=table.name, columns=(col.name,),
        description=f"{col.name} ({col.concept}) outside its declared domain",
        oracle_sql=(f"SELECT {table.primary_key} FROM {table.name} "
                    f"WHERE {col.name} IS NOT NULL "
                    f"AND ({_domain_predicate_sql(col)})"),
        plant=plant, regulation_refs=col.regulation_refs)


# ── class 3: duplicate business key ──────────────────────────────────────────

def _gen_duplicate(table: Table) -> GeneratedDefect:
    def plant(twin: Twin, rng: random.Random, k: int) -> list:
        planted = []
        rows = twin.rows[table.name]
        for row in rng.sample(rows, min(k, len(rows))):
            _insert_row(twin, table, dict(row))    # verbatim re-insert
            planted.append(row[table.primary_key])
        twin.conn.commit()
        return planted

    return GeneratedDefect(
        defect_id=f"C03:{table.name}.{table.primary_key}",
        dq_class=3, table=table.name, columns=(table.primary_key,),
        description=f"duplicate primary key {table.primary_key}",
        oracle_sql=(f"SELECT {table.primary_key} FROM {table.name} "
                    f"GROUP BY {table.primary_key} HAVING COUNT(*) > 1"),
        plant=plant)


# ── class 4: broken reference ────────────────────────────────────────────────

def _gen_reference(table: Table, fk) -> GeneratedDefect:
    def plant(twin: Twin, rng: random.Random, k: int) -> list:
        def mutate(row: dict) -> dict:
            row[fk.column] = f"APDQ_ORPHAN_{rng.randint(0, 9999):04d}"
            return row
        return _plant_rows(twin, table, f"C04_{fk.column}", k, rng, mutate)

    return GeneratedDefect(
        defect_id=f"C04:{table.name}.{fk.column}",
        dq_class=4, table=table.name, columns=(fk.column,),
        description=(f"{fk.column} references no row in "
                     f"{fk.ref_table}.{fk.ref_column}"),
        oracle_sql=(
            f"SELECT t.{table.primary_key} FROM {table.name} t "
            f"LEFT JOIN {fk.ref_table} p ON t.{fk.column} = p.{fk.ref_column} "
            f"WHERE t.{fk.column} IS NOT NULL AND p.{fk.ref_column} IS NULL"),
        plant=plant)


# ── class 5: declared intra-row constraint ───────────────────────────────────

def _violate_constraint(row: dict, table: Table, constraint: Constraint,
                        rng: random.Random) -> dict:
    if constraint.plant:
        row.update(constraint.plant)
        return row
    # automated search: single-column candidate values, sources re-derived
    refs = sorted(expr.columns(constraint.ast))
    for name in refs:
        col = table.column(name)
        for candidate in _candidates(col, rng):
            trial = dict(row)
            trial[name] = candidate
            if col.role == "source":
                trial = _rederive(trial, table)
                trial[name] = candidate   # re-derivation must not undo it
            val = expr.eval_row(constraint.ast, trial)
            if val is not None and not val:
                return trial
    raise PlantingError(
        f"{table.name}: cannot auto-violate constraint {constraint.id!r}; "
        f"add a 'plant:' hint with explicit violating values")


def _candidates(col: Column, rng: random.Random) -> list:
    if col.role == "derived" or col.domain is None:
        return [0, 1, -1, 999999.0, 0.0001]
    dom = col.domain
    if dom.values is not None:
        return list(dom.values)
    out: list = []
    if dom.min is not None:
        out.append(dom.min)
    if dom.max is not None:
        out.append(dom.max)
    if dom.min is not None and dom.max is not None:
        out.append((dom.min + dom.max) / 2)
    return out or [0, 1, 100]


def _gen_constraint(table: Table, constraint: Constraint) -> GeneratedDefect:
    cols = tuple(sorted(expr.columns(constraint.ast)))

    def plant(twin: Twin, rng: random.Random, k: int) -> list:
        def mutate(row: dict) -> dict:
            return _violate_constraint(row, table, constraint, rng)
        return _plant_rows(twin, table, f"C05_{constraint.id}", k, rng, mutate)

    return GeneratedDefect(
        defect_id=f"C05:{table.name}.{constraint.id}",
        dq_class=5, table=table.name, columns=cols,
        description=constraint.description or f"constraint {constraint.id}",
        oracle_sql=(f"SELECT {table.primary_key} FROM {table.name} "
                    f"WHERE NOT {expr.to_sql(constraint.ast)}"),
        plant=plant, regulation_refs=constraint.regulation_refs)


# ── class 6: derivation error (reperformance) ────────────────────────────────

def _gen_derivation(table: Table, col: Column) -> GeneratedDefect:
    formula_sql = expr.to_sql(col.formula_ast)

    def plant(twin: Twin, rng: random.Random, k: int) -> list:
        def mutate(row: dict) -> dict:
            correct = expr.eval_row(col.formula_ast, row)
            base = float(correct) if correct is not None else 0.0
            row[col.name] = round(base * 1.5 + 1000.0, 6)
            return row
        # only rows where the formula evaluates (inputs present)
        def has_inputs(row: dict) -> bool:
            return expr.eval_row(col.formula_ast, row) is not None
        return _plant_rows(twin, table, f"C06_{col.name}", k, rng, mutate,
                           predicate=has_inputs)

    return GeneratedDefect(
        defect_id=f"C06:{table.name}.{col.name}",
        dq_class=6, table=table.name, columns=(col.name,),
        description=(f"{col.name} ({col.concept}) deviates from its "
                     f"documented formula: {col.formula}"),
        oracle_sql=(f"SELECT {table.primary_key} FROM {table.name} "
                    f"WHERE ABS({col.name} - ({formula_sql})) > {EPS}"),
        plant=plant, regulation_refs=col.regulation_refs)


# ── class 7: reconciliation mismatch ─────────────────────────────────────────

def _gen_reconciliation(table: Table, col: Column, recon) -> GeneratedDefect:
    numeric = col.role == "derived" or col.domain.type in ("int", "real", "yyyymm")
    compare = (f"ABS(t.{col.name} - s.{recon.column}) > {EPS}" if numeric
               else f"t.{col.name} <> s.{recon.column}")

    def plant(twin: Twin, rng: random.Random, k: int) -> list:
        planted = []
        for i in range(k):
            row = _pick_clean(twin, table, rng,
                              predicate=lambda r: r[col.name] is not None)
            original = row[col.name]
            if numeric:
                row[col.name] = round(float(original) * 1.7 + 500.0, 6)
            else:
                row[col.name] = "APDQ_MISMATCH"
            pk = _fresh_pk(twin, table, f"C07_{col.name}", i)
            row[table.primary_key] = pk
            row[recon.join_column] = pk if recon.join_column == table.primary_key \
                else row[recon.join_column]
            _insert_row(twin, table, row)
            # the surface keeps the authoritative (original) value
            twin.conn.execute(
                f"INSERT INTO {recon.surface} ({recon.join_column}, "
                f"{recon.column}) VALUES (?, ?)",
                (row[recon.join_column], original))
            planted.append(pk)
        twin.conn.commit()
        return planted

    return GeneratedDefect(
        defect_id=f"C07:{table.name}.{col.name}~{recon.surface}",
        dq_class=7, table=table.name, columns=(col.name,),
        description=(f"{col.name} ({col.concept}) disagrees with "
                     f"authoritative surface {recon.surface}.{recon.column}"),
        oracle_sql=(
            f"SELECT t.{table.primary_key} FROM {table.name} t "
            f"JOIN {recon.surface} s "
            f"ON t.{recon.join_column} = s.{recon.join_column} "
            f"WHERE t.{col.name} IS NOT NULL AND s.{recon.column} IS NOT NULL "
            f"AND {compare}"),
        plant=plant, regulation_refs=col.regulation_refs)


# ── class 8: population (control totals) ─────────────────────────────────────

def _control_oracle(table: Table) -> str:
    parts = []
    for check in table.control.checks:
        if check.kind == "count":
            parts.append(
                f"SELECT 'count' AS metric FROM "
                f"(SELECT COUNT(*) AS actual FROM {table.name}) a "
                f"JOIN {table.control.surface} c ON c.metric = 'count' "
                f"WHERE ABS(a.actual - c.value) > 0.5")
        else:
            parts.append(
                f"SELECT 'sum:{check.column}' AS metric FROM "
                f"(SELECT COALESCE(SUM({check.column}), 0) AS actual "
                f"FROM {table.name}) a "
                f"JOIN {table.control.surface} c "
                f"ON c.metric = 'sum:{check.column}' "
                f"WHERE ABS(a.actual - c.value) > {EPS}")
    return " UNION ALL ".join(parts)


def _gen_population(table: Table, variant: str) -> GeneratedDefect:
    def plant_missing(twin: Twin, rng: random.Random, k: int) -> list:
        rows = twin.rows[table.name]
        victims = rng.sample(rows, min(k, len(rows)))
        for row in victims:
            twin.conn.execute(
                f"DELETE FROM {table.name} WHERE {table.primary_key} = ?",
                (row[table.primary_key],))
        twin.conn.commit()
        return []                                # aggregate: no PKs

    def plant_fabricated(twin: Twin, rng: random.Random, k: int) -> list:
        def identity(row: dict) -> dict:
            return row
        _plant_rows(twin, table, f"C08_{variant}", k, rng, identity)
        return []

    plant = plant_missing if variant == "missing" else plant_fabricated
    return GeneratedDefect(
        defect_id=f"C08:{table.name}.{variant}",
        dq_class=8, table=table.name, columns=(table.primary_key,),
        description=(f"{table.name}: {'rows missing vs' if variant == 'missing' else 'rows not reflected in'} "
                     f"control totals ({table.control.surface})"),
        oracle_sql=_control_oracle(table),
        plant=plant, aggregate=True)


# ── class 9: temporal ordering ───────────────────────────────────────────────

def _gen_temporal(table: Table, chain: tuple[str, ...],
                  idx: int) -> GeneratedDefect:
    pairs = list(zip(chain, chain[1:]))
    pair_sql = " UNION ALL ".join(
        f"SELECT {table.primary_key} FROM {table.name} "
        f"WHERE {a} IS NOT NULL AND {b} IS NOT NULL AND {a} > {b}"
        for a, b in pairs)

    def plant(twin: Twin, rng: random.Random, k: int) -> list:
        def both_present(row: dict) -> bool:
            return any(row.get(a) is not None and row.get(b) is not None
                       for a, b in pairs)

        def mutate(row: dict) -> dict:
            usable = [(a, b) for a, b in pairs
                      if row.get(a) is not None and row.get(b) is not None]
            a, b = rng.choice(usable)
            row[a] = int(row[b]) + 100      # one year past its successor
            return row
        return _plant_rows(twin, table, f"C09_{idx}", k, rng, mutate,
                           predicate=both_present)

    return GeneratedDefect(
        defect_id=f"C09:{table.name}.{'_'.join(chain)}",
        dq_class=9, table=table.name, columns=tuple(chain),
        description=f"lifecycle dates out of order: {' <= '.join(chain)}",
        oracle_sql=pair_sql, plant=plant)


# ── entry point ──────────────────────────────────────────────────────────────

def generate_defects(manifest: Manifest) -> list[GeneratedDefect]:
    out: list[GeneratedDefect] = []
    for table in manifest.tables:
        for col in table.columns:
            if col.role == "source":
                if not col.domain.nullable:
                    out.append(_gen_missing(table, col))
                if (col.domain.values is not None or col.domain.min is not None
                        or col.domain.max is not None):
                    out.append(_gen_domain(table, col))
            else:
                out.append(_gen_derivation(table, col))
            for recon in col.reconcile:
                out.append(_gen_reconciliation(table, col, recon))
        out.append(_gen_duplicate(table))
        for fk in table.foreign_keys:
            out.append(_gen_reference(table, fk))
        for constraint in table.constraints:
            out.append(_gen_constraint(table, constraint))
        if table.control:
            out.append(_gen_population(table, "missing"))
            out.append(_gen_population(table, "fabricated"))
        for idx, chain in enumerate(table.date_orderings):
            out.append(_gen_temporal(table, chain, idx))
    return out


def defects_by_class(defects: list[GeneratedDefect]) -> dict[int, list]:
    by: dict[int, list] = {}
    for d in defects:
        by.setdefault(d.dq_class, []).append(d)
    return by

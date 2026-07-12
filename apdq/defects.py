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
    """Recompute derivations; conditionally-derived columns keep their
    free-branch value where the condition does not hold."""
    for col in derivation_order(table):
        if col.when_ast is not None:
            cond = expr.eval_row(col.when_ast, row)
            if cond is None or not cond:
                continue
        row[col.name] = expr.eval_row(col.formula_ast, row)
        if isinstance(row[col.name], float):
            row[col.name] = round(row[col.name], 6)
    return row


def _formula_is_text(table: Table, ast) -> bool:
    """Whether a formula produces text (equality oracle) rather than a
    number (tolerance oracle). Handles the shapes manifests use: string
    literals, text-column passthroughs, if() over either."""
    if ast.kind == "str":
        return True
    if ast.kind == "col":
        col = table.column(str(ast.value))
        return bool(col.domain and col.domain.type in ("text", "date"))
    if ast.kind == "func" and ast.value == "if":
        return (_formula_is_text(table, ast.children[1])
                or _formula_is_text(table, ast.children[2]))
    return False


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

def _when_false(col: Column):
    """Predicate: the column's `when` condition strictly evaluates false
    (SQL's NOT(when) excludes NULL conditions, so planting must too)."""
    def pred(row: dict) -> bool:
        val = expr.eval_row(col.when_ast, row)
        return val is not None and not val
    return pred


def _free_branch_guard(col: Column) -> str:
    """SQL guard for a when-column's free branch."""
    return f"NOT {expr.to_sql(col.when_ast)} AND "


def _gen_missing(table: Table, col: Column) -> GeneratedDefect:
    is_pk = col.name == table.primary_key
    guard = _free_branch_guard(col) if col.when_ast is not None else ""
    predicate = _when_false(col) if col.when_ast is not None else None

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
        return _plant_rows(twin, table, f"C01_{col.name}", k, rng, mutate,
                           predicate=predicate)

    return GeneratedDefect(
        defect_id=f"C01:{table.name}.{col.name}",
        dq_class=1, table=table.name, columns=(col.name,),
        description=f"{col.name} ({col.concept}) is mandatory but missing",
        oracle_sql=(f"SELECT {table.primary_key} FROM {table.name} "
                    f"WHERE {guard}{col.name} IS NULL"),
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
        clauses.append(f"{col.name} < {_sql_literal(dom.min)}")
    if dom.max is not None:
        clauses.append(f"{col.name} > {_sql_literal(dom.max)}")
    return " OR ".join(clauses)


def _violating_value(col: Column, rng: random.Random) -> object:
    dom = col.domain
    if dom.values is not None:
        if dom.type in ("int", "real"):
            numeric = [v for v in dom.values if isinstance(v, (int, float))]
            return (max(numeric) + 7) if numeric else -1
        return "APDQ_INVALID"
    if dom.type == "date":
        return "9999-01-01" if dom.max is not None else "1000-01-01"
    if dom.max is not None:
        return dom.max + (abs(dom.max) or 1) * 0.5 + 3
    if dom.min is not None:
        return dom.min - (abs(dom.min) or 1) * 0.5 - 3
    raise PlantingError(f"{col.name}: domain has no bounds/values to violate")


def _gen_domain(table: Table, col: Column) -> GeneratedDefect:
    guard = _free_branch_guard(col) if col.when_ast is not None else ""
    predicate = _when_false(col) if col.when_ast is not None else None

    def plant(twin: Twin, rng: random.Random, k: int) -> list:
        def mutate(row: dict) -> dict:
            row[col.name] = _violating_value(col, rng)
            return row
        return _plant_rows(twin, table, f"C02_{col.name}", k, rng, mutate,
                           predicate=predicate)

    return GeneratedDefect(
        defect_id=f"C02:{table.name}.{col.name}",
        dq_class=2, table=table.name, columns=(col.name,),
        description=f"{col.name} ({col.concept}) outside its declared domain",
        oracle_sql=(f"SELECT {table.primary_key} FROM {table.name} "
                    f"WHERE {guard}{col.name} IS NOT NULL "
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

    def try_assignment(assign: dict) -> dict | None:
        trial = dict(row)
        trial.update(assign)
        if any(table.column(n).role == "source" for n in assign):
            trial = _rederive(trial, table)
            trial.update(assign)      # re-derivation must not undo it
        val = expr.eval_row(constraint.ast, trial)
        return trial if (val is not None and not val) else None

    refs = sorted(expr.columns(constraint.ast))
    # single-column candidates first, then pairs (bounded search)
    for name in refs:
        for candidate in _candidates(table.column(name), rng):
            found = try_assignment({name: candidate})
            if found is not None:
                return found
    for i, name_a in enumerate(refs):
        for name_b in refs[i + 1:]:
            for cand_a in _candidates(table.column(name_a), rng):
                for cand_b in _candidates(table.column(name_b), rng):
                    found = try_assignment({name_a: cand_a, name_b: cand_b})
                    if found is not None:
                        return found
    raise PlantingError(
        f"{table.name}: cannot auto-violate constraint {constraint.id!r} "
        f"(single columns and pairs searched); add a 'plant:' hint with "
        f"explicit violating values")


def _candidates(col: Column, rng: random.Random) -> list:
    if col.domain is None:
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
    is_text = _formula_is_text(table, col.formula_ast)
    guard = (f"{expr.to_sql(col.when_ast)} AND "
             if col.when_ast is not None else "")

    if is_text:
        # exact, null-safe equality for text-valued formulas
        mismatch = f"({col.name} IS NOT ({formula_sql}))"
    else:
        # null-aware numeric mismatch: a NULL where the formula yields a
        # value (or vice versa) is a defect too, not a silent skip
        null_flip = (
            f"(CASE WHEN {col.name} IS NULL THEN 1 ELSE 0 END) <> "
            f"(CASE WHEN ({formula_sql}) IS NULL THEN 1 ELSE 0 END)")
        mismatch = (f"({null_flip} OR "
                    f"COALESCE(ABS({col.name} - ({formula_sql})), 0) > {EPS})")

    def plant(twin: Twin, rng: random.Random, k: int) -> list:
        def in_scope(row: dict) -> bool:
            if col.when_ast is not None:
                cond = expr.eval_row(col.when_ast, row)
                if cond is None or not cond:
                    return False
            return True

        def mutate(row: dict) -> dict:
            correct = expr.eval_row(col.formula_ast, row)
            if is_text:
                row[col.name] = "APDQ_DRIFT"
            elif correct is None:
                row[col.name] = 999999.0     # value where NULL is documented
            else:
                row[col.name] = round(float(correct) * 1.5 + 1000.0, 6)
            return row
        return _plant_rows(twin, table, f"C06_{col.name}", k, rng, mutate,
                           predicate=in_scope)

    return GeneratedDefect(
        defect_id=f"C06:{table.name}.{col.name}",
        dq_class=6, table=table.name, columns=(col.name,),
        description=(f"{col.name} ({col.concept}) deviates from its "
                     f"documented formula: {col.formula}"
                     + (f" (where {col.when})" if col.when else "")),
        oracle_sql=(f"SELECT {table.primary_key} FROM {table.name} "
                    f"WHERE {guard}{mismatch}"),
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
            if isinstance(row[b], str):     # ISO date: strictly after any b
                row[a] = "9999-12-31"
            else:                           # yyyymm: one year past successor
                row[a] = int(row[b]) + 100
            return row
        return _plant_rows(twin, table, f"C09_{idx}", k, rng, mutate,
                           predicate=both_present)

    return GeneratedDefect(
        defect_id=f"C09:{table.name}.{'_'.join(chain)}",
        dq_class=9, table=table.name, columns=tuple(chain),
        description=f"lifecycle dates out of order: {' <= '.join(chain)}",
        oracle_sql=pair_sql, plant=plant)


# ── class 10: panel inconsistency ────────────────────────────────────────────

def _pick_series(twin: Twin, table: Table, rng: random.Random,
                 exclude: set) -> tuple[object, list[dict]]:
    sk = table.panel.series_key
    per = table.panel.period_column
    by_key: dict = {}
    for row in twin.rows[table.name]:
        by_key.setdefault(row[sk], []).append(row)
    keys = [key for key, rows in by_key.items()
            if key not in exclude and len(rows) >= 3]
    if not keys:
        raise PlantingError(f"{table.name}: no series left to mutate")
    key = rng.choice(keys)
    return key, sorted(by_key[key], key=lambda r: r[per])


def _gen_panel(table: Table, variant: str,
               cum_col: str | None = None) -> GeneratedDefect:
    sk = table.panel.series_key
    per = table.panel.period_column
    next_month = (f"CASE WHEN a.{per} % 100 = 12 "
                  f"THEN a.{per} + 89 ELSE a.{per} + 1 END")

    if variant == "gap":
        oracle = (
            f"SELECT DISTINCT a.{sk} FROM {table.name} a "
            f"WHERE a.{per} < (SELECT MAX(b.{per}) FROM {table.name} b "
            f"WHERE b.{sk} = a.{sk}) "
            f"AND NOT EXISTS (SELECT 1 FROM {table.name} c "
            f"WHERE c.{sk} = a.{sk} AND c.{per} = {next_month})")
        description = f"a month is missing from a {table.name} series"
    elif variant == "duplicate_period":
        oracle = (f"SELECT {sk} FROM {table.name} "
                  f"GROUP BY {sk}, {per} HAVING COUNT(*) > 1")
        description = f"the same month appears twice in a {table.name} series"
    else:   # decreasing cumulative
        oracle = (
            f"SELECT DISTINCT a.{sk} FROM {table.name} a "
            f"JOIN {table.name} b ON b.{sk} = a.{sk} "
            f"AND b.{per} > a.{per} AND b.{cum_col} < a.{cum_col} - {EPS}")
        description = (f"cumulative {cum_col} decreases within a "
                       f"{table.name} series")

    def plant(twin: Twin, rng: random.Random, k: int) -> list:
        planted: list = []
        used: set = set()
        for i in range(k):
            key, rows = _pick_series(twin, table, rng, used)
            used.add(key)
            middle = rows[len(rows) // 2]
            if variant == "gap":
                twin.conn.execute(
                    f"DELETE FROM {table.name} WHERE {table.primary_key} = ?",
                    (middle[table.primary_key],))
            elif variant == "duplicate_period":
                dup = dict(middle)
                dup[table.primary_key] = _fresh_pk(
                    twin, table, f"C10_DUP", len(used) * 10 + i)
                _insert_row(twin, table, dup)
            else:
                first = rows[0]
                twin.conn.execute(
                    f"UPDATE {table.name} SET {cum_col} = ? "
                    f"WHERE {table.primary_key} = ?",
                    (float(first[cum_col]) - 1000.0,
                     middle[table.primary_key]))
            planted.append(key)
        twin.conn.commit()
        return planted

    suffix = cum_col if cum_col else variant
    return GeneratedDefect(
        defect_id=f"C10:{table.name}.{suffix}",
        dq_class=10, table=table.name,
        columns=(sk, per) + ((cum_col,) if cum_col else ()),
        description=description, oracle_sql=oracle, plant=plant)


# ── entry point ──────────────────────────────────────────────────────────────

def generate_defects(manifest: Manifest) -> list[GeneratedDefect]:
    out: list[GeneratedDefect] = []
    for table in manifest.tables:
        for col in table.columns:
            if col.role == "derived":
                out.append(_gen_derivation(table, col))
            # sources — and the free branch of when-columns — carry a
            # domain and get completeness/validity oracles for it
            if col.domain is not None:
                if not col.domain.nullable:
                    out.append(_gen_missing(table, col))
                if (col.domain.values is not None or col.domain.min is not None
                        or col.domain.max is not None):
                    out.append(_gen_domain(table, col))
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
        if table.panel:
            out.append(_gen_panel(table, "gap"))
            out.append(_gen_panel(table, "duplicate_period"))
            for cum_col in table.panel.cumulative_columns:
                out.append(_gen_panel(table, "cumulative", cum_col))
    return out


def defects_by_class(defects: list[GeneratedDefect]) -> dict[int, list]:
    by: dict[int, list] = {}
    for d in defects:
        by.setdefault(d.dq_class, []).append(d)
    return by

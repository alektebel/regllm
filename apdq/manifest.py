"""Binding manifest: the single per-bank artifact.

A manifest binds a bank's physical schema to canonical concepts and
documents everything the standard needs to derive checks, mutations and
the synthetic twin:

* per column: concept, role (``source``/``derived``), domain or formula,
  reconciliation surfaces, regulation refs, optional signed waivers;
* per table: primary key, foreign keys, intra-row constraints, date
  orderings, control-total declarations.

The manifest is data, reviewable by a human who never reads generated
SQL. ``load_manifest`` validates shape, concept references, formula
parseability and dependency cycles, and raises ``ManifestError`` with an
actionable message otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import expr

CONCEPTS_DIR = Path(__file__).parent / "concepts"
DEFAULT_VOCABULARY = CONCEPTS_DIR / "pd_lgd.yaml"

DOMAIN_TYPES = ("int", "real", "text", "yyyymm")
ROLES = ("source", "derived")
FORMULA_EPS = 0.01


class ManifestError(ValueError):
    """Manifest fails validation."""


@dataclass(frozen=True)
class Domain:
    type: str
    nullable: bool = False
    unique: bool = False
    min: float | None = None
    max: float | None = None
    values: tuple | None = None    # enum for text/int


@dataclass(frozen=True)
class Reconciliation:
    surface: str                   # surface table name
    column: str                    # column on the surface holding the fact
    join_column: str               # PK column shared with the fact table


@dataclass(frozen=True)
class Column:
    name: str
    concept: str
    role: str                      # source | derived
    description: str = ""
    domain: Domain | None = None   # required for sources
    formula: str | None = None     # required for derived
    formula_ast: object = None
    reconcile: tuple[Reconciliation, ...] = ()
    regulation_refs: tuple[str, ...] = ()
    waivers: dict = field(default_factory=dict)  # obligation -> signed reason


@dataclass(frozen=True)
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass(frozen=True)
class Constraint:
    """Declared intra-row invariant: ``expr`` must hold on every row."""
    id: str
    expr: str
    ast: object
    description: str = ""
    regulation_refs: tuple[str, ...] = ()
    plant: dict = field(default_factory=dict)  # explicit violating values


@dataclass(frozen=True)
class ControlCheck:
    kind: str                      # count | sum
    column: str | None = None      # for sum


@dataclass(frozen=True)
class Control:
    surface: str                   # control-totals table name
    checks: tuple[ControlCheck, ...] = ()


@dataclass(frozen=True)
class Table:
    name: str
    primary_key: str
    rows: int = 500                # twin size hint
    columns: tuple[Column, ...] = ()
    foreign_keys: tuple[ForeignKey, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    date_orderings: tuple[tuple[str, ...], ...] = ()
    control: Control | None = None
    waivers: dict = field(default_factory=dict)

    def column(self, name: str) -> Column:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


@dataclass(frozen=True)
class Manifest:
    name: str
    vocabulary: str
    concepts: dict            # id -> concept dict
    tables: tuple[Table, ...]
    source_path: Path | None = None

    def table(self, name: str) -> Table:
        for t in self.tables:
            if t.name == name:
                return t
        raise KeyError(name)

    def bound_concepts(self) -> set[str]:
        return {c.concept for t in self.tables for c in t.columns}


# ── loading ──────────────────────────────────────────────────────────────────

def load_vocabulary(path: Path = DEFAULT_VOCABULARY) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {c["id"]: c for c in data["concepts"]}


def _domain(raw: dict, where: str) -> Domain:
    dtype = raw.get("type")
    if dtype not in DOMAIN_TYPES:
        raise ManifestError(f"{where}: domain.type must be one of {DOMAIN_TYPES}")
    values = raw.get("values")
    return Domain(
        type=dtype,
        nullable=bool(raw.get("nullable", False)),
        unique=bool(raw.get("unique", False)),
        min=raw.get("min"),
        max=raw.get("max"),
        values=tuple(values) if values else None,
    )


def _column(raw: dict, table_name: str, concepts: dict) -> Column:
    name = raw.get("name")
    where = f"{table_name}.{name}"
    if not name:
        raise ManifestError(f"table {table_name}: column without a name")
    concept = raw.get("concept")
    if not concept:
        raise ManifestError(f"{where}: missing concept binding")
    if concept not in concepts:
        raise ManifestError(f"{where}: unknown concept {concept!r} "
                            f"(not in vocabulary)")
    role = raw.get("role")
    if role not in ROLES:
        raise ManifestError(f"{where}: role must be one of {ROLES}")

    domain = None
    formula = raw.get("formula")
    formula_ast = None
    if role == "source":
        if "domain" not in raw:
            raise ManifestError(f"{where}: source column requires a domain")
        if formula:
            raise ManifestError(f"{where}: source column cannot have a formula")
        domain = _domain(raw["domain"], where)
    else:
        if not formula:
            raise ManifestError(
                f"{where}: derived column requires a formula "
                f"(undocumented derivation = finding by construction)")
        try:
            formula_ast = expr.parse(formula)
        except expr.ExprError as exc:
            raise ManifestError(f"{where}: bad formula: {exc}") from exc

    recons = []
    for r in raw.get("reconcile", ()):
        for key in ("surface", "column", "join_column"):
            if key not in r:
                raise ManifestError(f"{where}: reconcile entry missing {key!r}")
        recons.append(Reconciliation(r["surface"], r["column"], r["join_column"]))

    return Column(
        name=name, concept=concept, role=role,
        description=raw.get("description", ""),
        domain=domain, formula=formula, formula_ast=formula_ast,
        reconcile=tuple(recons),
        regulation_refs=tuple(raw.get("regulation_refs", ())),
        waivers=dict(raw.get("waivers", {})),
    )


def _table(raw: dict, concepts: dict) -> Table:
    name = raw.get("name")
    if not name:
        raise ManifestError("table without a name")
    pk = raw.get("primary_key")
    if not pk:
        raise ManifestError(f"table {name}: missing primary_key")

    cols = tuple(_column(c, name, concepts) for c in raw.get("columns", ()))
    col_names = {c.name for c in cols}
    if pk not in col_names:
        raise ManifestError(f"table {name}: primary_key {pk!r} is not a column")

    fks = []
    for fk in raw.get("foreign_keys", ()):
        for key in ("column", "ref_table", "ref_column"):
            if key not in fk:
                raise ManifestError(f"table {name}: foreign_key missing {key!r}")
        if fk["column"] not in col_names:
            raise ManifestError(
                f"table {name}: foreign_key column {fk['column']!r} unknown")
        fks.append(ForeignKey(fk["column"], fk["ref_table"], fk["ref_column"]))

    constraints = []
    for c in raw.get("constraints", ()):
        if "id" not in c or "expr" not in c:
            raise ManifestError(f"table {name}: constraint needs id and expr")
        try:
            ast = expr.parse(c["expr"])
        except expr.ExprError as exc:
            raise ManifestError(
                f"table {name}: constraint {c['id']}: bad expr: {exc}") from exc
        unknown = expr.columns(ast) - col_names
        if unknown:
            raise ManifestError(
                f"table {name}: constraint {c['id']} references unknown "
                f"columns {sorted(unknown)}")
        constraints.append(Constraint(
            id=c["id"], expr=c["expr"], ast=ast,
            description=c.get("description", ""),
            regulation_refs=tuple(c.get("regulation_refs", ())),
            plant=dict(c.get("plant", {})),
        ))

    orderings = []
    for chain in raw.get("date_orderings", ()):
        if len(chain) < 2:
            raise ManifestError(f"table {name}: date_ordering needs >=2 columns")
        for col in chain:
            if col not in col_names:
                raise ManifestError(
                    f"table {name}: date_ordering references unknown {col!r}")
        orderings.append(tuple(chain))

    control = None
    if "control" in raw:
        c = raw["control"]
        if "surface" not in c:
            raise ManifestError(f"table {name}: control needs a surface name")
        checks = []
        for chk in c.get("checks", ()):
            kind = chk.get("kind")
            if kind not in ("count", "sum"):
                raise ManifestError(
                    f"table {name}: control check kind must be count|sum")
            if kind == "sum" and chk.get("column") not in col_names:
                raise ManifestError(
                    f"table {name}: control sum column "
                    f"{chk.get('column')!r} unknown")
            checks.append(ControlCheck(kind=kind, column=chk.get("column")))
        control = Control(surface=c["surface"], checks=tuple(checks))

    return Table(
        name=name, primary_key=pk, rows=int(raw.get("rows", 500)),
        columns=cols, foreign_keys=tuple(fks),
        constraints=tuple(constraints), date_orderings=tuple(orderings),
        control=control, waivers=dict(raw.get("waivers", {})),
    )


def _check_dependencies(manifest: Manifest) -> None:
    """Derived formulas may reference only columns of the same table, and
    the intra-table derivation graph must be acyclic (twin build order)."""
    for table in manifest.tables:
        names = set(table.column_names)
        deps: dict[str, set[str]] = {}
        for col in table.columns:
            if col.role != "derived":
                continue
            refs = expr.columns(col.formula_ast)
            unknown = refs - names
            if unknown:
                raise ManifestError(
                    f"{table.name}.{col.name}: formula references unknown "
                    f"columns {sorted(unknown)} (cross-table formulas are a "
                    f"roadmap item; reconcile: declares cross-table facts)")
            deps[col.name] = {r for r in refs
                              if table.column(r).role == "derived"}
        # cycle detection over derived-only edges
        state: dict[str, int] = {}

        def visit(node: str, chain: tuple[str, ...]) -> None:
            if state.get(node) == 1:
                raise ManifestError(
                    f"{table.name}: circular derivation "
                    f"{' -> '.join(chain + (node,))}")
            if state.get(node) == 2:
                return
            state[node] = 1
            for dep in deps.get(node, ()):
                visit(dep, chain + (node,))
            state[node] = 2

        for name in deps:
            visit(name, ())


def derivation_order(table: Table) -> list[Column]:
    """Derived columns of ``table`` in evaluation order."""
    remaining = {c.name: c for c in table.columns if c.role == "derived"}
    done: set[str] = {c.name for c in table.columns if c.role == "source"}
    ordered: list[Column] = []
    while remaining:
        progressed = False
        for name in list(remaining):
            refs = expr.columns(remaining[name].formula_ast)
            if refs <= done:
                ordered.append(remaining.pop(name))
                done.add(name)
                progressed = True
        if not progressed:   # unreachable if _check_dependencies passed
            raise ManifestError(
                f"{table.name}: cannot order derivations {sorted(remaining)}")
    return ordered


def load_manifest(path: Path | str) -> Manifest:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("apdq_manifest") != 1:
        raise ManifestError(f"{path}: expected 'apdq_manifest: 1' header")
    if not data.get("name"):
        raise ManifestError(f"{path}: manifest needs a name")

    vocab_path = DEFAULT_VOCABULARY
    if data.get("concepts_file"):
        vocab_path = (path.parent / data["concepts_file"]).resolve()
    concepts = load_vocabulary(vocab_path)

    tables = tuple(_table(t, concepts) for t in data.get("tables", ()))
    if not tables:
        raise ManifestError(f"{path}: manifest declares no tables")
    table_names = {t.name for t in tables}
    for t in tables:
        for fk in t.foreign_keys:
            if fk.ref_table not in table_names:
                raise ManifestError(
                    f"table {t.name}: foreign key references undeclared "
                    f"table {fk.ref_table!r}")

    manifest = Manifest(
        name=data["name"], vocabulary=str(vocab_path.name),
        concepts=concepts, tables=tables, source_path=path,
    )
    _check_dependencies(manifest)
    return manifest

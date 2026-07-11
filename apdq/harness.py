"""Certification harness: the executable Level-B protocol.

Given a manifest (and optionally a requirement register), runs the full
certification described in ``docs/AUDITOR_PARITY_STANDARD.md`` §6:

1. **Lineage gate** — every node carries its obligations or a waiver.
2. **Register gate** — 0 unsigned rows; signed rows' concepts all bound.
3. **Specificity** — every generated oracle returns 0 rows on the clean
   twin (a firing oracle is a bug, clean-by-construction contract).
4. **Recall** — per defect, k mutations planted on a fresh twin copy;
   the defect's oracle must catch all k (aggregate classes: must fire).
5. **Mixed run** — all defects planted at once; hits attributed by
   planted PK; the confusion matrix surfaces overlap and overbroad
   oracles.

The result object is everything the audit pack renders. No LLM anywhere
in this path.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import lineage as lineage_mod
from . import register as register_mod
from .defects import CLASS_SLUGS, GeneratedDefect, generate_defects
from .manifest import Manifest
from .register import Register
from .twin import Twin, build_twin, clone_twin

TRAP_K = 3
OVERBROAD_THRESHOLD = 3     # oracle hitting > N other defects' rows
_QUERY_ROW_LIMIT = 100_000


@dataclass
class DefectResult:
    defect_id: str
    dq_class: int
    table: str
    description: str
    oracle_sql: str
    clean_rows: int = 0                  # must be 0 (specificity)
    planted: int = 0
    caught: int = 0
    aggregate: bool = False
    fired_aggregate: bool = False
    error: str = ""

    @property
    def recall(self) -> float:
        if self.aggregate:
            return 1.0 if self.fired_aggregate else 0.0
        return (self.caught / self.planted) if self.planted else 0.0

    @property
    def clean_ok(self) -> bool:
        return self.clean_rows == 0 and not self.error


@dataclass
class Certification:
    manifest_name: str
    seed: int
    k: int
    obligations: list = field(default_factory=list)
    obligation_summary: dict = field(default_factory=dict)
    register_summary: dict = field(default_factory=dict)
    unsigned_requirements: list = field(default_factory=list)
    unbound_requirements: list = field(default_factory=list)
    defect_results: list[DefectResult] = field(default_factory=list)
    overlap: dict = field(default_factory=dict)     # oracle -> other defects hit
    overbroad: list = field(default_factory=list)

    @property
    def lineage_ok(self) -> bool:
        return self.obligation_summary.get("missing", 0) == 0

    @property
    def register_ok(self) -> bool:
        return not self.unsigned_requirements and not self.unbound_requirements

    @property
    def specificity_ok(self) -> bool:
        return all(r.clean_ok for r in self.defect_results)

    @property
    def recall_ok(self) -> bool:
        return all(r.recall == 1.0 for r in self.defect_results)

    @property
    def certified(self) -> bool:
        return (self.lineage_ok and self.register_ok
                and self.specificity_ok and self.recall_ok
                and bool(self.defect_results))

    def per_class(self) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for res in self.defect_results:
            bucket = out.setdefault(
                res.dq_class,
                {"slug": CLASS_SLUGS[res.dq_class], "defects": 0,
                 "caught": 0, "planted": 0, "recall_min": 1.0})
            bucket["defects"] += 1
            bucket["caught"] += res.caught
            bucket["planted"] += res.planted
            bucket["recall_min"] = min(bucket["recall_min"], res.recall)
        return out


def _run_oracle(conn, sql: str) -> list:
    cur = conn.execute(sql)
    return cur.fetchmany(_QUERY_ROW_LIMIT)


def _first_column(rows: list) -> set:
    return {r[0] for r in rows}


def certify(manifest: Manifest, register: Register | None = None,
            seed: int = 42, k: int = TRAP_K) -> Certification:
    cert = Certification(manifest_name=manifest.name, seed=seed, k=k)

    # 1. lineage gate
    obligations = lineage_mod.check_obligations(manifest)
    cert.obligations = obligations
    cert.obligation_summary = lineage_mod.summary(obligations)

    # 2. register gate (optional artifact, mandatory once supplied)
    if register is not None:
        cert.register_summary = register_mod.summary(register)
        cert.unsigned_requirements = [
            r.id for r in register_mod.unsigned(register)]
        cert.unbound_requirements = [
            {"id": req.id, "missing_concepts": missing}
            for req, missing in register_mod.unbound(register, manifest)]

    # 3 + 4. specificity on the clean twin, recall on per-defect traps
    clean = build_twin(manifest, seed=seed)
    defects = generate_defects(manifest)
    for defect in defects:
        res = DefectResult(
            defect_id=defect.defect_id, dq_class=defect.dq_class,
            table=defect.table, description=defect.description,
            oracle_sql=defect.oracle_sql, aggregate=defect.aggregate)
        try:
            res.clean_rows = len(_run_oracle(clean.conn, defect.oracle_sql))
            trap = clone_twin(clean)
            planted = defect.plant(trap, random.Random(seed + 1), k)
            rows = _run_oracle(trap.conn, defect.oracle_sql)
            if defect.aggregate:
                res.fired_aggregate = len(rows) > 0
            else:
                res.planted = len(planted)
                res.caught = len(set(planted) & _first_column(rows))
            trap.conn.close()
        except Exception as exc:  # noqa: BLE001 — recorded, fails the cert
            res.error = f"{type(exc).__name__}: {exc}"
        cert.defect_results.append(res)

    # 5. mixed run — all point defects planted at once, attribution by PK
    mixed = clone_twin(clean)
    ownership: dict[object, str] = {}
    point_defects = [d for d in defects if not d.aggregate]
    for idx, defect in enumerate(point_defects):
        try:
            for pk in defect.plant(mixed, random.Random(seed + 2 + idx), k):
                ownership[pk] = defect.defect_id
        except Exception:  # noqa: BLE001 — already recorded above
            continue
    for defect in point_defects:
        try:
            hits = _first_column(_run_oracle(mixed.conn, defect.oracle_sql))
        except Exception:  # noqa: BLE001
            continue
        others = sorted({
            ownership[pk] for pk in hits
            if pk in ownership and ownership[pk] != defect.defect_id})
        if others:
            cert.overlap[defect.defect_id] = others
        # recomputation oracles (class 6) legitimately fire on any upstream
        # corruption — that propagation IS the completeness argument, so
        # they are exempt from the overbroad smell
        if len(others) > OVERBROAD_THRESHOLD and defect.dq_class != 6:
            cert.overbroad.append(defect.defect_id)
    mixed.conn.close()
    clean.conn.close()
    return cert

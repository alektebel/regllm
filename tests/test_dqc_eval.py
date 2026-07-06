"""Tests for the DQC eval harness (DQC/eval/) — the artifact that certifies
check quality, so it gets its own regression suite.

Uses small row counts to stay fast; the harness is deterministic by seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parent.parent / "DQC" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from defect_catalog import (  # noqa: E402
    DEFECTS, DEFECTS_BY_ID, DIMENSIONS, PK_COLUMN, TABLE_NAME,
)
from generate_db import (  # noqa: E402
    build_clean_conn, build_mixed, build_trap, _has_aux,
)
from eval_harness import (  # noqa: E402
    coverage, score_check, selftest, _safe_run,
)
from coverage_matrix import (  # noqa: E402
    build_matrix, canon_ref, parse_dictionary, split_refs,
)

ROWS = 400
SEED = 7


# ── catalog sanity ──────────────────────────────────────────────────────────

def test_catalog_shape():
    ids = [d.defect_id for d in DEFECTS]
    assert len(ids) == len(set(ids)), "duplicate defect ids"
    for d in DEFECTS:
        assert d.dimension in DIMENSIONS, d.defect_id
        assert d.oracle_sql.upper().startswith("SELECT"), d.defect_id
        assert d.columns, d.defect_id
    # every dimension is exercised by at least one non-decoy defect
    covered_dims = {d.dimension for d in DEFECTS if not d.decoy}
    assert covered_dims == set(DIMENSIONS)


def test_every_defect_has_regulatory_anchor_or_is_decoy():
    missing = [d.defect_id for d in DEFECTS
               if not d.decoy and not d.regulation_ref
               and d.defect_id not in ("D07", "D08", "D09", "D11", "D24")]
    assert not missing, f"defects without regulation_ref: {missing}"


# ── selftest: 0 on clean, all k planted rows on the trap ────────────────────

def test_selftest_all_oracles_pass():
    rep = selftest(rows=ROWS, seed=SEED, k=2)
    assert rep["failed"] == 0, rep["failures"]
    assert rep["passed"] == len(DEFECTS)


def test_trap_plants_k_distinct_rows():
    d = DEFECTS_BY_ID["D01"]
    trap, planted = build_trap(d, ROWS, SEED, k=3)
    assert len(planted) == len(set(planted)) == 3
    n = trap.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    assert n == ROWS + 3


def test_duplicate_pk_defect_is_a_real_duplicate():
    d = DEFECTS_BY_ID["D33"]
    assert d.plants_duplicate_pk
    trap, planted = build_trap(d, ROWS, SEED, k=2)
    for pk in planted:
        n = trap.execute(
            f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE {PK_COLUMN}=?", (pk,)
        ).fetchone()[0]
        assert n == 2, "planted PK is not duplicated"
    _, rows = _safe_run(trap, d.oracle_sql)
    assert {r[PK_COLUMN] for r in rows} == set(planted)


# ── scoring ─────────────────────────────────────────────────────────────────

def test_score_check_fractional_catches():
    d = DEFECTS_BY_ID["D01"]
    clean = build_clean_conn(ROWS, SEED)
    trap, planted = build_trap(d, ROWS, SEED, k=3)
    # a check that names exactly one planted PK catches 1/3
    sql = (f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE {PK_COLUMN} = '{planted[0]}'")
    s = score_check(sql, d, clean, trap, planted_pks=planted)
    assert s.clean
    assert abs(s.components["r_catches"] - 1 / 3) < 1e-3
    # the oracle itself catches all three
    s2 = score_check(d.oracle_sql, d, clean, trap, planted_pks=planted)
    assert s2.components["r_catches"] == 1.0
    assert s2.clean and s2.catches


def test_score_check_rejects_non_select():
    d = DEFECTS_BY_ID["D01"]
    clean = build_clean_conn(ROWS, SEED)
    trap, planted = build_trap(d, ROWS, SEED)
    s = score_check(f"DELETE FROM {TABLE_NAME}", d, clean, trap, planted)
    assert s.error and s.total == 0.0


# ── coverage on the mixed DB ────────────────────────────────────────────────

def test_coverage_attributes_hits_by_planted_pk():
    oracle_sqls = [DEFECTS_BY_ID[i].oracle_sql for i in ("D01", "D19", "D33")]
    rep = coverage(oracle_sqls, rows=ROWS, seed=SEED, k=2)
    assert rep.n_valid == 3
    assert {"D01", "D19", "D33"} <= set(rep.caught_defects)
    # per-article recall surfaces the defects' citations
    assert any(st["caught"] for st in rep.per_article.values())


def test_coverage_flags_overbroad_pk_fishing():
    # A check that fishes for the trap PKs directly (reward hacking) fires on
    # every planted defect and must be flagged overbroad.
    sql = (f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE {PK_COLUMN} LIKE '~~DIRTY~%' ESCAPE '~'")
    # __DIRTY uses underscores (SQL wildcards) — escape them
    sql = (f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE {PK_COLUMN} LIKE '\\_\\_DIRTY\\_%' ESCAPE '\\'")
    rep = coverage([sql], rows=ROWS, seed=SEED, k=1)
    assert rep.overbroad_checks == 1
    assert rep.per_check[0]["overbroad"] is True


def test_coverage_requires_clean_zero():
    # fires on every row → invalid, no hits
    rep = coverage([f"SELECT {PK_COLUMN} FROM {TABLE_NAME}"],
                   rows=ROWS, seed=SEED, k=1)
    assert rep.n_valid == 0
    assert rep.recall == 0.0


def test_clean_db_has_source_tables_and_no_cross_table_violations():
    conn = build_clean_conn(ROWS, SEED)
    assert _has_aux(conn)
    for tbl in ("contratos", "basilea_mensual", "colaterales"):
        n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        assert n > 0, tbl
    # every cross-table oracle is clean on the clean DB
    for d in DEFECTS:
        if d.cross_table:
            _, rows = _safe_run(conn, d.oracle_sql)
            assert rows == [], f"{d.defect_id} fires on clean DB"


def test_cross_table_defect_isolated_in_mixed_db():
    # A cross-table oracle must catch its own planted rows and not be dominated
    # by other defects (the collision bug we fixed). Verify D50 (segment
    # reconciliation) catches exactly its own planted PKs on the mixed DB.
    d = DEFECTS_BY_ID["D50"]
    mixed, planted = build_mixed(DEFECTS, ROWS, SEED, k=1)
    pk_to_defect = {pk: did for did, pks in planted.items() for pk in pks}
    _, rows = _safe_run(mixed, d.oracle_sql)
    caught = {r[PK_COLUMN] for r in rows}
    attributed = {pk_to_defect.get(pk) for pk in caught}
    # every planted D50 row caught, and nothing wildly over-broad
    assert set(planted["D50"]) <= caught
    assert len(attributed - {"D50", None}) <= 2


def test_referential_integrity_orphan_only_caught_by_d48():
    trap, planted = build_trap(DEFECTS_BY_ID["D48"], ROWS, SEED, k=2)
    _, rows = _safe_run(trap, DEFECTS_BY_ID["D48"].oracle_sql)
    assert set(planted) <= {r[PK_COLUMN] for r in rows}
    # a reconciliation oracle (inner join) must NOT catch the orphan
    _, recon = _safe_run(trap, DEFECTS_BY_ID["D49"].oracle_sql)
    assert not (set(planted) & {r[PK_COLUMN] for r in recon})


def test_monthly_panel_present_and_clean():
    conn = build_clean_conn(ROWS, SEED)
    n = conn.execute("SELECT COUNT(*) FROM evolucion_mensual").fetchone()[0]
    assert n > ROWS, "panel should have several months per cycle"
    # every panel (temporal) oracle is clean on the clean DB
    for d in DEFECTS:
        if d.target_table == "evolucion_mensual":
            _, rows = _safe_run(conn, d.oracle_sql)
            assert rows == [], f"{d.defect_id} fires on clean panel"


def test_panel_defect_series_planted_and_caught():
    for did in ("D58", "D59", "D60", "D61", "D62", "D63"):
        d = DEFECTS_BY_ID[did]
        assert d.target_table == "evolucion_mensual"
        trap, planted = build_trap(d, ROWS, SEED, k=2)
        # the dirty series lands in the panel, not the reporting table
        n = trap.execute(
            "SELECT COUNT(DISTINCT ID_CONTR_CICLO_LGD) FROM evolucion_mensual "
            "WHERE ID_CONTR_CICLO_LGD LIKE '\\_\\_DIRTY%' ESCAPE '\\'"
        ).fetchone()[0]
        assert n == 2, did
        _, rows = _safe_run(trap, d.oracle_sql)
        assert set(planted) <= {r[PK_COLUMN] for r in rows}, did


def test_weird_cross_domain_defects_catch_and_clean():
    conn = build_clean_conn(ROWS, SEED)
    for did in ("D64", "D65", "D66"):
        d = DEFECTS_BY_ID[did]
        assert len(_safe_run(conn, d.oracle_sql)[1]) == 0, f"{did} not clean"
        trap, planted = build_trap(d, ROWS, SEED, k=2)
        _, rows = _safe_run(trap, d.oracle_sql)
        assert set(planted) <= {r[PK_COLUMN] for r in rows}, did


def test_date_interrelation_defect_catches():
    for did in ("D54", "D55", "D56", "D57"):
        d = DEFECTS_BY_ID[did]
        trap, planted = build_trap(d, ROWS, SEED, k=2)
        _, rows = _safe_run(trap, d.oracle_sql)
        assert set(planted) <= {r[PK_COLUMN] for r in rows}, did


def test_mixed_db_contains_all_defects():
    mixed, planted = build_mixed(DEFECTS, ROWS, SEED, k=1)
    assert set(planted) == {d.defect_id for d in DEFECTS}
    assert all(len(v) == 1 for v in planted.values())


# ── coverage matrix ─────────────────────────────────────────────────────────

def test_canon_ref_equivalences():
    assert canon_ref("CRR Art. 181.1(b)") == canon_ref("CRR Art. 181.1.b")
    assert canon_ref("EBA GL 2017/16 §73") == canon_ref("EBA GL 2017/16 §73 ")
    assert split_refs("IFRS 9 / CRR Art. 158") == ["IFRS 9", "CRR Art. 158"]
    assert split_refs("—") == []


def test_dictionary_parses_all_schema_columns():
    from generate_db import COLUMNS
    fields = parse_dictionary()
    missing = [c for c in COLUMNS if c not in fields]
    assert not missing, f"schema columns missing from data_dictionary.md: {missing}"


def test_matrix_has_no_todo_cells():
    m = build_matrix()
    assert m["todo"] == 0, (
        f"uncovered (field × article) cells: "
        f"{[c for c in m['cells'] if c['status'] == 'todo']}"
    )
    assert m["coverage"] == 1.0
    assert not m["fields_uncovered"]


def test_example_checks_score_end_to_end():
    checks_file = EVAL_DIR / "example_checks.sql"
    from eval_harness import _parse_sql_file
    checks = _parse_sql_file(checks_file)
    assert checks
    rep = coverage(checks, rows=ROWS, seed=SEED, k=1)
    assert rep.recall > 0

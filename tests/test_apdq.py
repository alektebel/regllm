"""Regression tests for the APDQ reference implementation (the certifier
is code too — same philosophy as tests/test_dqc_eval.py)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from apdq import expr
from apdq.defects import generate_defects
from apdq.harness import certify
from apdq.lineage import check_obligations, gaps
from apdq.manifest import ManifestError, load_manifest, load_vocabulary
from apdq.register import (RegisterError, load_register, stale, text_hash,
                           unbound, unsigned)
from apdq.twin import build_twin

EXAMPLES = Path(__file__).parent.parent / "apdq" / "examples"
MINI = EXAMPLES / "mini_ciclos" / "manifest.yaml"
RETAIL = EXAMPLES / "retail_mortgages" / "manifest.yaml"
REGISTER = EXAMPLES / "mini_ciclos" / "register.yaml"
FULL = EXAMPLES / "ciclos_full" / "manifest.yaml"
CROSSWALK = EXAMPLES / "ciclos_full" / "crosswalk.yaml"


# ── expression language ──────────────────────────────────────────────────────

@pytest.mark.parametrize("formula,row,expected", [
    ("A + B * 2", {"A": 1, "B": 3}, 7),
    ("max(A, B)", {"A": 0.02, "B": 0.05}, 0.05),
    ("if(S = 'X', A, B)", {"S": "X", "A": 1, "B": 2}, 1),
    ("if(S = 'X', A, B)", {"S": "Y", "A": 1, "B": 2}, 2),
    ("A / B", {"A": 10, "B": 4}, 2.5),
    ("A > 1 and B < 5", {"A": 2, "B": 3}, True),
    ("not (A = 1)", {"A": 1}, False),
    ("A + B", {"A": None, "B": 1}, None),           # NULL propagation
    ("min(A, B, 0.5)", {"A": 0.9, "B": 0.7}, 0.5),
    ("abs(A - B)", {"A": 1, "B": 4}, 3),
    ("sqrt(A)", {"A": 9}, 3.0),
])
def test_expr_eval(formula, row, expected):
    assert expr.eval_row(expr.parse(formula), row) == expected


@pytest.mark.parametrize("formula,row", [
    ("A + B * 2 - C / 4", {"A": 1.5, "B": 3.0, "C": 8.0}),
    ("max(A, B) * min(A, B)", {"A": 0.25, "B": 0.5}),
    ("if(A > B, A - B, B - A)", {"A": 2.0, "B": 5.0}),
    ("sqrt(A) + abs(B)", {"A": 16.0, "B": -3.0}),
])
def test_expr_sql_matches_python(formula, row):
    """Compiled SQL and the Python evaluator must agree — the twin is
    built with one and the oracle with the other."""
    ast = expr.parse(formula)
    conn = sqlite3.connect(":memory:")
    expr.register_sql_functions(conn)
    cols = ", ".join(f"{k}" for k in row)
    conn.execute(f"CREATE TABLE t ({cols})")
    conn.execute(f"INSERT INTO t VALUES ({', '.join('?' for _ in row)})",
                 tuple(row.values()))
    (sql_val,) = conn.execute(f"SELECT {expr.to_sql(ast)} FROM t").fetchone()
    assert sql_val == pytest.approx(expr.eval_row(ast, row))


@pytest.mark.parametrize("bad", [
    "", "A +", "unknownfn(A)", "A ++ B", "if(A, B)", "(A", "1 2",
])
def test_expr_rejects_garbage(bad):
    with pytest.raises(expr.ExprError):
        expr.parse(bad)


# ── manifest validation ──────────────────────────────────────────────────────

def test_examples_load():
    for path in (MINI, RETAIL):
        manifest = load_manifest(path)
        assert manifest.tables


def _write(tmp_path, text: str) -> Path:
    p = tmp_path / "m.yaml"
    p.write_text(text, encoding="utf-8")
    return p


BASE = """apdq_manifest: 1
name: t
tables:
  - name: x
    primary_key: ID
    columns:
      - {name: ID, concept: contract_id, role: source, domain: {type: text, unique: true}}
"""


def test_manifest_rejects_derived_without_formula(tmp_path):
    text = BASE + \
        "      - {name: Y, concept: expected_credit_loss, role: derived}\n"
    with pytest.raises(ManifestError, match="requires a formula"):
        load_manifest(_write(tmp_path, text))


def test_manifest_rejects_unknown_concept(tmp_path):
    text = BASE.replace("contract_id", "no_such_concept")
    with pytest.raises(ManifestError, match="unknown concept"):
        load_manifest(_write(tmp_path, text))


def test_manifest_rejects_circular_derivation(tmp_path):
    text = BASE + (
        "      - {name: A, concept: pd_final, role: derived, formula: 'B + 1'}\n"
        "      - {name: B, concept: lgd_final, role: derived, formula: 'A + 1'}\n")
    with pytest.raises(ManifestError, match="circular"):
        load_manifest(_write(tmp_path, text))


def test_manifest_rejects_source_without_domain(tmp_path):
    text = BASE + "      - {name: Z, concept: pd_final, role: source}\n"
    with pytest.raises(ManifestError, match="requires a domain"):
        load_manifest(_write(tmp_path, text))


# ── twin: clean by construction ──────────────────────────────────────────────

def test_twin_satisfies_all_oracles():
    manifest = load_manifest(MINI)
    twin = build_twin(manifest, seed=7)
    for defect in generate_defects(manifest):
        rows = twin.conn.execute(defect.oracle_sql).fetchall()
        assert rows == [], f"{defect.defect_id} fired on the clean twin"


def test_twin_deterministic():
    manifest = load_manifest(RETAIL)
    a = build_twin(manifest, seed=3)
    b = build_twin(manifest, seed=3)
    assert a.rows == b.rows


# ── lineage obligations ──────────────────────────────────────────────────────

def test_lineage_examples_have_no_gaps():
    for path in (MINI, RETAIL):
        assert gaps(check_obligations(load_manifest(path))) == []


def test_lineage_detects_missing_reconciliation(tmp_path):
    # a source column with neither reconcile nor waiver = missing
    text = BASE + (
        "      - {name: V, concept: exposure_at_default, role: source, "
        "domain: {type: real, min: 0, max: 10}}\n")
    obligations = check_obligations(load_manifest(_write(tmp_path, text)))
    missing = gaps(obligations)
    assert any(o.node == "x.V" and o.kind == "reconcile" for o in missing)
    assert any(o.node == "x" and o.kind == "control" for o in missing)


# ── requirement register ─────────────────────────────────────────────────────

def test_register_loads_and_is_fully_signed():
    vocab = load_vocabulary()
    register = load_register(REGISTER, vocab)
    assert len(register.requirements) >= 8
    assert unsigned(register) == []


def test_register_stale_detection():
    register = load_register(REGISTER)
    req = register.requirements[0]
    assert stale(register, {req.paragraph: "texto enmendado por circular"})
    # unchanged text (normalization-insensitive) is not stale
    assert text_hash("  a  b ") == text_hash("A B")


def test_register_unbound_concepts(tmp_path):
    register = load_register(REGISTER)
    # a manifest binding almost nothing leaves signed requirements unbound
    manifest = load_manifest(_write(tmp_path, BASE))
    missing = unbound(register, manifest)
    assert missing, "expected unbound concepts against a near-empty manifest"


def test_register_rejects_signed_without_signer(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(
        "apdq_register: 1\nrequirements:\n"
        "  - {id: R1, obligation: x, text_sha256: abc, status: signed}\n",
        encoding="utf-8")
    with pytest.raises(RegisterError, match="signed_by"):
        load_register(p)


# ── end-to-end certification ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mini_cert():
    manifest = load_manifest(MINI)
    register = load_register(REGISTER, manifest.concepts)
    return certify(manifest, register, seed=42, k=3)


def test_mini_ciclos_certifies(mini_cert):
    assert mini_cert.lineage_ok
    assert mini_cert.register_ok
    assert mini_cert.specificity_ok
    assert mini_cert.recall_ok
    assert mini_cert.certified


def test_mini_ciclos_covers_all_generic_classes(mini_cert):
    assert set(mini_cert.per_class()) == {1, 2, 3, 4, 5, 6, 7, 8, 9}


def test_second_schema_certifies_without_code_edits():
    """THE MVP gate (docs/MVP_ROADMAP.md item 3): a different schema is
    certified end-to-end from its manifest alone."""
    cert = certify(load_manifest(RETAIL), seed=42, k=3)
    assert cert.certified
    assert set(cert.per_class()) == {1, 2, 3, 4, 5, 6, 7, 8, 9}


def test_expr_null_functions():
    assert expr.eval_row(expr.parse("isnull(A)"), {"A": None}) is True
    assert expr.eval_row(expr.parse("isnull(A)"), {"A": 3}) is False
    assert expr.eval_row(expr.parse("null()"), {}) is None
    assert expr.eval_row(expr.parse("yyyymm(D)"), {"D": "2024-06-15"}) == 202406
    assert expr.eval_row(
        expr.parse("if(isnull(D), null(), yyyymm(D))"), {"D": None}) is None


def test_manifest_rejects_nonmonotone_ordering_domains(tmp_path):
    text = BASE + (
        "      - {name: D1, concept: default_date, role: source, "
        "domain: {type: date, min: '2024-01-01', max: '2024-12-01'}}\n"
        "      - {name: D2, concept: cycle_close_date, role: source, "
        "domain: {type: date, min: '2020-01-01', max: '2020-06-01'}}\n"
        "    date_orderings:\n"
        "      - [D1, D2]\n")
    with pytest.raises(ManifestError, match="monotone"):
        load_manifest(_write(tmp_path, text))


def test_manifest_rejects_non_pk_reconcile_join(tmp_path):
    text = BASE + (
        "      - {name: V, concept: exposure_at_default, role: source, "
        "domain: {type: real, min: 0, max: 10}, "
        "reconcile: [{surface: s, column: V_SRC, join_column: V}]}\n")
    with pytest.raises(ManifestError, match="primary key"):
        load_manifest(_write(tmp_path, text))


# ── full-width schema: the catalog-parity gate ───────────────────────────────

@pytest.fixture(scope="module")
def full_manifest():
    return load_manifest(FULL)


@pytest.fixture(scope="module")
def full_cert(full_manifest):
    register = load_register(REGISTER, full_manifest.concepts)
    return certify(full_manifest, register, seed=42, k=3)


def test_ciclos_full_certifies(full_cert):
    assert full_cert.certified, [
        (r.defect_id, r.clean_rows, r.caught, r.planted, r.error)
        for r in full_cert.defect_results
        if not r.clean_ok or r.recall < 1.0]


def test_ciclos_full_covers_all_ten_generic_classes(full_cert):
    assert set(full_cert.per_class()) == set(range(1, 11))


def test_crosswalk_is_total(full_manifest):
    """Every hand-written catalog defect maps to a generated defect or
    carries an explicit documented exception — the gap-plan item 2 gate."""
    import yaml
    sys_path = str(Path(__file__).parent.parent / "DQC" / "eval")
    import sys as _sys
    if sys_path not in _sys.path:
        _sys.path.insert(0, sys_path)
    from defect_catalog import DEFECTS

    crosswalk = yaml.safe_load(CROSSWALK.read_text(encoding="utf-8"))["crosswalk"]
    generated_ids = {d.defect_id for d in generate_defects(full_manifest)}

    missing = [d.defect_id for d in DEFECTS if d.defect_id not in crosswalk]
    assert not missing, f"catalog defects absent from crosswalk: {missing}"

    for did, entry in crosswalk.items():
        status = entry["status"]
        assert status in ("mapped", "partial", "excluded"), (did, status)
        if status in ("mapped", "partial"):
            assert entry["apdq"] in generated_ids, (
                f"{did} maps to {entry['apdq']}, which the manifest does "
                f"not generate")
        if status in ("partial", "excluded"):
            assert entry.get("note"), f"{did}: {status} requires a note"


def test_panel_class_present(full_manifest):
    from apdq.defects import defects_by_class
    by_class = defects_by_class(generate_defects(full_manifest))
    slugs = {d.defect_id for d in by_class[10]}
    assert {"C10:ciclos_panel.gap", "C10:ciclos_panel.duplicate_period",
            "C10:ciclos_panel.RECUPERACION_ACUM",
            "C10:ciclos_panel.DPD_MES"} <= slugs


def test_certification_fails_on_broken_oracle(monkeypatch, mini_cert):
    # sanity: the verdict is not vacuously true — flipping any recall
    # or specificity bit must fail certification
    import copy
    cert = copy.deepcopy(mini_cert)
    cert.defect_results[0].clean_rows = 1
    assert not cert.certified
    cert = copy.deepcopy(mini_cert)
    point = next(r for r in cert.defect_results if not r.aggregate)
    point.caught -= 1
    assert not cert.certified

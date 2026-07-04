"""Tests for the golden-trace eval harness (DQC/eval/golden_eval.py).

The gold answers must score perfectly, and — just as important — the harness
must FAIL candidates that hallucinate articles, dodge with single-column
checks, or answer the adversarial traces wrongly.
"""

from __future__ import annotations

import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "DQC" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from defect_catalog import DIMENSIONS  # noqa: E402
from golden_eval import (  # noqa: E402
    GoldenEvaluator, gold_candidates, load_traces,
)

ROWS = 300


def _ev() -> GoldenEvaluator:
    return GoldenEvaluator(rows=ROWS, seed=11)


def _trace(tid: str) -> dict:
    return next(t for t in load_traces() if t["trace_id"] == tid)


def test_traces_cover_all_eight_dimensions():
    dims = {t["dimension"] for t in load_traces()}
    assert dims == set(DIMENSIONS)


def test_gold_answers_score_perfect():
    rep = _ev().evaluate(gold_candidates())
    assert rep["pass_rate"] == 1.0, [
        t for t in rep["traces"] if not t["passed"]]
    for dim in DIMENSIONS:
        assert rep["rubric"][dim]["pass_rate"] == 1.0


def test_empty_candidates_fail_but_adversarial_silence_passes():
    rep = _ev().evaluate({})
    by_id = {t["trace_id"]: t for t in rep["traces"]}
    # normal traces fail with nothing produced
    assert not by_id["G01"]["passed"]
    # the nonexistent-field trace expects silence — empty IS correct
    assert by_id["G15"]["passed"]


def test_hallucinated_article_is_rejected():
    ev = _ev()
    trace = _trace("G15")
    dqcs = [{
        "regla_sql": "SELECT ID_CONTR_CICLO_LGD FROM ciclos_calibrados "
                     "WHERE COLOR_FAVORITO IS NULL AND SEGMENTO = 'CORP'",
        "referencia_regulatoria": "Circular 4/2022 Art. 99",
    }]
    s = ev.score_trace(trace, dqcs)
    assert s.hallucinated_refs
    assert not s.passed


def test_single_column_decoy_answer_fails_coherence():
    ev = _ev()
    trace = _trace("G16")  # decoy bait: "EAD_TOTAL no sea cero"
    naive = [{
        "regla_sql": "SELECT ID_CONTR_CICLO_LGD FROM ciclos_calibrados "
                     "WHERE EAD_TOTAL <= 0",
        "referencia_regulatoria": "CRR Art. 166",
    }]
    s = ev.score_trace(trace, naive)
    # it catches the trap but flunks the coherence requirement
    assert s.caught == ["DA"]
    assert s.components["c_coherence"] == 0.0
    assert not s.passed


def test_check_firing_on_clean_db_is_invalid():
    ev = _ev()
    trace = _trace("G01")
    noisy = [{
        "regla_sql": "SELECT ID_CONTR_CICLO_LGD FROM ciclos_calibrados "
                     "WHERE ECL >= 0 AND EAD_TOTAL >= 0",   # fires everywhere
        "referencia_regulatoria": "IFRS 9 / CRR Art. 158",
    }]
    s = ev.score_trace(trace, noisy)
    assert s.components["c_valid"] == 0.0
    assert s.components["c_catches"] == 0.0
    assert not s.passed


def test_wrong_citation_fails_grounding():
    ev = _ev()
    trace = _trace("G02")  # requires CRR Art. 153
    gold_sql = _trace("G02")["gold"][0]["regla_sql"]
    mislabeled = [{"regla_sql": gold_sql,
                   "referencia_regulatoria": "IFRS 9"}]  # known ref, wrong one
    s = ev.score_trace(trace, mislabeled)
    assert s.components["c_catches"] == 1.0      # SQL is right
    assert s.components["c_grounding"] == 0.0    # citation is not
    assert not s.passed

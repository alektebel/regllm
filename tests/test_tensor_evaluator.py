"""Tests for the differentiable tensor-based AST evaluator.

Verifies that:
- The tensor evaluator produces values numerically equal (within 1e-6) to
  the eager Python evaluator for arithmetic-only SAS pipelines.
- ``TensorEvaluator.make_path_env`` correctly interpolates numeric inputs.
- Branch logging records IF/SELECT/WHERE decisions.
- ``excluded`` is set when a WHERE filter rejects the row.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from src.sas_diff.tensor_evaluator import (
    BranchEvent,
    TensorEvaluator,
    evaluate_tensor,
)
from src.sas_logic_tree import SASLogicTree


SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


# ── Equivalence on simple pipelines ──────────────────────────────────────────


def test_assignment_matches_python_eval():
    sas = "DATA out; SET in; ECL = PD * LGD * EAD; RUN;"
    tree = SASLogicTree()
    nodes = tree.parse(sas)
    row = {"PD": 0.05, "LGD": 0.4, "EAD": 100_000.0}

    py = tree.evaluate(nodes, row).final["ECL"]
    tens = evaluate_tensor(nodes, row).scalar("ECL")
    assert tens == pytest.approx(py, rel=1e-9)


def test_max_min_match_python_eval():
    sas = "DATA out; SET in; LGD_FLOORED = MAX(LGD, 0.30); RUN;"
    tree = SASLogicTree()
    nodes = tree.parse(sas)
    for lgd in (0.10, 0.30, 0.50):
        py = tree.evaluate(nodes, {"LGD": lgd}).final["LGD_FLOORED"]
        tens = evaluate_tensor(nodes, {"LGD": lgd}).scalar("LGD_FLOORED")
        assert tens == pytest.approx(py, rel=1e-9)


def test_if_branch_matches_python_eval():
    sas = """
    DATA out;
        SET in;
        STAGE = 1;
        IF DPDS >= 30 THEN STAGE = 2;
        IF DPDS >= 90 THEN STAGE = 3;
    RUN;
    """
    tree = SASLogicTree()
    nodes = tree.parse(sas)
    for dpds in (5, 30, 60, 90, 200):
        py = tree.evaluate(nodes, {"DPDS": dpds}).final["STAGE"]
        tens = evaluate_tensor(nodes, {"DPDS": dpds}).scalar("STAGE")
        assert tens == pytest.approx(py, rel=1e-9)


def test_where_filter_excludes_row():
    sas = """
    DATA out;
        SET in;
        WHERE PROVISION_PERIOD_MONTHS >= 9;
        Y = 1;
    RUN;
    """
    tree = SASLogicTree()
    nodes = tree.parse(sas)

    excluded_row = {"PROVISION_PERIOD_MONTHS": 6}
    trace = evaluate_tensor(nodes, excluded_row)
    assert trace.excluded is True

    kept_row = {"PROVISION_PERIOD_MONTHS": 12}
    trace = evaluate_tensor(nodes, kept_row)
    assert trace.excluded is False


def test_branch_log_records_decisions():
    sas = "DATA out; SET in; IF X > 0 THEN Y = 1; ELSE Y = 0; RUN;"
    tree = SASLogicTree()
    nodes = tree.parse(sas)
    trace = evaluate_tensor(nodes, {"X": 5})
    if_events = [e for e in trace.branch_log if e.kind == "if"]
    assert len(if_events) == 1
    assert if_events[0].taken is True
    assert "X" in if_events[0].vars_in_condition


# ── Path-env interpolation ───────────────────────────────────────────────────


def test_make_path_env_interpolates_numerics():
    v2 = {"A": 0.0, "B": 100.0, "S": "a"}
    v3 = {"A": 1.0, "B": 200.0, "S": "b"}
    env, knobs = TensorEvaluator.make_path_env(v2, v3, t=0.5)
    assert isinstance(env["A"], torch.Tensor)
    assert isinstance(env["B"], torch.Tensor)
    assert env["A"].item() == pytest.approx(0.5)
    assert env["B"].item() == pytest.approx(150.0)
    assert env["S"] == "b"
    assert env["A"].requires_grad and env["B"].requires_grad
    assert set(knobs.keys()) == {"A", "B"}


def test_autograd_propagates_through_pipeline():
    sas = "DATA out; SET in; ECL = PD * LGD * EAD; RUN;"
    nodes = SASLogicTree().parse(sas)
    env = TensorEvaluator.make_env({"PD": 0.05, "LGD": 0.4, "EAD": 100_000.0}, requires_grad=True)
    trace = TensorEvaluator().run(nodes, env)
    y = trace.value("ECL")
    assert isinstance(y, torch.Tensor)
    grads = torch.autograd.grad(y, [env["PD"], env["LGD"], env["EAD"]])
    # ∂(PD*LGD*EAD)/∂PD = LGD*EAD = 0.4*100_000 = 40_000
    assert float(grads[0]) == pytest.approx(40_000.0, rel=1e-9)
    assert float(grads[1]) == pytest.approx(0.05 * 100_000, rel=1e-9)
    assert float(grads[2]) == pytest.approx(0.05 * 0.4, rel=1e-9)


# ── Equivalence on the bundled sample pipeline ───────────────────────────────


def _load_cycles():
    p = SAMPLES / "cycles_sample.csv"
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        out = []
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if v == "":
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            out.append(row)
        return out


def test_tensor_eval_matches_python_eval_on_sample_pipeline():
    sas_path = SAMPLES / "sample_lgd.sas"
    if not sas_path.exists():
        pytest.skip("sample_lgd.sas not found")
    cycles = _load_cycles()
    if not cycles:
        pytest.skip("cycles_sample.csv not found")

    nodes = SASLogicTree().parse(sas_path.read_text(encoding="utf-8"))
    tree = SASLogicTree()

    # Sample 50 random rows
    for row in cycles[::10][:50]:
        py_trace = tree.evaluate(nodes, row)
        ts_trace = evaluate_tensor(nodes, row)
        py_ecl = py_trace.final.get("ECL")
        ts_ecl = ts_trace.scalar("ECL")
        if isinstance(py_ecl, (int, float)) and ts_ecl is not None:
            assert ts_ecl == pytest.approx(py_ecl, rel=1e-6)

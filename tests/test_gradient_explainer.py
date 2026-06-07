"""Tests for the path-integrated gradient explainer.

Verifies the *efficiency* axiom (Σφ_i ≈ ΔY) on linear and non-linear cases,
plus correct sign and ranking on hand-crafted pipelines.
"""

from __future__ import annotations

import pytest

from src.sas_diff import explain_field_diff
from src.sas_diff.gradient_explainer import path_integrated_gradients
from src.sas_logic_tree import SASLogicTree


# ── Linear pipeline ──────────────────────────────────────────────────────────


def _parse(sas: str):
    return SASLogicTree().parse(sas)


def test_efficiency_linear_pipeline():
    """For a linear pipeline ECL = PD * LGD * EAD, gradient + Shapley should
    sum exactly to ΔECL (modulo float rounding)."""
    nodes = _parse("DATA out; SET in; ECL = PD * LGD * EAD; RUN;")
    v2 = {"PD": 0.05, "LGD": 0.40, "EAD": 100_000.0}
    v3 = {"PD": 0.06, "LGD": 0.45, "EAD": 110_000.0}
    rep = path_integrated_gradients(nodes, v2, v3, target="ECL", steps=64)
    assert rep.delta_y is not None
    total = sum(a.contribution for a in rep.attributions)
    assert total == pytest.approx(rep.delta_y, abs=1e-6)
    assert abs(rep.unexplained_residual) < 1e-6


def test_efficiency_nonlinear_pipeline():
    """For a non-linear pipeline (log + max), Aumann-Shapley path integrals
    should still satisfy the efficiency axiom."""
    nodes = _parse("DATA out; SET in; Y = LOG(MAX(X, 1)) * Z; RUN;")
    v2 = {"X": 2.0, "Z": 3.0}
    v3 = {"X": 5.0, "Z": 4.5}
    rep = path_integrated_gradients(nodes, v2, v3, target="Y", steps=128)
    assert rep.delta_y is not None
    total = sum(a.contribution for a in rep.attributions)
    assert total == pytest.approx(rep.delta_y, abs=1e-3)


def test_attribution_sign_and_ranking():
    """When only PD changes, all attribution must go to PD."""
    nodes = _parse("DATA out; SET in; ECL = PD * LGD * EAD; RUN;")
    v2 = {"PD": 0.05, "LGD": 0.40, "EAD": 100_000.0}
    v3 = {"PD": 0.06, "LGD": 0.40, "EAD": 100_000.0}
    rep = path_integrated_gradients(nodes, v2, v3, target="ECL", steps=32)
    # PD must dominate; others either zero or absent
    pd_attr = next(a for a in rep.attributions if a.field == "PD")
    assert pd_attr.contribution > 0
    for a in rep.attributions:
        if a.field != "PD":
            assert abs(a.contribution) < 1e-9


def test_branch_flip_detection():
    """Predicate flips between V2 and V3 should be reported."""
    sas = """
    DATA out;
        SET in;
        IF X >= 100 THEN Y = 1; ELSE Y = 0;
    RUN;
    """
    nodes = _parse(sas)
    v2 = {"X": 50}     # condition false
    v3 = {"X": 200}    # condition true
    rep = path_integrated_gradients(nodes, v2, v3, target="Y", steps=8)
    flips = rep.branch_flips
    assert len(flips) >= 1
    assert any("X" in f.vars_in_condition for f in flips)


def test_explain_field_diff_orchestrator():
    """High-level orchestrator returns both gradient and Shapley reports."""
    sas = "DATA out; SET in; ECL = PD * LGD * EAD; RUN;"
    v2 = {"PD": 0.05, "LGD": 0.40, "EAD": 100_000.0}
    v3 = {"PD": 0.06, "LGD": 0.40, "EAD": 110_000.0}
    rep = explain_field_diff(sas_code=sas, row_v2=v2, row_v3=v3, target="ECL")
    assert rep.gradient is not None and rep.shapley is not None
    # Both methods should agree on the linear case
    g = {a.field: a.contribution for a in rep.gradient.attributions}
    s = {a.field: a.contribution for a in rep.shapley.attributions}
    for f in ("PD", "EAD"):
        assert g[f] == pytest.approx(s[f], abs=1e-6)
    assert "PD" in rep.suspects[:2] and "EAD" in rep.suspects[:2]

"""Tests for the KernelSHAP-style Shapley explainer.

Verifies:
- Efficiency axiom (Σφ_i = ΔY) for exact mode.
- Categorical inputs are attributed.
- Permutation mode gives unbiased estimates within a tolerance.
"""

from __future__ import annotations

import pytest

from src.sas_diff.shapley_explainer import shapley_values
from src.sas_logic_tree import SASLogicTree


def _parse(sas: str):
    return SASLogicTree().parse(sas)


def test_exact_efficiency_axiom():
    sas = "DATA out; SET in; ECL = PD * LGD * EAD; RUN;"
    v2 = {"PD": 0.05, "LGD": 0.40, "EAD": 100_000.0}
    v3 = {"PD": 0.06, "LGD": 0.45, "EAD": 110_000.0}
    rep = shapley_values(_parse(sas), v2, v3, target="ECL")
    assert rep.method == "exact"
    total = sum(a.contribution for a in rep.attributions)
    assert total == pytest.approx(rep.delta_y, abs=1e-9)
    assert abs(rep.unexplained_residual) < 1e-9


def test_categorical_attribution():
    sas = """
    DATA out;
        SET in;
        IF SEGMENT = 'CORP' THEN MULT = 1.5; ELSE MULT = 1.0;
        Y = X * MULT;
    RUN;
    """
    v2 = {"X": 100.0, "SEGMENT": "RETAIL"}    # Y = 100
    v3 = {"X": 100.0, "SEGMENT": "CORP"}      # Y = 150
    rep = shapley_values(_parse(sas), v2, v3, target="Y")
    by_field = {a.field: a for a in rep.attributions}
    assert "SEGMENT" in by_field
    assert by_field["SEGMENT"].is_numeric is False
    assert by_field["SEGMENT"].contribution == pytest.approx(50.0, abs=1e-6)


def test_attribution_to_only_changed_field():
    sas = "DATA out; SET in; Y = 2*A + 3*B + 4*C; RUN;"
    v2 = {"A": 1.0, "B": 2.0, "C": 3.0}
    v3 = {"A": 1.0, "B": 5.0, "C": 3.0}
    rep = shapley_values(_parse(sas), v2, v3, target="Y")
    assert len(rep.attributions) == 1
    assert rep.attributions[0].field == "B"
    assert rep.attributions[0].contribution == pytest.approx(9.0, abs=1e-9)


def test_handcrafted_three_feature_pipeline():
    """Hand-crafted pipeline with a known Shapley split.

    Y(0,0,0) = 0, Y(1,1,1) = A+B+A·B+C = 4. Δ = 4. By symmetry of A and B
    over the interaction term, φ_A = φ_B = 1.5, φ_C = 1.0.
    """
    sas = "DATA out; SET in; Y = A + B + A * B + C; RUN;"
    v2 = {"A": 0.0, "B": 0.0, "C": 0.0}
    v3 = {"A": 1.0, "B": 1.0, "C": 1.0}
    rep = shapley_values(_parse(sas), v2, v3, target="Y")
    by_field = {a.field: a.contribution for a in rep.attributions}
    total = sum(by_field.values())
    # Efficiency
    assert total == pytest.approx(4.0, abs=1e-9)
    assert rep.delta_y == pytest.approx(4.0, abs=1e-9)
    # Symmetry of A and B
    assert by_field["A"] == pytest.approx(by_field["B"], abs=1e-9)
    # The interaction term A·B splits evenly between A and B, so each gets
    # the linear contribution 1 plus 1/2 of the interaction value 1 → 1.5.
    assert by_field["A"] == pytest.approx(1.5, abs=1e-9)
    assert by_field["C"] == pytest.approx(1.0, abs=1e-9)


def test_permutation_mode_for_many_fields():
    """When there are >12 differing fields, the explainer falls back to
    sampling. The total should still approximate ΔY within tolerance."""
    # Linear sum of 15 features
    sas = "DATA out; SET in; Y = A + B + C + D + E + F + G + H + I + J + K + L + M + N + O; RUN;"
    fields = list("ABCDEFGHIJKLMNO")
    v2 = {f: 0.0 for f in fields}
    v3 = {f: 1.0 for f in fields}
    rep = shapley_values(_parse(sas), v2, v3, target="Y", n_samples=1000, exact_threshold=10)
    assert rep.method == "permutation"
    total = sum(a.contribution for a in rep.attributions)
    # Linear → permutation sampling is exact in expectation; with 1000 samples
    # the variance should be negligible for this fully separable function.
    assert total == pytest.approx(rep.delta_y, abs=1e-6)

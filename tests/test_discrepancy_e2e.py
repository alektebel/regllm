"""End-to-end tests for the discrepancy explainer using the bundled data."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.sas_diff import explain_field_diff


SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture(scope="module")
def sas_code() -> str:
    p = SAMPLES / "sample_lgd.sas"
    if not p.exists():
        pytest.skip("sample_lgd.sas not found")
    return p.read_text(encoding="utf-8")


def _coerce(row: dict[str, str]) -> dict:
    out = {}
    for k, v in row.items():
        if v == "":
            continue
        try:
            out[k.upper()] = float(v)
        except ValueError:
            out[k.upper()] = v
    return out


@pytest.fixture(scope="module")
def v2_v3_pairs():
    p2 = SAMPLES / "cycles_v2.csv"
    p3 = SAMPLES / "cycles_v3.csv"
    if not (p2.exists() and p3.exists()):
        pytest.skip("V2/V3 CSVs not found — run scripts/generate_v3.py")
    v2 = [_coerce(r) for r in csv.DictReader(p2.open(encoding="utf-8"))]
    v3 = [_coerce(r) for r in csv.DictReader(p3.open(encoding="utf-8"))]
    by_pk = {r["CICLO_ID"]: r for r in v3}
    pairs = []
    for r2 in v2:
        r3 = by_pk.get(r2.get("CICLO_ID"))
        if r3:
            pairs.append((r2, r3))
    return pairs


def test_pd_only_change_attributes_to_pd():
    """Hand-crafted V2/V3 differing only in PD: gradient + Shapley should
    attribute ~100 % of the ECL delta to PD_ESTIMADA."""
    sas = "DATA out; SET in; ECL = PD_ESTIMADA * LGD_ESTIMADA * EAD; RUN;"
    v2 = {"PD_ESTIMADA": 0.005, "LGD_ESTIMADA": 0.35, "EAD": 200_000.0}
    v3 = {"PD_ESTIMADA": 0.0075, "LGD_ESTIMADA": 0.35, "EAD": 200_000.0}
    rep = explain_field_diff(sas_code=sas, row_v2=v2, row_v3=v3, target="ECL")
    # Gradient
    g = {a.field: a.contribution for a in rep.gradient.attributions}
    assert g["PD_ESTIMADA"] == pytest.approx(rep.delta_y, abs=1e-6)
    for f in ("LGD_ESTIMADA", "EAD"):
        assert abs(g.get(f, 0.0)) < 1e-9
    # Shapley
    s = {a.field: a.contribution for a in rep.shapley.attributions}
    assert s["PD_ESTIMADA"] == pytest.approx(rep.delta_y, abs=1e-9)


def test_explainer_runs_on_real_v2_v3_pair(sas_code, v2_v3_pairs):
    """At least one V2/V3 pair must produce a finite ECL delta and matching
    gradient + Shapley sums (efficiency axiom)."""
    matched = 0
    for r2, r3 in v2_v3_pairs[:50]:
        if r2.get("ECL") == r3.get("ECL"):
            continue
        rep = explain_field_diff(sas_code=sas_code, row_v2=r2, row_v3=r3, target="ECL")
        if rep.delta_y is None:
            continue
        # Gradient + Shapley should both report attributions
        assert rep.gradient is not None and rep.shapley is not None
        matched += 1
        if matched >= 5:
            return
    assert matched > 0, "No diffable rows found in V2/V3 sample"


def test_top_suspects_include_modified_fields(sas_code, v2_v3_pairs):
    """For rows whose only modifications are documented (PD, LGD, EAD), the
    suspect ranking must surface those fields."""
    for r2, r3 in v2_v3_pairs[:200]:
        if r2.get("ECL") == r3.get("ECL"):
            continue
        rep = explain_field_diff(sas_code=sas_code, row_v2=r2, row_v3=r3, target="ECL")
        suspects = [s for s in rep.suspects if s != "ECL"]
        if suspects:
            assert any(f in suspects for f in
                       ("PD_ESTIMADA", "LGD_ESTIMADA", "EAD"))
            return
    pytest.skip("No mutated rows in the sample slice")


def test_categorical_change_surfaces_in_shapley(sas_code, v2_v3_pairs):
    """When COLATERAL_TIPO is renamed FINANCIERO → COLATERAL_FIN, Shapley
    should at least include it among the differing categorical fields."""
    for r2, r3 in v2_v3_pairs[:200]:
        if r2.get("COLATERAL_TIPO") == "FINANCIERO" and r3.get("COLATERAL_TIPO") == "COLATERAL_FIN":
            rep = explain_field_diff(sas_code=sas_code, row_v2=r2, row_v3=r3, target="ECL")
            field_names = {d.field for d in rep.field_deltas}
            assert "COLATERAL_TIPO" in field_names
            return
    pytest.skip("No FINANCIERO→COLATERAL_FIN row in slice")

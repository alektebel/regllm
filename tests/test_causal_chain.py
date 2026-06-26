"""Tests for causal chain builder."""

from __future__ import annotations

from src.agent.causal_chain import build_causal_chain
from src.agent.tools import _load_sas, _t_trace_causal_chain
from src.sas_logic_tree import SASLogicTree


def test_trace_causal_chain_ecl_debug_lgd() -> None:
    out = _t_trace_causal_chain("ECL", "debug_lgd")
    assert out["found"] is True
    assert out["narrative"]
    assert out["sas_comments"]
    assert "data" in out
    assert out["data"]["rows_missing_target"]

    fields = {s["field"] for s in out["steps"]}
    assert "ECL" in fields
    assert "MOC" in fields

    exprs = {s["field"]: s["expr"] for s in out["steps"] if s.get("expr")}
    assert "0.05 * LGD_ESTIMADA" in (exprs.get("MOC") or "")
    assert "PD_ESTIMADA * LGD_CON_MOC * EAD" in (exprs.get("ECL") or "")


def test_build_causal_chain_maps_source_file() -> None:
    sas = _load_sas("debug_lgd")
    tree = SASLogicTree()
    trace = tree.trace_lineage(tree.parse(sas), "MoC")
    out = build_causal_chain(trace, "debug_lgd")
    moc_step = next(s for s in out["steps"] if s["field"] == "MOC")
    assert moc_step.get("source_file") == "proj_03_suelos_lgd.sas"

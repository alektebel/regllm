"""Tests for pipeline data store and query_cycle_data tool."""

from __future__ import annotations

from src.agent.pipeline_store import query_cycle_data, sample_for_causal_chain
from src.agent.tools import _t_query_cycle_data, _t_trace_causal_chain


def test_query_cycle_by_pk() -> None:
    out = query_cycle_data("debug_lgd", ciclo_id="CIC_00042")
    assert out["found"] is True
    row = out["rows"][0]
    assert row["CICLO_ID"] == "CIC_00042"
    assert row["SW_FUSION"] == 1
    assert row["LGD_ESTIMADA"] is None
    assert row["ECL"] is None


def test_query_missing_ecl() -> None:
    out = query_cycle_data(
        "debug_lgd",
        missing_field="ECL",
        fields=["CICLO_ID", "SW_FUSION", "LGD_ESTIMADA", "MOC", "ECL"],
        limit=10,
    )
    assert out["row_count"] >= 3
    for row in out["rows"]:
        assert row["ECL"] is None


def test_trace_causal_chain_includes_data() -> None:
    out = _t_trace_causal_chain("ECL", "debug_lgd")
    assert "data" in out
    assert out["data"]["missing_count"] >= 3
    assert out["data"]["rows_missing_target"]
    sample = out["data"]["rows_missing_target"][0]
    assert sample.get("LGD_ESTIMADA") is None
    assert "Valores reales en BD" in out["narrative"]


def test_dispatch_query_cycle_data() -> None:
    out = _t_query_cycle_data(session_id="debug_lgd", ciclo_id="CIC_00044")
    assert out["rows"][0]["ECL"] == 1680.0
    assert out["rows"][0]["SW_FUSION"] == 1

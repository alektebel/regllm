"""Unit tests for the agent tool dispatcher and a sampling of tools."""

from __future__ import annotations

from src.agent import TOOL_REGISTRY, dispatch_tool, tool_schemas


def test_registry_has_expected_tools() -> None:
    expected = {
        "find_row",
        "find_rows_by_field_value",
        "inspect_lineage",
        "trace_dependencies",
        "compute_attribution",
        "compare_sas_versions",
        "search_docs",
        "get_field_definition",
        "search_changelog",
        "search_regulation",
        "enrich_fields",
        "get_schema_context",
        "validate_sas",
        # Knowledge graph tools
        "query_regulation",
        "find_governing_rules",
        "trace_regulation_chain",
        "search_experience",
        "save_insight",
    }
    assert expected == set(TOOL_REGISTRY)


def test_tool_schemas_are_well_formed() -> None:
    schemas = tool_schemas()
    assert len(schemas) == len(TOOL_REGISTRY)
    for s in schemas:
        assert s["type"] == "function"
        f = s["function"]
        assert isinstance(f["name"], str) and f["name"]
        assert isinstance(f["description"], str)
        assert isinstance(f["parameters"], dict)
        assert f["parameters"]["type"] == "object"


def test_dispatch_unknown_tool_returns_error_dict() -> None:
    out = dispatch_tool("does_not_exist", {})
    assert "error" in out
    assert out["error"].startswith("unknown_tool")
    assert "available" in out


def test_dispatch_bad_arguments_returns_error_dict() -> None:
    out = dispatch_tool("find_row", {"completely_wrong_arg": True})
    assert "error" in out


def test_inspect_lineage_returns_ancestors_for_ecl(monkeypatch) -> None:
    """Uses the bundled sample_lgd.sas to test lineage for ECL."""
    from pathlib import Path
    sample = Path(__file__).resolve().parent.parent / "data" / "samples" / "sample_lgd.sas"
    sas_text = sample.read_text(encoding="utf-8")
    import src.agent.tools as _tools
    monkeypatch.setattr(_tools, "_load_sas", lambda version: sas_text)
    from src.agent.tools import _t_inspect_lineage
    out = _t_inspect_lineage("ECL", "v3")
    assert "ancestors" in out
    # ECL = PD * LGD * EAD → expect those three among the ancestors
    expected = {"PD_ESTIMADA", "LGD_ESTIMADA", "EAD"}
    assert expected.issubset(set(out["ancestors"]))


def test_find_row_unknown_pk() -> None:
    out = dispatch_tool("find_row", {"pk": "DOES_NOT_EXIST", "version": "v3"})
    assert out["found"] is False


def test_compare_sas_versions_returns_full_shape() -> None:
    out = dispatch_tool("compare_sas_versions", {"target": "ECL"})
    if out.get("truncated"):
        # Large diff gets truncated by _truncate() — verify preview is present
        assert "preview" in out
    else:
        for key in ("v2_present", "v3_present", "added_steps",
                    "removed_steps", "modified_steps", "unchanged_steps", "unified_diff"):
            assert key in out, f"missing key {key}"


def test_search_docs_runs_against_default_index() -> None:
    out = dispatch_tool("search_docs", {"query": "EAD", "k": 3})
    assert "hits" in out
    assert isinstance(out["hits"], list)


def test_trace_dependencies_returns_edges(monkeypatch) -> None:
    """trace_dependencies should return edges with expr and kind."""
    from pathlib import Path
    sample = Path(__file__).resolve().parent.parent / "data" / "samples" / "sample_lgd.sas"
    sas_text = sample.read_text(encoding="utf-8")
    import src.agent.tools as _tools
    monkeypatch.setattr(_tools, "_load_sas", lambda version: sas_text)
    from src.agent.tools import _t_trace_dependencies
    out = _t_trace_dependencies("ECL", "v3")
    assert out["found"] is True
    assert out["ancestor_count"] >= 3
    assert "edges" in out
    assert len(out["edges"]) > 0
    edge_targets = {e["target"] for e in out["edges"]}
    assert "ECL" in edge_targets
    edge_sources = {e["source"] for e in out["edges"]}
    assert {"PD_ESTIMADA", "LGD_ESTIMADA", "EAD"}.issubset(edge_sources)


def test_trace_dependencies_max_depth(monkeypatch) -> None:
    from pathlib import Path
    sample = Path(__file__).resolve().parent.parent / "data" / "samples" / "sample_lgd.sas"
    sas_text = sample.read_text(encoding="utf-8")
    import src.agent.tools as _tools
    monkeypatch.setattr(_tools, "_load_sas", lambda version: sas_text)
    from src.agent.tools import _t_trace_dependencies
    out = _t_trace_dependencies("ECL", "v3", max_depth=1)
    # With depth=1, only direct predecessors
    for field, d in out["depth"].items():
        assert d <= 1


def test_enrich_fields_returns_definitions() -> None:
    from src.agent.tools import _t_enrich_fields
    out = _t_enrich_fields(["EAD", "PD_ESTIMADA"])
    assert out["count"] == 2
    assert "EAD" in out["enriched"]
    assert "PD_ESTIMADA" in out["enriched"]
    for field_data in out["enriched"].values():
        assert "definition" in field_data


def test_enrich_fields_caps_at_15() -> None:
    from src.agent.tools import _t_enrich_fields
    many = [f"FIELD_{i}" for i in range(20)]
    out = _t_enrich_fields(many)
    assert out["count"] == 15


def test_get_schema_context_finds_tables() -> None:
    from src.agent.tools import _t_get_schema_context
    out = _t_get_schema_context("PD")
    assert "tables" in out
    # Should find at least parametros_irb which has PD columns
    table_names = [t["table"] for t in out["tables"]]
    assert len(table_names) > 0


def test_get_schema_context_filter_by_table() -> None:
    from src.agent.tools import _t_get_schema_context
    out = _t_get_schema_context("anything", tables=["contratos"])
    table_names = [t["table"] for t in out["tables"]]
    assert table_names == ["contratos"]


def test_validate_sas_detects_missing_dataset(monkeypatch) -> None:
    import src.agent.tools as _tools
    code = "DATA work.out; SET missing_table; Z = 1; RUN;"
    monkeypatch.setattr(_tools, "_load_sas", lambda version: code)
    from src.agent.tools import _t_validate_sas
    out = _t_validate_sas("v3")
    codes = [d["code"] for d in out["diagnostics"]]
    assert "MISSING_DATASET" in codes


def test_validate_sas_clean_code(monkeypatch) -> None:
    import src.agent.tools as _tools
    code = "DATA work.out; SET work.inp; X = A + B; RUN;"
    monkeypatch.setattr(_tools, "_load_sas", lambda version: code)
    from src.agent.tools import _t_validate_sas
    out = _t_validate_sas("v3")
    errors = [d for d in out["diagnostics"] if d["level"] == "error"]
    assert len(errors) == 0

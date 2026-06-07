"""Unit tests for the agent tool dispatcher and a sampling of tools."""

from __future__ import annotations

from src.agent import TOOL_REGISTRY, dispatch_tool, tool_schemas


def test_registry_has_expected_tools() -> None:
    expected = {
        "find_row",
        "find_rows_by_field_value",
        "inspect_lineage",
        "compute_attribution",
        "compare_sas_versions",
        "search_docs",
        "get_field_definition",
        "search_changelog",
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


def test_inspect_lineage_returns_ancestors_for_ecl() -> None:
    """Uses the bundled sample_lgd.sas as the fallback when data/sas is empty."""
    out = dispatch_tool("inspect_lineage", {"target": "ECL", "sas_version": "v3"})
    assert "ancestors" in out
    # ECL = PD * LGD * EAD → expect those three among the ancestors
    expected = {"PD_ESTIMADA", "LGD_ESTIMADA", "EAD"}
    assert expected.issubset(set(out["ancestors"]))


def test_find_row_unknown_pk() -> None:
    out = dispatch_tool("find_row", {"pk": "DOES_NOT_EXIST", "version": "v3"})
    assert out["found"] is False


def test_compare_sas_versions_returns_full_shape() -> None:
    out = dispatch_tool("compare_sas_versions", {"target": "ECL"})
    for key in ("v2_present", "v3_present", "added_steps",
                "removed_steps", "modified_steps", "unchanged_steps", "unified_diff"):
        assert key in out, f"missing key {key}"


def test_search_docs_runs_against_default_index() -> None:
    out = dispatch_tool("search_docs", {"query": "EAD", "k": 3})
    assert "hits" in out
    assert isinstance(out["hits"], list)

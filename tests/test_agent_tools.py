"""Unit tests for the agent tool dispatcher and a sampling of tools."""

from __future__ import annotations

from src.agent import TOOL_REGISTRY, dispatch_tool, tool_schemas


def test_registry_has_expected_tools() -> None:
    expected = {
        "find_row",
        "find_rows_by_field_value",
        "inspect_lineage",
        "trace_dependencies",
        "get_sas_formula",
        "trace_causal_chain",
        "query_cycle_data",
        "compute_shapley",
        "search_docs",
        "get_field_definition",
        "search_changelog",
        "search_regulation",
        "enrich_fields",
        "get_schema_context",
        "validate_sas",
        "describe_egp_project",
        # Knowledge graph tools
        "query_regulation",
        "find_governing_rules",
        "trace_regulation_chain",
        "search_experience",
        "save_insight",
        "save_feedback",
        # Sleep-cycle memory organization
        "merge_nodes",
        "add_tags",
        "link_nodes",
        "flag_conflict",
        "delete_node",
        "create_concept",
        "link_to_concept",
        # Investigation tools
        "backtrace_sas_field",
        "create_investigation_plan",
        "formulate_data_request",
        "analyze_uploaded_data",
    }
    assert expected.issubset(set(TOOL_REGISTRY))


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
    monkeypatch.setattr(_tools, "_load_sas", lambda session_id="default": sas_text)
    from src.agent.tools import _t_inspect_lineage
    out = _t_inspect_lineage("ECL")
    assert "ancestors" in out
    # ECL = PD * LGD * EAD → expect those three among the ancestors
    expected = {"PD_ESTIMADA", "LGD_ESTIMADA", "EAD"}
    assert expected.issubset(set(out["ancestors"]))


def test_find_row_unknown_pk() -> None:
    out = dispatch_tool("find_row", {"pk": "DOES_NOT_EXIST"})
    assert out["found"] is False



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
    monkeypatch.setattr(_tools, "_load_sas", lambda session_id="default": sas_text)
    from src.agent.tools import _t_trace_dependencies
    out = _t_trace_dependencies("ECL")
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
    monkeypatch.setattr(_tools, "_load_sas", lambda session_id="default": sas_text)
    from src.agent.tools import _t_trace_dependencies
    out = _t_trace_dependencies("ECL", max_depth=1)
    # With depth=1, only direct predecessors
    for field, d in out["depth"].items():
        assert d <= 1


def test_get_sas_formula_returns_expr(monkeypatch) -> None:
    from pathlib import Path
    sample = Path(__file__).resolve().parent.parent / "data" / "samples" / "sample_lgd.sas"
    sas_text = sample.read_text(encoding="utf-8")
    import src.agent.tools as _tools
    monkeypatch.setattr(_tools, "_load_sas", lambda session_id="default": sas_text)
    from src.agent.tools import _t_get_sas_formula
    out = _t_get_sas_formula("ECL")
    assert out["found"] is True
    assert out["definitions"]
    exprs = {d["expr"] for d in out["definitions"]}
    assert any("PD_ESTIMADA" in e and "EAD" in e for e in exprs)


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
    monkeypatch.setattr(_tools, "_load_sas", lambda session_id="default": code)
    from src.agent.tools import _t_validate_sas
    out = _t_validate_sas()
    codes = [d["code"] for d in out["diagnostics"]]
    assert "MISSING_DATASET" in codes


def test_validate_sas_clean_code(monkeypatch) -> None:
    import src.agent.tools as _tools
    code = "DATA work.out; SET work.inp; X = A + B; RUN;"
    monkeypatch.setattr(_tools, "_load_sas", lambda session_id="default": code)
    from src.agent.tools import _t_validate_sas
    out = _t_validate_sas()
    errors = [d for d in out["diagnostics"] if d["level"] == "error"]
    assert len(errors) == 0


# ── New investigation tools ─────────────────────────────────────────────────


def test_backtrace_sas_field(monkeypatch) -> None:
    from pathlib import Path
    sample = Path(__file__).resolve().parent.parent / "data" / "samples" / "sample_lgd.sas"
    sas_text = sample.read_text(encoding="utf-8")
    import src.agent.tools as _tools
    monkeypatch.setattr(_tools, "_load_sas", lambda session_id="default": sas_text)
    from src.agent.tools import _t_backtrace_sas_field
    out = _t_backtrace_sas_field("ECL")
    assert out["found"] is True
    assert out["ancestor_count"] >= 3
    assert len(out["input_tables"]) > 0
    # Each input table entry should have fields
    for t in out["input_tables"]:
        assert "table" in t
        assert "fields" in t
        assert len(t["fields"]) > 0


def test_backtrace_sas_field_not_found(monkeypatch) -> None:
    import src.agent.tools as _tools
    monkeypatch.setattr(_tools, "_load_sas", lambda version: "DATA work.out; X = 1; RUN;")
    from src.agent.tools import _t_backtrace_sas_field
    out = _t_backtrace_sas_field("NONEXISTENT_FIELD")
    assert out["found"] is False


def test_formulate_data_request() -> None:
    from src.agent.tools import _t_formulate_data_request
    out = _t_formulate_data_request(
        extractions=[
            {"table": "contratos", "fields": ["PD_ESTIMADA", "LGD_ESTIMADA"], "cycles": ["202312"]},
            {"table": "garantias", "fields": ["VALOR_GARANTIA"], "description": "Collateral values"},
        ],
        reason="Need to verify LGD floor application",
    )
    assert out["data_request"] is True
    assert out["count"] == 2
    assert out["extractions"][0]["table"] == "contratos"
    assert "PD_ESTIMADA" in out["extractions"][0]["fields"]


def test_analyze_uploaded_data_csv(tmp_path, monkeypatch) -> None:
    # Create a test CSV in data/uploads/ under fake project root
    import src.agent.tools as _tools
    uploads = tmp_path / "data" / "uploads"
    uploads.mkdir(parents=True)
    csv_file = uploads / "test.csv"
    csv_file.write_text("CICLO_ID,PD_ESTIMADA,LGD_ESTIMADA\nCIC_001,0.05,0.45\nCIC_002,0.10,0.50\n")
    monkeypatch.setattr(_tools, "_PROJECT_ROOT", tmp_path)
    from src.agent.tools import _t_analyze_uploaded_data
    out = _t_analyze_uploaded_data("test.csv")
    assert out["row_count"] == 2
    assert "PD_ESTIMADA" in out["columns"]
    assert len(out["preview"]) == 2
    assert "PD_ESTIMADA" in out["stats"]


def test_analyze_uploaded_data_with_filter(tmp_path, monkeypatch) -> None:
    import src.agent.tools as _tools
    uploads = tmp_path / "data" / "uploads"
    uploads.mkdir(parents=True)
    csv_file = uploads / "data.csv"
    csv_file.write_text("CICLO_ID,VALUE\nCIC_001,10\nCIC_002,20\nCIC_001,30\n")
    monkeypatch.setattr(_tools, "_PROJECT_ROOT", tmp_path)
    from src.agent.tools import _t_analyze_uploaded_data
    out = _t_analyze_uploaded_data("data.csv", cycle_filter="CIC_001")
    assert out["row_count"] == 2  # only CIC_001 rows


def test_analyze_uploaded_data_missing_file(tmp_path, monkeypatch) -> None:
    import src.agent.tools as _tools
    uploads = tmp_path / "data" / "uploads"
    uploads.mkdir(parents=True)
    monkeypatch.setattr(_tools, "_PROJECT_ROOT", tmp_path)
    from src.agent.tools import _t_analyze_uploaded_data
    out = _t_analyze_uploaded_data("nonexistent.csv")
    assert "error" in out


def test_create_investigation_plan(monkeypatch) -> None:
    """Test plan creation with a mocked GraphStore."""
    from unittest.mock import MagicMock
    import src.agent.tools as _tools

    mock_store = MagicMock()
    mock_store.get_node.return_value = None  # no linked nodes
    monkeypatch.setattr(_tools, "_graph_store", mock_store)

    from src.agent.tools import _t_create_investigation_plan
    out = _t_create_investigation_plan(
        target_field="LGD_ESTIMADA",
        problem="LGD shows 0 for cycle 202312 on guaranteed segments",
        steps=[
            "Trace LGD_ESTIMADA dependencies",
            "Check LGD floor regulation",
            "Request data for cycle 202312",
        ],
        cycles=["202312"],
    )
    assert out["saved"] is True
    assert "plan" in out
    assert "LGD_ESTIMADA" in out["plan"]
    assert "202312" in out["plan"]
    mock_store.add_node.assert_called_once()


def test_save_feedback(monkeypatch) -> None:
    """Test feedback saving with a mocked GraphStore."""
    from unittest.mock import MagicMock
    import src.agent.tools as _tools

    mock_store = MagicMock()
    mock_store.get_node.return_value = None
    monkeypatch.setattr(_tools, "_graph_store", mock_store)

    from src.agent.tools import _t_save_feedback
    out = _t_save_feedback(
        feedback_type="correction",
        content="The LGD floor is 0.25 not 0.20",
        original_claim="LGD floor is 0.20 for CORP segment",
        corrected_understanding="LGD floor for CORP changed to 0.25 in Circular 4/2022",
        related_fields=["LGD_ESTIMADA"],
    )
    assert out["saved"] is True
    # Verify the Insight node was created with priority=1.0
    call_args = mock_store.add_node.call_args
    node = call_args[0][0]
    assert node.priority == 1.0
    assert node.feedback_type == "correction"
    assert "feedback" in node.tags


def test_search_experience_priority_sort(monkeypatch) -> None:
    """Verify search_experience returns feedback (priority=1.0) before auto-insights."""
    from unittest.mock import MagicMock
    from src.knowledge.ontology import Insight, NodeType
    import src.agent.tools as _tools

    auto_insight = Insight(
        id="ins:auto", label="auto insight", summary="discovered", priority=0.5,
    )
    feedback_insight = Insight(
        id="ins:fb", label="feedback insight", summary="corrected", priority=1.0,
        feedback_type="correction",
    )

    mock_store = MagicMock()
    mock_store.search_nodes.return_value = [auto_insight, feedback_insight]
    monkeypatch.setattr(_tools, "_graph_store", mock_store)

    from src.agent.tools import _t_search_experience
    out = _t_search_experience("LGD", k=5)
    assert out["count"] == 2
    # Feedback should come first
    assert out["results"][0]["id"] == "ins:fb"
    assert out["results"][1]["id"] == "ins:auto"

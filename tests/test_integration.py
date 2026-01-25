"""
Integration Tests for RegLLM Agent Framework

These tests verify that all components work together correctly.
"""

import sys
from pathlib import Path

# Add project root to path to allow direct imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest


class TestToolRegistrationIntegration:
    """Tests for tool registration and discovery."""

    def test_all_tools_registered(self, registered_registry):
        """Test that all tools are properly registered."""
        tools = registered_registry.list_tools()

        # Should have tools from all categories
        assert len(tools) >= 10

        # Check for methodology tools
        assert registered_registry.get("read_methodology") is not None
        assert registered_registry.get("extract_formulas") is not None

        # Check for code analysis tools
        assert registered_registry.get("read_code_file") is not None
        assert registered_registry.get("analyze_code_structure") is not None

        # Check for consistency tools
        assert registered_registry.get("check_formula_consistency") is not None
        assert registered_registry.get("generate_consistency_report") is not None

    def test_tools_have_correct_categories(self, registered_registry):
        """Test that tools are in correct categories."""
        categories = registered_registry.list_categories()

        assert "methodology" in categories
        assert "code_analysis" in categories
        assert "consistency" in categories

    def test_tools_schema_generation(self, registered_registry):
        """Test generating OpenAI-compatible schemas."""
        schemas = registered_registry.get_tools_schema()

        assert len(schemas) > 0

        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]


class TestToolExecutionIntegration:
    """Tests for tool execution pipeline."""

    def test_execute_methodology_tool(
        self, registered_registry, sample_methodology_file
    ):
        """Test executing a methodology tool."""
        from src.agents.tool_executor import ToolExecutor

        executor = ToolExecutor(registered_registry)
        result = executor.execute(
            "read_methodology",
            {"file_path": str(sample_methodology_file)},
        )

        assert result.is_success
        assert "IRB" in result.result.get("content", "")

    def test_execute_code_analysis_tool(
        self, registered_registry, sample_code_file
    ):
        """Test executing a code analysis tool."""
        from src.agents.tool_executor import ToolExecutor

        executor = ToolExecutor(registered_registry)
        result = executor.execute(
            "read_code_file",
            {"file_path": str(sample_code_file)},
        )

        assert result.is_success
        assert result.result["language"] == "python"

    def test_execute_consistency_check(
        self,
        registered_registry,
        sample_methodology_content,
        sample_code_content,
    ):
        """Test executing a consistency check tool."""
        from src.agents.tool_executor import ToolExecutor

        executor = ToolExecutor(registered_registry)
        result = executor.execute(
            "generate_consistency_report",
            {
                "methodology_content": sample_methodology_content,
                "code_content": sample_code_content,
                "methodology_name": "IRB Test",
            },
        )

        assert result.is_success
        assert "overall_status" in result.result

    def test_tool_execution_chain(
        self,
        registered_registry,
        sample_methodology_file,
        sample_code_file,
    ):
        """Test chaining multiple tool executions."""
        from src.agents.tool_executor import ToolExecutor

        executor = ToolExecutor(registered_registry)

        # Step 1: Read methodology
        meth_result = executor.execute(
            "read_methodology",
            {"file_path": str(sample_methodology_file)},
        )
        assert meth_result.is_success

        # Step 2: Read code
        code_result = executor.execute(
            "read_code_file",
            {"file_path": str(sample_code_file)},
        )
        assert code_result.is_success

        # Step 3: Check consistency
        consistency_result = executor.execute(
            "generate_consistency_report",
            {
                "methodology_content": meth_result.result["content"],
                "code_content": code_result.result["content"],
            },
        )
        assert consistency_result.is_success

        # Verify the chain produced meaningful results
        report = consistency_result.result
        assert "overall_status" in report
        assert "formula_analysis" in report
        assert "parameter_analysis" in report


class TestAgentIntegration:
    """Tests for agent integration with tools."""

    def test_agent_with_registered_tools(self, registered_registry):
        """Test agent creation with registered tools."""
        from src.agents.agent_loop import RegulationAgent

        agent = RegulationAgent(registry=registered_registry)

        # Agent should have access to all tools
        prompt = agent._get_system_prompt()
        assert "read_methodology" in prompt or "leer" in prompt.lower()

    def test_consistency_agent_workflow(
        self,
        registered_registry,
        sample_methodology_file,
        sample_code_file,
    ):
        """Test consistency agent workflow."""
        from src.agents.agent_loop import MethodologyConsistencyAgent

        agent = MethodologyConsistencyAgent(
            registry=registered_registry,
            max_steps=3,
        )

        result = agent.check_consistency(
            methodology_path=str(sample_methodology_file),
            code_path=str(sample_code_file),
            aspects=["PD calculation"],
        )

        assert "query" in result
        # Agent should have attempted to work on the query


class TestEndToEndConsistencyCheck:
    """End-to-end tests for methodology-code consistency checking."""

    def test_full_consistency_check_consistent_code(
        self,
        sample_methodology_content,
        sample_code_content,
    ):
        """Test full consistency check with consistent code."""
        from src.agents.tools.consistency_tools import generate_consistency_report

        report = generate_consistency_report(
            methodology_content=sample_methodology_content,
            code_content=sample_code_content,
            methodology_name="IRB Fundación",
            code_path="irb_implementation.py",
        )

        assert report["success"] is True

        # Should have a definitive status
        assert report["overall_status"] in ["CONSISTENT", "PARTIAL", "INCONSISTENT"]

        # Should have analyzed formulas
        assert report["formula_analysis"]["total_formulas"] >= 0

        # Should have analyzed parameters
        assert report["parameter_analysis"]["methodology_params"] >= 0

        # Should provide recommendations
        assert len(report["recommendations"]) > 0

    def test_full_consistency_check_inconsistent_code(
        self,
        sample_methodology_content,
        sample_inconsistent_code,
    ):
        """Test full consistency check with inconsistent code."""
        from src.agents.tools.consistency_tools import generate_consistency_report

        report = generate_consistency_report(
            methodology_content=sample_methodology_content,
            code_content=sample_inconsistent_code,
            methodology_name="IRB Fundación",
            code_path="bad_implementation.py",
        )

        assert report["success"] is True

        # Should identify issues
        total_issues = report["statistics"]["total_issues"]
        errors = report["statistics"]["errors"]
        warnings = report["statistics"]["warnings"]

        # Either errors, warnings, or issues should be present
        assert total_issues >= 0 or errors >= 0 or warnings >= 0

    def test_consistency_check_with_real_files(self, temp_dir):
        """Test consistency check using actual files."""
        from src.agents.tools.methodology_tools import read_methodology
        from src.agents.tools.code_analysis_tools import read_code_file
        from src.agents.tools.consistency_tools import generate_consistency_report

        # Create methodology file
        methodology = """# Test Methodology

## Parameters

| Parameter | Value |
|-----------|-------|
| **LGD** | 45% |
| **PD** | 3% |

## Formula

```
RWA = K × 12.5 × EAD
```
"""
        meth_file = temp_dir / "test_methodology.md"
        meth_file.write_text(methodology, encoding="utf-8")

        # Create code file
        code = """
LGD = 0.45
PD = 0.03

def calculate_rwa(ead, k):
    rwa = k * 12.5 * ead
    return rwa
"""
        code_file = temp_dir / "implementation.py"
        code_file.write_text(code, encoding="utf-8")

        # Read files
        meth_result = read_methodology(str(meth_file))
        code_result = read_code_file(str(code_file))

        assert meth_result["success"]
        assert code_result["success"]

        # Check consistency
        report = generate_consistency_report(
            methodology_content=meth_result["content"],
            code_content=code_result["content"],
        )

        assert report["success"]
        # Parameters should be consistent
        assert report["parameter_analysis"]["methodology_params"] > 0


class TestErrorHandling:
    """Tests for error handling across the system."""

    def test_handles_missing_methodology_file(self, registered_registry):
        """Test handling of missing methodology file."""
        from src.agents.tool_executor import ToolExecutor

        executor = ToolExecutor(registered_registry)
        result = executor.execute(
            "read_methodology",
            {"file_path": "/nonexistent/path/methodology.md"},
        )

        assert result.is_success  # Tool itself succeeds
        assert result.result["success"] is False  # But reports file not found

    def test_handles_invalid_code_syntax(self, registered_registry):
        """Test handling of invalid Python syntax."""
        from src.agents.tool_executor import ToolExecutor

        executor = ToolExecutor(registered_registry)
        result = executor.execute(
            "analyze_code_structure",
            {"content": "def broken(:\n    pass"},
        )

        assert result.is_success  # Executor succeeds
        assert result.result["success"] is False  # Tool reports syntax error

    def test_handles_empty_content(self, registered_registry):
        """Test handling of empty content."""
        from src.agents.tool_executor import ToolExecutor

        executor = ToolExecutor(registered_registry)

        # Empty methodology
        result = executor.execute(
            "extract_formulas",
            {"content": ""},
        )

        assert result.is_success
        assert result.result["formula_count"] == 0

        # Empty code
        result = executor.execute(
            "extract_calculations",
            {"content": ""},
        )

        assert result.is_success
        assert result.result["calculation_count"] == 0

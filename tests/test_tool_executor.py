"""
Tests for Tool Executor
"""

import sys
from pathlib import Path

# Add project root to path to allow direct imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import json

# Direct imports to avoid src/__init__.py heavy dependencies
from src.agents.tool_registry import Tool, ToolRegistry
from src.agents.tool_executor import (
    ToolExecutor,
    ToolResult,
    ExecutionStatus,
)


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self):
        """Test creating a successful result."""
        result = ToolResult(
            tool_name="test",
            status=ExecutionStatus.SUCCESS,
            result={"data": "value"},
        )

        assert result.is_success
        assert result.tool_name == "test"
        assert result.result["data"] == "value"

    def test_error_result(self):
        """Test creating an error result."""
        result = ToolResult(
            tool_name="test",
            status=ExecutionStatus.ERROR,
            error="Something went wrong",
            error_type="ValueError",
        )

        assert not result.is_success
        assert result.error == "Something went wrong"
        assert result.error_type == "ValueError"

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ToolResult(
            tool_name="test",
            status=ExecutionStatus.SUCCESS,
            result=42,
            execution_time_ms=100.5,
        )

        d = result.to_dict()

        assert d["tool_name"] == "test"
        assert d["status"] == "success"
        assert d["result"] == 42
        assert d["execution_time_ms"] == 100.5

    def test_to_json(self):
        """Test converting result to JSON."""
        result = ToolResult(
            tool_name="test",
            status=ExecutionStatus.SUCCESS,
            result={"key": "value"},
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["tool_name"] == "test"
        assert parsed["result"]["key"] == "value"

    def test_format_for_llm_success(self):
        """Test formatting successful result for LLM."""
        result = ToolResult(
            tool_name="test",
            status=ExecutionStatus.SUCCESS,
            result={"items": [1, 2, 3]},
        )

        formatted = result.format_for_llm()
        assert "items" in formatted
        assert "[1, 2, 3]" in formatted or "1" in formatted

    def test_format_for_llm_error(self):
        """Test formatting error result for LLM."""
        result = ToolResult(
            tool_name="test",
            status=ExecutionStatus.ERROR,
            error="Tool failed",
        )

        formatted = result.format_for_llm()
        assert "Error" in formatted
        assert "test" in formatted
        assert "Tool failed" in formatted


class TestToolExecutor:
    """Tests for ToolExecutor class."""

    @pytest.fixture
    def executor_with_tools(self):
        """Create executor with sample tools."""
        registry = ToolRegistry()

        def add(a: int, b: int) -> int:
            return a + b

        def multiply(x: int, y: int) -> int:
            return x * y

        def failing_tool():
            raise ValueError("This tool always fails")

        tools = [
            Tool(
                name="add",
                description="Add two numbers",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
                function=add,
            ),
            Tool(
                name="multiply",
                description="Multiply two numbers",
                parameters={
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                    "required": ["x", "y"],
                },
                function=multiply,
            ),
            Tool(
                name="failing",
                description="A tool that fails",
                parameters={"type": "object", "properties": {}},
                function=failing_tool,
            ),
        ]

        for t in tools:
            registry.register(t)

        return ToolExecutor(registry)

    def test_execute_success(self, executor_with_tools):
        """Test successful tool execution."""
        result = executor_with_tools.execute("add", {"a": 5, "b": 3})

        assert result.is_success
        assert result.result == 8
        assert result.execution_time_ms >= 0

    def test_execute_not_found(self, executor_with_tools):
        """Test executing non-existent tool."""
        result = executor_with_tools.execute("nonexistent", {})

        assert not result.is_success
        assert result.status == ExecutionStatus.NOT_FOUND
        assert "no encontrada" in result.error.lower()

    def test_execute_with_validation_error(self, executor_with_tools):
        """Test execution with missing required parameter."""
        result = executor_with_tools.execute("add", {"a": 5})  # Missing 'b'

        assert not result.is_success
        assert result.status == ExecutionStatus.VALIDATION_ERROR
        assert "b" in result.error

    def test_execute_tool_error(self, executor_with_tools):
        """Test execution when tool raises exception."""
        result = executor_with_tools.execute("failing", {})

        assert not result.is_success
        assert result.status == ExecutionStatus.ERROR
        assert result.error_type == "ValueError"

    def test_execute_without_validation(self, executor_with_tools):
        """Test execution with validation disabled on required params."""
        # Execute with validation disabled - params still need to match function signature
        result = executor_with_tools.execute(
            "add",
            {"a": 5, "b": 3},
            validate=False,
        )

        assert result.is_success
        assert result.result == 8

    def test_validate_parameters_type_checking(self, executor_with_tools):
        """Test parameter type validation."""
        tool = executor_with_tools.registry.get("add")

        # Valid types
        is_valid, error = executor_with_tools.validate_parameters(
            tool, {"a": 5, "b": 3}
        )
        assert is_valid

        # Invalid type
        is_valid, error = executor_with_tools.validate_parameters(
            tool, {"a": "not_an_int", "b": 3}
        )
        assert not is_valid
        assert "tipo" in error.lower()

    def test_execute_multiple(self, executor_with_tools):
        """Test executing multiple tools in sequence."""
        calls = [
            {"name": "add", "parameters": {"a": 1, "b": 2}},
            {"name": "multiply", "parameters": {"x": 3, "y": 4}},
        ]

        results = executor_with_tools.execute_multiple(calls)

        assert len(results) == 2
        assert results[0].result == 3
        assert results[1].result == 12

    def test_execute_multiple_stop_on_error(self, executor_with_tools):
        """Test stopping execution on first error."""
        calls = [
            {"name": "add", "parameters": {"a": 1, "b": 2}},
            {"name": "failing", "parameters": {}},
            {"name": "multiply", "parameters": {"x": 3, "y": 4}},
        ]

        results = executor_with_tools.execute_multiple(calls, stop_on_error=True)

        # Should stop after failing tool
        assert len(results) == 2
        assert results[0].is_success
        assert not results[1].is_success

    def test_parse_and_execute(self, executor_with_tools):
        """Test parsing JSON and executing."""
        json_call = json.dumps({
            "name": "add",
            "parameters": {"a": 10, "b": 20},
        })

        result = executor_with_tools.parse_and_execute(json_call)

        assert result.is_success
        assert result.result == 30

    def test_parse_and_execute_invalid_json(self, executor_with_tools):
        """Test parsing invalid JSON."""
        result = executor_with_tools.parse_and_execute("not valid json")

        assert not result.is_success
        assert result.status == ExecutionStatus.VALIDATION_ERROR
        assert "JSON" in result.error

    def test_execution_history(self, executor_with_tools):
        """Test that execution history is maintained."""
        executor_with_tools.execute("add", {"a": 1, "b": 2})
        executor_with_tools.execute("multiply", {"x": 3, "y": 4})

        history = executor_with_tools.get_history()

        assert len(history) == 2
        assert history[0].tool_name == "add"
        assert history[1].tool_name == "multiply"

    def test_execution_history_filter_by_tool(self, executor_with_tools):
        """Test filtering history by tool name."""
        executor_with_tools.execute("add", {"a": 1, "b": 2})
        executor_with_tools.execute("multiply", {"x": 3, "y": 4})
        executor_with_tools.execute("add", {"a": 5, "b": 6})

        add_history = executor_with_tools.get_history(tool_name="add")

        assert len(add_history) == 2
        assert all(h.tool_name == "add" for h in add_history)

    def test_execution_history_filter_by_status(self, executor_with_tools):
        """Test filtering history by status."""
        executor_with_tools.execute("add", {"a": 1, "b": 2})
        executor_with_tools.execute("failing", {})
        executor_with_tools.execute("add", {"a": 3, "b": 4})

        success_history = executor_with_tools.get_history(
            status=ExecutionStatus.SUCCESS
        )

        assert len(success_history) == 2
        assert all(h.is_success for h in success_history)

    def test_get_available_tools(self, executor_with_tools):
        """Test getting list of available tools."""
        tools = executor_with_tools.get_available_tools()

        assert "add" in tools
        assert "multiply" in tools
        assert "failing" in tools

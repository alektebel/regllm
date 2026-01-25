"""
Tests for Tool Registry
"""

import sys
from pathlib import Path

# Add project root to path to allow direct imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

# Direct import to avoid src/__init__.py heavy dependencies
from src.agents.tool_registry import Tool, ToolRegistry, tool, create_tool_from_function


class TestTool:
    """Tests for the Tool dataclass."""

    def test_tool_creation(self):
        """Test creating a Tool instance."""
        def sample_func(x: int) -> int:
            return x * 2

        t = Tool(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Input number"}
                },
                "required": ["x"],
            },
            function=sample_func,
            category="test",
        )

        assert t.name == "test_tool"
        assert t.description == "A test tool"
        assert t.category == "test"

    def test_tool_to_schema(self):
        """Test converting Tool to OpenAI schema format."""
        def sample_func():
            pass

        t = Tool(
            name="test_tool",
            description="Test description",
            parameters={"type": "object", "properties": {}},
            function=sample_func,
        )

        schema = t.to_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "Test description"

    def test_tool_to_prompt_description(self):
        """Test generating prompt description."""
        def sample_func():
            pass

        t = Tool(
            name="search",
            description="Search for documents",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results"},
                },
                "required": ["query"],
            },
            function=sample_func,
            examples=["search('IRB')", "search('capital', 5)"],
        )

        desc = t.to_prompt_description()

        assert "search" in desc
        assert "Search for documents" in desc
        assert "query*" in desc  # Required parameter marked
        assert "limit" in desc


class TestToolRegistry:
    """Tests for the ToolRegistry class."""

    def test_register_tool(self, tool_registry):
        """Test registering a tool."""
        def sample_func():
            return "result"

        t = Tool(
            name="sample",
            description="Sample tool",
            parameters={"type": "object", "properties": {}},
            function=sample_func,
            category="test",
        )

        tool_registry.register(t)

        assert "sample" in tool_registry
        assert len(tool_registry) == 1

    def test_get_tool(self, tool_registry):
        """Test retrieving a tool by name."""
        def sample_func():
            return "result"

        t = Tool(
            name="sample",
            description="Sample tool",
            parameters={"type": "object", "properties": {}},
            function=sample_func,
        )

        tool_registry.register(t)
        retrieved = tool_registry.get("sample")

        assert retrieved is not None
        assert retrieved.name == "sample"

    def test_get_nonexistent_tool(self, tool_registry):
        """Test retrieving a non-existent tool."""
        retrieved = tool_registry.get("nonexistent")
        assert retrieved is None

    def test_unregister_tool(self, tool_registry):
        """Test unregistering a tool."""
        def sample_func():
            pass

        t = Tool(
            name="to_remove",
            description="Will be removed",
            parameters={"type": "object", "properties": {}},
            function=sample_func,
        )

        tool_registry.register(t)
        assert "to_remove" in tool_registry

        result = tool_registry.unregister("to_remove")
        assert result is True
        assert "to_remove" not in tool_registry

    def test_list_tools(self, tool_registry):
        """Test listing all tools."""
        for i in range(3):
            def func():
                pass

            t = Tool(
                name=f"tool_{i}",
                description=f"Tool {i}",
                parameters={"type": "object", "properties": {}},
                function=func,
            )
            tool_registry.register(t)

        tools = tool_registry.list_tools()
        assert len(tools) == 3

    def test_list_tools_by_category(self, tool_registry):
        """Test listing tools filtered by category."""
        categories = ["search", "search", "calculation"]

        for i, cat in enumerate(categories):
            def func():
                pass

            t = Tool(
                name=f"tool_{i}",
                description=f"Tool {i}",
                parameters={"type": "object", "properties": {}},
                function=func,
                category=cat,
            )
            tool_registry.register(t)

        search_tools = tool_registry.list_tools("search")
        calc_tools = tool_registry.list_tools("calculation")

        assert len(search_tools) == 2
        assert len(calc_tools) == 1

    def test_execute_tool(self, tool_registry):
        """Test executing a tool."""
        def add_numbers(a: int, b: int) -> int:
            return a + b

        t = Tool(
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
            function=add_numbers,
        )

        tool_registry.register(t)
        result = tool_registry.execute("add", a=5, b=3)

        assert result == 8

    def test_execute_unknown_tool(self, tool_registry):
        """Test executing an unknown tool raises error."""
        with pytest.raises(ValueError, match="Unknown tool"):
            tool_registry.execute("nonexistent")

    def test_get_tools_schema(self, tool_registry):
        """Test getting OpenAI-compatible schema for all tools."""
        def func1():
            pass

        def func2():
            pass

        for name, func in [("tool1", func1), ("tool2", func2)]:
            t = Tool(
                name=name,
                description=f"{name} description",
                parameters={"type": "object", "properties": {}},
                function=func,
            )
            tool_registry.register(t)

        schemas = tool_registry.get_tools_schema()

        assert len(schemas) == 2
        assert all(s["type"] == "function" for s in schemas)


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_tool_decorator_adds_metadata(self):
        """Test that decorator adds metadata to function."""
        @tool(
            name="decorated_func",
            description="A decorated function",
            parameters={"type": "object", "properties": {}},
            category="decorated",
        )
        def my_func():
            return "hello"

        assert hasattr(my_func, "_tool_metadata")
        assert my_func._tool_metadata["name"] == "decorated_func"
        assert my_func._tool_metadata["category"] == "decorated"

    def test_create_tool_from_decorated_function(self):
        """Test creating Tool from decorated function."""
        @tool(
            name="from_decorator",
            description="Created from decorator",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        )
        def my_func(x: int) -> int:
            return x * 2

        t = create_tool_from_function(my_func)

        assert t.name == "from_decorator"
        assert t.description == "Created from decorator"
        assert t.function(5) == 10


class TestCreateToolFromFunction:
    """Tests for create_tool_from_function helper."""

    def test_create_with_explicit_params(self):
        """Test creating tool with explicit parameters."""
        def simple_func(x: int) -> int:
            """Double the input."""
            return x * 2

        t = create_tool_from_function(
            simple_func,
            name="doubler",
            description="Double a number",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            },
            category="math",
        )

        assert t.name == "doubler"
        assert t.description == "Double a number"
        assert t.category == "math"

    def test_create_with_inferred_params(self):
        """Test creating tool with inferred parameters."""
        def my_function():
            """This is the docstring."""
            pass

        t = create_tool_from_function(my_function)

        assert t.name == "my_function"
        assert "docstring" in t.description

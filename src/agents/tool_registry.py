#!/usr/bin/env python3
"""
Tool Registry for Agent Framework

Manages registration and discovery of tools available to the agent.
"""

from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Represents a tool that can be called by the agent."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema format
    function: Callable
    category: str = "general"
    examples: List[str] = field(default_factory=list)

    def to_schema(self) -> Dict:
        """Convert to OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def to_prompt_description(self) -> str:
        """Generate description for prompt injection."""
        params_desc = []
        props = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])

        for param_name, param_info in props.items():
            req_marker = "*" if param_name in required else ""
            param_type = param_info.get("type", "any")
            param_desc = param_info.get("description", "")
            params_desc.append(f"  - {param_name}{req_marker} ({param_type}): {param_desc}")

        params_text = "\n".join(params_desc) if params_desc else "  (sin parametros)"

        examples_text = ""
        if self.examples:
            examples_text = "\n  Ejemplos: " + ", ".join(self.examples[:2])

        return f"""**{self.name}**: {self.description}
  Parametros:
{params_text}{examples_text}
"""


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, tool: Tool):
        """Register a new tool."""
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool: {tool.name}")

        self._tools[tool.name] = tool

        # Track by category
        if tool.category not in self._categories:
            self._categories[tool.category] = []
        if tool.name not in self._categories[tool.category]:
            self._categories[tool.category].append(tool.name)

        logger.info(f"Registered tool: {tool.name} [{tool.category}]")

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name."""
        if name in self._tools:
            tool = self._tools[name]
            del self._tools[name]

            # Remove from category
            if tool.category in self._categories:
                self._categories[tool.category] = [
                    n for n in self._categories[tool.category] if n != name
                ]

            return True
        return False

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> List[Tool]:
        """List all tools, optionally filtered by category."""
        if category:
            names = self._categories.get(category, [])
            return [self._tools[n] for n in names]
        return list(self._tools.values())

    def list_categories(self) -> List[str]:
        """List all tool categories."""
        return list(self._categories.keys())

    def get_tools_schema(self) -> List[Dict]:
        """Get OpenAI-compatible tools schema for all tools."""
        return [tool.to_schema() for tool in self._tools.values()]

    def get_tools_prompt(self) -> str:
        """Generate tools description for prompt injection."""
        sections = []

        for category in sorted(self._categories.keys()):
            tools = self.list_tools(category)
            if not tools:
                continue

            section = f"### {category.title()}\n\n"
            for tool in tools:
                section += tool.to_prompt_description() + "\n"

            sections.append(section)

        return "\n".join(sections)

    def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name with given arguments."""
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")

        try:
            return tool.function(**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            raise

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    category: str = "general",
    examples: Optional[List[str]] = None,
):
    """Decorator to register a function as a tool."""
    def decorator(func: Callable) -> Callable:
        # Create tool but don't register yet
        func._tool_metadata = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "category": category,
            "examples": examples or [],
        }
        return func

    return decorator


def create_tool_from_function(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict] = None,
    category: str = "general",
    examples: Optional[List[str]] = None,
) -> Tool:
    """Create a Tool from a function, inferring metadata if possible."""
    # Check for decorator metadata
    if hasattr(func, '_tool_metadata'):
        meta = func._tool_metadata
        return Tool(
            name=meta["name"],
            description=meta["description"],
            parameters=meta["parameters"],
            function=func,
            category=meta["category"],
            examples=meta["examples"],
        )

    # Infer from function
    func_name = name or func.__name__
    func_desc = description or func.__doc__ or f"Execute {func_name}"

    # Default parameters if not provided
    if parameters is None:
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

    return Tool(
        name=func_name,
        description=func_desc,
        parameters=parameters,
        function=func,
        category=category,
        examples=examples or [],
    )


# Global registry instance
default_registry = ToolRegistry()


def main():
    """Demo of tool registry."""
    registry = ToolRegistry()

    # Register a sample tool
    def search_regulations(query: str, limit: int = 5) -> str:
        return f"Searching for: {query} (limit: {limit})"

    tool = Tool(
        name="search_regulations",
        description="Search the regulatory database for relevant documents",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query in Spanish or English"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 5
                }
            },
            "required": ["query"]
        },
        function=search_regulations,
        category="search",
        examples=["search_regulations('capital CET1')"]
    )

    registry.register(tool)

    print("Registered tools:")
    for t in registry.list_tools():
        print(f"  - {t.name}: {t.description}")

    print("\nTools prompt:")
    print(registry.get_tools_prompt())


if __name__ == "__main__":
    main()

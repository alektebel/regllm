#!/usr/bin/env python3
"""
Tool Registration Module

Registers all agent tools with the default registry.
"""

from typing import Optional

from ..tool_registry import (
    Tool,
    ToolRegistry,
    default_registry,
    create_tool_from_function,
)

from .methodology_tools import (
    read_methodology,
    parse_methodology_sections,
    extract_formulas,
    extract_parameters,
    compare_methodologies,
)

from .code_analysis_tools import (
    read_code_file,
    analyze_code_structure,
    extract_functions,
    extract_calculations,
    find_pattern_in_code,
)

from .consistency_tools import (
    check_formula_consistency,
    check_parameter_consistency,
    check_implementation_completeness,
    generate_consistency_report,
)


def register_all_tools(registry: Optional[ToolRegistry] = None) -> ToolRegistry:
    """Register all available tools with the registry.

    Args:
        registry: Registry to use. Defaults to global default_registry.

    Returns:
        The registry with all tools registered.
    """
    reg = registry if registry is not None else default_registry

    # Methodology tools
    methodology_tools = [
        read_methodology,
        parse_methodology_sections,
        extract_formulas,
        extract_parameters,
        compare_methodologies,
    ]

    # Code analysis tools
    code_analysis_tools = [
        read_code_file,
        analyze_code_structure,
        extract_functions,
        extract_calculations,
        find_pattern_in_code,
    ]

    # Consistency tools
    consistency_tools = [
        check_formula_consistency,
        check_parameter_consistency,
        check_implementation_completeness,
        generate_consistency_report,
    ]

    # Register all tools
    all_tools = methodology_tools + code_analysis_tools + consistency_tools

    for tool_func in all_tools:
        tool = create_tool_from_function(tool_func)
        reg.register(tool)

    return reg


def get_tools_by_category(
    registry: Optional[ToolRegistry] = None,
) -> dict:
    """Get all tools organized by category.

    Args:
        registry: Registry to use. Defaults to global default_registry.

    Returns:
        Dict mapping category names to lists of tools.
    """
    reg = registry or default_registry

    categories = {}
    for category in reg.list_categories():
        categories[category] = reg.list_tools(category)

    return categories


def print_registered_tools(registry: Optional[ToolRegistry] = None) -> None:
    """Print all registered tools.

    Args:
        registry: Registry to use.
    """
    reg = registry or default_registry

    print("=" * 60)
    print("Registered Tools")
    print("=" * 60)

    for category in sorted(reg.list_categories()):
        tools = reg.list_tools(category)
        print(f"\n## {category.title()} ({len(tools)} tools)")
        print("-" * 40)

        for tool in tools:
            print(f"\n  {tool.name}")
            print(f"    {tool.description}")

            # Show parameters
            props = tool.parameters.get("properties", {})
            required = tool.parameters.get("required", [])

            if props:
                print("    Parameters:")
                for param_name, param_info in props.items():
                    req_marker = "*" if param_name in required else ""
                    param_type = param_info.get("type", "any")
                    print(f"      - {param_name}{req_marker} ({param_type})")


def main():
    """Demo of tool registration."""
    # Create fresh registry
    registry = ToolRegistry()

    # Register all tools
    register_all_tools(registry)

    # Print registered tools
    print_registered_tools(registry)

    print("\n" + "=" * 60)
    print(f"Total tools registered: {len(registry)}")
    print("=" * 60)

    # Show tools prompt
    print("\nTools prompt (for LLM):")
    print("-" * 40)
    print(registry.get_tools_prompt())


if __name__ == "__main__":
    main()

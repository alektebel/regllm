"""
Agent Tools for RegLLM

This module provides tools for the regulation agent:
- Methodology document tools (reading, parsing)
- Code analysis tools
- Consistency checking tools
"""

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

from .registration import register_all_tools

__all__ = [
    # Methodology tools
    "read_methodology",
    "parse_methodology_sections",
    "extract_formulas",
    "extract_parameters",
    "compare_methodologies",
    # Code analysis tools
    "read_code_file",
    "analyze_code_structure",
    "extract_functions",
    "extract_calculations",
    "find_pattern_in_code",
    # Consistency tools
    "check_formula_consistency",
    "check_parameter_consistency",
    "check_implementation_completeness",
    "generate_consistency_report",
    # Registration
    "register_all_tools",
]

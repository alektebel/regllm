#!/usr/bin/env python3
"""
Demo: Methodology Consistency Checker

This script demonstrates how to use REGLLM to check if a code implementation
is consistent with a methodology document.

Usage:
    python demo_consistency_check.py [methodology_file] [code_file]

Example:
    python demo_consistency_check.py data/methodology/methodology_irb_foundation.md src/example_implementation.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.tools.methodology_tools import read_methodology, extract_formulas, extract_parameters
from src.agents.tools.code_analysis_tools import read_code_file, analyze_code_structure, extract_calculations
from src.agents.tools.consistency_tools import generate_consistency_report
from src.agents.tools.registration import register_all_tools
from src.agents.tool_registry import ToolRegistry
from src.agents.tool_executor import ToolExecutor


def print_section(title: str, char: str = "="):
    """Print a section header."""
    print()
    print(char * 60)
    print(title)
    print(char * 60)


def demo_with_sample_data():
    """Run demo with sample methodology and code."""
    print_section("REGLLM Methodology Consistency Checker Demo")

    # Sample methodology content (simulating a methodology document)
    methodology_content = """# Metodología IRB Fundación (F-IRB)

## Descripción General

El enfoque IRB Fundación es una metodología para el cálculo de requisitos de capital.

## Parámetros de Riesgo

| Parámetro | Fuente | Valor |
|-----------|--------|-------|
| **PD** (Probabilidad de Impago) | Estimación propia | Variable |
| **LGD** (Pérdida en caso de Impago) | Valores supervisores | 45% senior, 75% subordinado |
| **EAD** (Exposición en caso de Impago) | Valores supervisores | CCF regulatorios |

## Fórmula de RWA

Los activos ponderados por riesgo se calculan como:

```
RWA = K × 12.5 × EAD
```

Donde K es el requerimiento de capital:

```
K = LGD × f(PD, R) × MA
```

## Requisitos

- Mínimo 5 años de datos para estimación de PD
- Validación independiente anual
"""

    # Sample code implementation
    code_content = '''"""
IRB Foundation Implementation
"""

import numpy as np

# Supervisory LGD values
LGD_SENIOR = 0.45
LGD_SUBORDINATED = 0.75

# CCF values
CCF_COMMITTED = 0.75


def calculate_pd(historical_data: list, years: int = 5) -> float:
    """Calculate Probability of Default.

    Args:
        historical_data: Historical default data
        years: Minimum years required (default 5)

    Returns:
        Estimated PD
    """
    if years < 5:
        raise ValueError("Minimum 5 years required")

    defaults = sum(1 for d in historical_data if d.get("defaulted"))
    total = len(historical_data)
    return defaults / total if total > 0 else 0.03


def calculate_lgd(exposure_type: str = "senior") -> float:
    """Get supervisory LGD value."""
    if exposure_type == "senior":
        return LGD_SENIOR
    return LGD_SUBORDINATED


def calculate_ead(committed: float, drawn: float) -> float:
    """Calculate Exposure at Default."""
    undrawn = committed - drawn
    return drawn + (CCF_COMMITTED * undrawn)


def calculate_capital_requirement(pd: float, lgd: float, m: float = 2.5) -> float:
    """Calculate capital requirement K."""
    # Simplified - would use full Basel formula
    r = 0.12  # Asset correlation
    ma = 1 + (m - 2.5) * 0.05  # Maturity adjustment
    k = lgd * pd * ma
    return k


def calculate_rwa(ead: float, pd: float, lgd: float) -> float:
    """Calculate Risk-Weighted Assets.

    RWA = K × 12.5 × EAD
    """
    k = calculate_capital_requirement(pd, lgd)
    rwa = k * 12.5 * ead
    return rwa
'''

    # Step 1: Extract from methodology
    print_section("Step 1: Analyzing Methodology Document", "-")

    formulas = extract_formulas(methodology_content)
    print(f"Found {formulas['formula_count']} formulas:")
    for f in formulas['formulas'][:3]:
        print(f"  - {f['name']}: {f.get('expression', '')[:50]}")

    params = extract_parameters(methodology_content)
    print(f"\nFound {params['parameter_count']} parameters:")
    for p in params['parameters'][:5]:
        print(f"  - {p['name']}: {p.get('value', 'N/A')}")

    # Step 2: Analyze code
    print_section("Step 2: Analyzing Code Implementation", "-")

    structure = analyze_code_structure(code_content)
    print(f"Found {structure['structure']['function_count']} functions:")
    for f in structure['structure']['functions']:
        print(f"  - {f['name']}() at line {f['line']}")

    calcs = extract_calculations(code_content)
    print(f"\nFound {calcs['calculation_count']} calculations:")
    for c in calcs['calculations'][:5]:
        print(f"  - {c['variable']} = {c['expression'][:40]}...")

    # Step 3: Generate consistency report
    print_section("Step 3: Consistency Report", "-")

    report = generate_consistency_report(
        methodology_content=methodology_content,
        code_content=code_content,
        methodology_name="IRB Fundación",
        code_path="irb_implementation.py",
    )

    print(f"Overall Status: {report['overall_status']}")
    print(f"Message: {report['overall_message']}")

    print("\nStatistics:")
    stats = report['statistics']
    print(f"  - Total issues: {stats['total_issues']}")
    print(f"  - Errors: {stats['errors']}")
    print(f"  - Warnings: {stats['warnings']}")
    print(f"  - Completeness: {stats['completeness_percentage']}%")

    if report['all_issues']:
        print("\nIssues Found:")
        for issue in report['all_issues'][:5]:
            print(f"  [{issue['severity'].upper()}] {issue['message']}")
            if issue.get('suggestion'):
                print(f"    Suggestion: {issue['suggestion']}")

    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")

    # Step 4: Show how to use the agent
    print_section("Step 4: Using the Agent Framework", "-")

    # Register all tools
    registry = ToolRegistry()
    register_all_tools(registry)
    executor = ToolExecutor(registry)

    print(f"Registered {len(registry)} tools across categories:")
    for category in registry.list_categories():
        tools = registry.list_tools(category)
        print(f"  - {category}: {len(tools)} tools")

    print("\nExample tool execution:")
    result = executor.execute(
        "check_formula_consistency",
        {
            "methodology_content": methodology_content,
            "code_content": code_content,
        },
    )

    if result.is_success:
        summary = result.result["summary"]
        print(f"  Formula check: {summary['consistency_level']}")
        print(f"  Matched: {summary['formulas_matched']}/{summary['total_formulas']}")


def demo_with_files(methodology_path: str, code_path: str):
    """Run demo with actual files."""
    print_section("REGLLM Methodology Consistency Checker")

    # Read methodology
    print(f"\nReading methodology: {methodology_path}")
    meth_result = read_methodology(methodology_path)

    if not meth_result["success"]:
        print(f"Error: {meth_result['error']}")
        return

    print(f"  Title: {meth_result['title']}")
    print(f"  Lines: {meth_result['line_count']}")

    # Read code
    print(f"\nReading code: {code_path}")
    code_result = read_code_file(code_path)

    if not code_result["success"]:
        print(f"Error: {code_result['error']}")
        return

    print(f"  Language: {code_result['language']}")
    print(f"  Lines: {code_result['total_lines']}")

    # Generate report
    print_section("Consistency Report", "-")

    report = generate_consistency_report(
        methodology_content=meth_result["content"],
        code_content=code_result["content"],
        methodology_name=meth_result["title"],
        code_path=code_path,
    )

    print(f"\nOverall Status: {report['overall_status']}")
    print(f"Message: {report['overall_message']}")

    print("\nStatistics:")
    for key, value in report['statistics'].items():
        print(f"  - {key}: {value}")

    if report['all_issues']:
        print("\nIssues Found:")
        for issue in report['all_issues']:
            print(f"  [{issue['severity'].upper()}] {issue['message']}")

    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")


def main():
    """Main entry point."""
    if len(sys.argv) == 1:
        # Run with sample data
        demo_with_sample_data()
    elif len(sys.argv) == 3:
        # Run with provided files
        demo_with_files(sys.argv[1], sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

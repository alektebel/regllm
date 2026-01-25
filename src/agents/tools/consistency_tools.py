#!/usr/bin/env python3
"""
Consistency Checking Tools

Tools for verifying consistency between methodology documents and code implementations.
"""

import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from ..tool_registry import tool
from .methodology_tools import extract_formulas, extract_parameters
from .code_analysis_tools import extract_calculations, extract_functions


class ConsistencyLevel(Enum):
    """Level of consistency."""
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    WARNING = "warning"


@dataclass
class ConsistencyIssue:
    """A consistency issue found during verification."""
    severity: str  # "error", "warning", "info"
    category: str
    message: str
    methodology_ref: Optional[str]
    code_ref: Optional[str]
    suggestion: Optional[str]


@tool(
    name="check_formula_consistency",
    description="Verifica la consistencia entre fórmulas de metodología y su implementación en código",
    parameters={
        "type": "object",
        "properties": {
            "methodology_content": {
                "type": "string",
                "description": "Contenido del documento de metodología",
            },
            "code_content": {
                "type": "string",
                "description": "Contenido del archivo de código",
            },
            "formula_names": {
                "type": "array",
                "description": "Nombres específicos de fórmulas a verificar (opcional)",
                "items": {"type": "string"},
            },
        },
        "required": ["methodology_content", "code_content"],
    },
    category="consistency",
)
def check_formula_consistency(
    methodology_content: str,
    code_content: str,
    formula_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Check consistency between methodology formulas and code implementation.

    Args:
        methodology_content: Methodology document content.
        code_content: Code file content.
        formula_names: Optional specific formulas to check.

    Returns:
        Consistency check results.
    """
    # Extract from methodology
    meth_formulas = extract_formulas(methodology_content)["formulas"]

    # Extract from code
    code_calcs = extract_calculations(code_content)["calculations"]

    issues = []
    matches = []

    # Map common formula names to code patterns
    formula_code_mapping = {
        "RWA": ["rwa", "risk_weighted_assets", "rwa_"],
        "K": ["k_", "capital_req", "capital_requirement"],
        "PD": ["pd", "prob_default", "probability_default"],
        "LGD": ["lgd", "loss_given_default"],
        "EAD": ["ead", "exposure_at_default", "exposure"],
        "CCF": ["ccf", "credit_conversion", "conversion_factor"],
        "MA": ["maturity_adj", "ma_", "adjustment"],
    }

    # Filter formulas if specific names provided
    if formula_names:
        meth_formulas = [
            f for f in meth_formulas
            if f["name"] in formula_names
        ]

    for formula in meth_formulas:
        formula_name = formula["name"]
        formula_expr = formula.get("expression", "")

        # Find corresponding code calculation
        code_patterns = formula_code_mapping.get(
            formula_name.upper(),
            [formula_name.lower()]
        )

        matching_calcs = []
        for calc in code_calcs:
            calc_var = calc["variable"].lower()
            if any(pat in calc_var for pat in code_patterns):
                matching_calcs.append(calc)

        if not matching_calcs:
            issues.append({
                "severity": "error",
                "category": "missing_implementation",
                "message": f"Fórmula '{formula_name}' no encontrada en el código",
                "methodology_ref": f"Línea {formula.get('line', '?')}: {formula_expr[:50]}",
                "code_ref": None,
                "suggestion": f"Implementar cálculo de {formula_name}",
            })
        else:
            # Compare expressions
            for calc in matching_calcs:
                # Extract variables from both
                meth_vars = set(formula.get("variables", []))
                code_vars = set(calc.get("used_variables", []))

                # Check for common variables
                common_vars = meth_vars & code_vars
                missing_vars = meth_vars - code_vars
                extra_vars = code_vars - meth_vars

                match_info = {
                    "formula_name": formula_name,
                    "methodology_line": formula.get("line"),
                    "methodology_expression": formula_expr,
                    "code_line": calc["line"],
                    "code_expression": calc["expression"],
                    "common_variables": list(common_vars),
                    "consistency": "partial" if missing_vars or extra_vars else "full",
                }

                if missing_vars:
                    issues.append({
                        "severity": "warning",
                        "category": "missing_variables",
                        "message": f"Variables de metodología no encontradas en código: {missing_vars}",
                        "methodology_ref": f"{formula_name} línea {formula.get('line', '?')}",
                        "code_ref": f"Línea {calc['line']}: {calc['expression'][:50]}",
                        "suggestion": f"Verificar que {missing_vars} estén incluidas en el cálculo",
                    })

                matches.append(match_info)

    # Summary
    total_formulas = len(meth_formulas)
    matched = len(matches)
    errors = len([i for i in issues if i["severity"] == "error"])
    warnings = len([i for i in issues if i["severity"] == "warning"])

    return {
        "success": True,
        "summary": {
            "total_formulas": total_formulas,
            "formulas_matched": matched,
            "errors": errors,
            "warnings": warnings,
            "consistency_level": (
                "consistent" if errors == 0 and warnings == 0
                else "partial" if errors == 0
                else "inconsistent"
            ),
        },
        "matches": matches,
        "issues": issues,
    }


@tool(
    name="check_parameter_consistency",
    description="Verifica la consistencia entre parámetros de metodología y código",
    parameters={
        "type": "object",
        "properties": {
            "methodology_content": {
                "type": "string",
                "description": "Contenido del documento de metodología",
            },
            "code_content": {
                "type": "string",
                "description": "Contenido del archivo de código",
            },
        },
        "required": ["methodology_content", "code_content"],
    },
    category="consistency",
)
def check_parameter_consistency(
    methodology_content: str,
    code_content: str,
) -> Dict[str, Any]:
    """Check consistency of parameters between methodology and code.

    Args:
        methodology_content: Methodology document content.
        code_content: Code file content.

    Returns:
        Parameter consistency check results.
    """
    # Extract methodology parameters
    meth_params = extract_parameters(methodology_content)["parameters"]

    # Look for parameter definitions in code
    issues = []
    matches = []

    # Patterns for finding parameter values in code
    code_lines = code_content.split("\n")

    # Common parameter patterns in code
    param_patterns = {
        "LGD": [
            r"lgd\s*=\s*([\d.]+)",
            r"loss_given_default\s*=\s*([\d.]+)",
            r"LGD_SENIOR\s*=\s*([\d.]+)",
            r"LGD_SUBORDINATED\s*=\s*([\d.]+)",
        ],
        "CCF": [
            r"ccf\s*=\s*([\d.]+)",
            r"credit_conversion_factor\s*=\s*([\d.]+)",
            r"CCF_\w+\s*=\s*([\d.]+)",
        ],
        "PD": [
            r"pd\s*=\s*([\d.]+)",
            r"prob_default\s*=\s*([\d.]+)",
            r"PD_FLOOR\s*=\s*([\d.]+)",
        ],
    }

    # Find code parameters
    code_params = []
    for i, line in enumerate(code_lines):
        for param_name, patterns in param_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    code_params.append({
                        "name": param_name,
                        "value": value,
                        "line": i + 1,
                        "full_match": match.group(0),
                    })

    # Compare methodology params with code params
    for meth_param in meth_params:
        param_name = meth_param["name"]
        meth_value = meth_param.get("value", "")

        # Find matching code params
        matching_code = [
            cp for cp in code_params
            if cp["name"].upper() == param_name.upper()
        ]

        if not matching_code:
            issues.append({
                "severity": "warning",
                "category": "parameter_not_found",
                "message": f"Parámetro '{param_name}' de metodología no encontrado en código",
                "methodology_ref": f"Línea {meth_param.get('line', '?')}: {meth_value}",
                "code_ref": None,
                "suggestion": f"Definir constante para {param_name}",
            })
            continue

        # Compare values if both have numeric values
        for code_param in matching_code:
            try:
                # Extract numeric value from methodology
                meth_num = None
                if meth_value:
                    meth_match = re.search(r"([\d.]+)%?", str(meth_value))
                    if meth_match:
                        meth_num = float(meth_match.group(1))
                        if "%" in str(meth_value):
                            meth_num = meth_num / 100

                # Extract numeric value from code
                code_num = float(code_param["value"])

                # Compare
                if meth_num is not None:
                    # Allow small tolerance for floating point
                    if abs(meth_num - code_num) < 0.001:
                        matches.append({
                            "parameter": param_name,
                            "methodology_value": meth_value,
                            "code_value": code_param["value"],
                            "methodology_line": meth_param.get("line"),
                            "code_line": code_param["line"],
                            "status": "consistent",
                        })
                    else:
                        issues.append({
                            "severity": "error",
                            "category": "value_mismatch",
                            "message": f"Valor de {param_name} no coincide: metodología={meth_value}, código={code_param['value']}",
                            "methodology_ref": f"Línea {meth_param.get('line', '?')}",
                            "code_ref": f"Línea {code_param['line']}",
                            "suggestion": f"Actualizar código para usar {meth_value}",
                        })
                        matches.append({
                            "parameter": param_name,
                            "methodology_value": meth_value,
                            "code_value": code_param["value"],
                            "status": "inconsistent",
                        })

            except (ValueError, TypeError):
                # Non-numeric comparison
                matches.append({
                    "parameter": param_name,
                    "methodology_value": meth_value,
                    "code_value": code_param["value"],
                    "status": "review_needed",
                })

    # Summary
    errors = len([i for i in issues if i["severity"] == "error"])
    warnings = len([i for i in issues if i["severity"] == "warning"])
    consistent = len([m for m in matches if m.get("status") == "consistent"])

    return {
        "success": True,
        "summary": {
            "methodology_parameters": len(meth_params),
            "code_parameters": len(code_params),
            "consistent": consistent,
            "errors": errors,
            "warnings": warnings,
        },
        "matches": matches,
        "issues": issues,
    }


@tool(
    name="check_implementation_completeness",
    description="Verifica que todas las funciones requeridas por la metodología estén implementadas",
    parameters={
        "type": "object",
        "properties": {
            "methodology_content": {
                "type": "string",
                "description": "Contenido del documento de metodología",
            },
            "code_content": {
                "type": "string",
                "description": "Contenido del archivo de código",
            },
            "required_components": {
                "type": "array",
                "description": "Lista de componentes requeridos a verificar",
                "items": {"type": "string"},
            },
        },
        "required": ["methodology_content", "code_content"],
    },
    category="consistency",
)
def check_implementation_completeness(
    methodology_content: str,
    code_content: str,
    required_components: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Check that all methodology requirements are implemented.

    Args:
        methodology_content: Methodology document content.
        code_content: Code file content.
        required_components: Optional list of required components.

    Returns:
        Completeness check results.
    """
    # Default required components for banking regulation
    if not required_components:
        required_components = [
            "PD calculation",
            "LGD calculation",
            "EAD calculation",
            "RWA calculation",
            "Capital requirement",
            "Maturity adjustment",
            "Asset correlation",
        ]

    # Extract code functions
    code_funcs = extract_functions(code_content)["functions"]
    func_names = [f["name"].lower() for f in code_funcs]

    # Component to function mapping
    component_patterns = {
        "PD calculation": ["pd", "probability", "default_prob"],
        "LGD calculation": ["lgd", "loss_given"],
        "EAD calculation": ["ead", "exposure"],
        "RWA calculation": ["rwa", "risk_weight"],
        "Capital requirement": ["capital", "k_calc", "requirement"],
        "Maturity adjustment": ["maturity", "ma_", "adjustment"],
        "Asset correlation": ["correlation", "rho", "asset_corr"],
    }

    results = []
    missing = []
    found = []

    for component in required_components:
        patterns = component_patterns.get(component, [component.lower().split()[0]])

        # Check if any pattern matches a function name
        matched = False
        matching_funcs = []

        for func_name in func_names:
            if any(pat in func_name for pat in patterns):
                matched = True
                matching_funcs.append(func_name)

        if matched:
            found.append({
                "component": component,
                "status": "implemented",
                "matching_functions": matching_funcs,
            })
        else:
            missing.append({
                "component": component,
                "status": "not_found",
                "expected_patterns": patterns,
            })

    # Check for methodology requirements in document
    methodology_reqs = []
    req_patterns = [
        r"requisito[s]?\s*:?\s*(.+)",
        r"debe[n]?\s+(.+)",
        r"obligatori[oa]\s*:?\s*(.+)",
        r"necesari[oa]\s*:?\s*(.+)",
    ]

    for pattern in req_patterns:
        for match in re.finditer(pattern, methodology_content, re.IGNORECASE):
            methodology_reqs.append({
                "text": match.group(1)[:100],
                "pattern": pattern,
            })

    # Summary
    total = len(required_components)
    implemented = len(found)
    completeness_pct = (implemented / total * 100) if total > 0 else 0

    return {
        "success": True,
        "summary": {
            "total_components": total,
            "implemented": implemented,
            "missing": len(missing),
            "completeness_percentage": round(completeness_pct, 1),
        },
        "implemented": found,
        "missing": missing,
        "methodology_requirements": methodology_reqs[:10],  # Limit to first 10
        "code_functions": func_names,
    }


@tool(
    name="generate_consistency_report",
    description="Genera un reporte completo de consistencia entre metodología y código",
    parameters={
        "type": "object",
        "properties": {
            "methodology_content": {
                "type": "string",
                "description": "Contenido del documento de metodología",
            },
            "code_content": {
                "type": "string",
                "description": "Contenido del archivo de código",
            },
            "methodology_name": {
                "type": "string",
                "description": "Nombre de la metodología",
            },
            "code_path": {
                "type": "string",
                "description": "Ruta del archivo de código",
            },
        },
        "required": ["methodology_content", "code_content"],
    },
    category="consistency",
)
def generate_consistency_report(
    methodology_content: str,
    code_content: str,
    methodology_name: Optional[str] = None,
    code_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive consistency report.

    Args:
        methodology_content: Methodology document content.
        code_content: Code file content.
        methodology_name: Optional methodology name.
        code_path: Optional code file path.

    Returns:
        Complete consistency report.
    """
    # Run all consistency checks
    formula_check = check_formula_consistency(methodology_content, code_content)
    param_check = check_parameter_consistency(methodology_content, code_content)
    completeness_check = check_implementation_completeness(methodology_content, code_content)

    # Aggregate all issues
    all_issues = (
        formula_check["issues"] +
        param_check["issues"]
    )

    # Count by severity
    errors = len([i for i in all_issues if i["severity"] == "error"])
    warnings = len([i for i in all_issues if i["severity"] == "warning"])
    infos = len([i for i in all_issues if i["severity"] == "info"])

    # Determine overall status
    if errors > 0:
        overall_status = "INCONSISTENT"
        overall_message = f"Se encontraron {errors} errores de consistencia"
    elif warnings > 0:
        overall_status = "PARTIAL"
        overall_message = f"Parcialmente consistente con {warnings} advertencias"
    else:
        overall_status = "CONSISTENT"
        overall_message = "La implementación es consistente con la metodología"

    # Build report
    report = {
        "success": True,
        "metadata": {
            "methodology_name": methodology_name or "No especificada",
            "code_path": code_path or "No especificada",
            "generated_at": None,  # Will be filled by caller
        },
        "overall_status": overall_status,
        "overall_message": overall_message,
        "statistics": {
            "total_issues": len(all_issues),
            "errors": errors,
            "warnings": warnings,
            "info": infos,
            "formula_consistency": formula_check["summary"]["consistency_level"],
            "completeness_percentage": completeness_check["summary"]["completeness_percentage"],
        },
        "formula_analysis": {
            "total_formulas": formula_check["summary"]["total_formulas"],
            "matched": formula_check["summary"]["formulas_matched"],
            "matches": formula_check["matches"],
        },
        "parameter_analysis": {
            "methodology_params": param_check["summary"]["methodology_parameters"],
            "code_params": param_check["summary"]["code_parameters"],
            "consistent": param_check["summary"]["consistent"],
            "matches": param_check["matches"],
        },
        "completeness_analysis": {
            "total_components": completeness_check["summary"]["total_components"],
            "implemented": completeness_check["summary"]["implemented"],
            "missing": completeness_check["missing"],
        },
        "all_issues": all_issues,
        "recommendations": _generate_recommendations(all_issues, completeness_check["missing"]),
    }

    return report


def _generate_recommendations(
    issues: List[Dict],
    missing_components: List[Dict],
) -> List[str]:
    """Generate recommendations based on issues found."""
    recommendations = []

    # Group issues by category
    categories = {}
    for issue in issues:
        cat = issue.get("category", "other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(issue)

    # Generate recommendations per category
    if "missing_implementation" in categories:
        count = len(categories["missing_implementation"])
        recommendations.append(
            f"Implementar {count} fórmula(s) faltante(s) definida(s) en la metodología"
        )

    if "value_mismatch" in categories:
        count = len(categories["value_mismatch"])
        recommendations.append(
            f"Revisar y corregir {count} valor(es) de parámetro(s) que no coinciden"
        )

    if "missing_variables" in categories:
        recommendations.append(
            "Verificar que todas las variables de las fórmulas estén incluidas en los cálculos"
        )

    if missing_components:
        comp_names = [c["component"] for c in missing_components[:3]]
        recommendations.append(
            f"Implementar componentes faltantes: {', '.join(comp_names)}"
        )

    if not recommendations:
        recommendations.append(
            "La implementación parece correcta. Realizar pruebas unitarias para validar."
        )

    return recommendations

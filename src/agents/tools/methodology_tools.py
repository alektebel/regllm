#!/usr/bin/env python3
"""
Methodology Document Tools

Tools for reading, parsing, and analyzing methodology documents.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from ..tool_registry import tool


@dataclass
class MethodologySection:
    """A section from a methodology document."""
    title: str
    level: int
    content: str
    subsections: List["MethodologySection"] = field(default_factory=list)


@dataclass
class Formula:
    """A formula extracted from methodology."""
    name: str
    expression: str
    variables: List[str]
    context: str
    source_line: int


@dataclass
class Parameter:
    """A parameter definition from methodology."""
    name: str
    description: str
    value: Optional[str]
    unit: Optional[str]
    source: str


@tool(
    name="read_methodology",
    description="Lee un documento de metodología desde un archivo",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Ruta al archivo de metodología (.md o .txt)",
            }
        },
        "required": ["file_path"],
    },
    category="methodology",
    examples=["read_methodology('data/methodology/methodology_irb_foundation.md')"],
)
def read_methodology(file_path: str) -> Dict[str, Any]:
    """Read a methodology document from file.

    Args:
        file_path: Path to the methodology file.

    Returns:
        Dict with document content and metadata.
    """
    path = Path(file_path)

    if not path.exists():
        return {
            "success": False,
            "error": f"Archivo no encontrado: {file_path}",
        }

    try:
        content = path.read_text(encoding="utf-8")

        # Extract title from first heading
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1) if title_match else path.stem

        return {
            "success": True,
            "file_path": str(path.absolute()),
            "title": title,
            "content": content,
            "line_count": len(content.splitlines()),
            "char_count": len(content),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error leyendo archivo: {str(e)}",
        }


@tool(
    name="parse_methodology_sections",
    description="Parsea un documento de metodología en secciones estructuradas",
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Contenido del documento de metodología en formato Markdown",
            }
        },
        "required": ["content"],
    },
    category="methodology",
)
def parse_methodology_sections(content: str) -> Dict[str, Any]:
    """Parse methodology document into structured sections.

    Args:
        content: Markdown content of the methodology document.

    Returns:
        Dict with parsed sections hierarchy.
    """
    sections = []
    current_section = None
    current_content = []

    lines = content.split("\n")

    for i, line in enumerate(lines):
        # Check for headings
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)

        if heading_match:
            # Save previous section
            if current_section:
                current_section["content"] = "\n".join(current_content).strip()
                sections.append(current_section)

            level = len(heading_match.group(1))
            title = heading_match.group(2)

            current_section = {
                "title": title,
                "level": level,
                "start_line": i + 1,
                "content": "",
            }
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_section:
        current_section["content"] = "\n".join(current_content).strip()
        sections.append(current_section)

    return {
        "success": True,
        "section_count": len(sections),
        "sections": sections,
    }


@tool(
    name="extract_formulas",
    description="Extrae fórmulas matemáticas de un documento de metodología",
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Contenido del documento de metodología",
            }
        },
        "required": ["content"],
    },
    category="methodology",
)
def extract_formulas(content: str) -> Dict[str, Any]:
    """Extract mathematical formulas from methodology document.

    Args:
        content: Methodology document content.

    Returns:
        Dict with extracted formulas and their context.
    """
    formulas = []
    lines = content.split("\n")

    # Pattern for code blocks (which often contain formulas)
    code_block_pattern = re.compile(r"```[\s\S]*?```")

    # Pattern for inline formulas
    inline_formula_pattern = re.compile(r"([A-Z_]+)\s*=\s*([^,\n]+)")

    # Find code blocks
    for match in code_block_pattern.finditer(content):
        block_content = match.group()
        block_start = content[:match.start()].count("\n") + 1

        # Extract formulas from code block
        for line in block_content.split("\n"):
            formula_match = inline_formula_pattern.search(line)
            if formula_match:
                name = formula_match.group(1)
                expression = formula_match.group(2).strip()

                # Extract variables (uppercase letters or words)
                variables = re.findall(r"\b([A-Z][A-Za-z_]*)\b", expression)
                variables = list(set(variables) - {name})

                formulas.append({
                    "name": name,
                    "expression": expression,
                    "variables": variables,
                    "in_code_block": True,
                    "line": block_start,
                })

    # Find inline formulas in text
    for i, line in enumerate(lines):
        if "```" in line:
            continue  # Skip code block markers

        formula_match = inline_formula_pattern.search(line)
        if formula_match:
            name = formula_match.group(1)
            expression = formula_match.group(2).strip()

            # Skip if already found in code block
            if any(f["name"] == name for f in formulas):
                continue

            variables = re.findall(r"\b([A-Z][A-Za-z_]*)\b", expression)
            variables = list(set(variables) - {name})

            formulas.append({
                "name": name,
                "expression": expression,
                "variables": variables,
                "in_code_block": False,
                "line": i + 1,
            })

    # Look for common banking regulation formulas
    banking_patterns = [
        (r"RWA\s*=", "RWA"),
        (r"K\s*=", "Capital Requirement"),
        (r"PD\s*[×\*]", "PD calculation"),
        (r"LGD\s*[×\*]", "LGD calculation"),
        (r"EAD\s*[×\*]", "EAD calculation"),
    ]

    for pattern, formula_type in banking_patterns:
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                # Get context (line before and after)
                context_start = max(0, i - 1)
                context_end = min(len(lines), i + 2)
                context = "\n".join(lines[context_start:context_end])

                # Check if not already captured
                existing = any(
                    f["line"] == i + 1 and f["name"] in line
                    for f in formulas
                )
                if not existing:
                    formulas.append({
                        "name": formula_type,
                        "expression": line.strip(),
                        "variables": [],
                        "in_code_block": False,
                        "line": i + 1,
                        "context": context,
                    })

    return {
        "success": True,
        "formula_count": len(formulas),
        "formulas": formulas,
    }


@tool(
    name="extract_parameters",
    description="Extrae definiciones de parámetros de un documento de metodología",
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Contenido del documento de metodología",
            }
        },
        "required": ["content"],
    },
    category="methodology",
)
def extract_parameters(content: str) -> Dict[str, Any]:
    """Extract parameter definitions from methodology document.

    Args:
        content: Methodology document content.

    Returns:
        Dict with extracted parameters.
    """
    parameters = []

    # Common banking risk parameters
    param_patterns = {
        "PD": r"(?:PD|Probabilidad de (?:Impago|Default))",
        "LGD": r"(?:LGD|Pérdida (?:en caso de|dado el) [Ii]mpago)",
        "EAD": r"(?:EAD|Exposición (?:en caso de|al) [Ii]mpago)",
        "M": r"(?:M|Vencimiento|Maturity)",
        "CCF": r"(?:CCF|Factor de Conversión)",
        "RWA": r"(?:RWA|Activos Ponderados por Riesgo)",
    }

    lines = content.split("\n")

    # Extract from tables
    table_pattern = re.compile(r"\|\s*\*?\*?([^|]+)\*?\*?\s*\|\s*([^|]+)\s*\|")

    for i, line in enumerate(lines):
        table_match = table_pattern.search(line)
        if table_match:
            param_name = table_match.group(1).strip()
            param_value = table_match.group(2).strip()

            # Check if it's a parameter row
            for key, pattern in param_patterns.items():
                if re.search(pattern, param_name, re.IGNORECASE):
                    parameters.append({
                        "name": key,
                        "full_name": param_name.replace("**", ""),
                        "value": param_value,
                        "source_type": "table",
                        "line": i + 1,
                    })
                    break

    # Extract from text definitions
    definition_patterns = [
        r"(?P<param>PD|LGD|EAD|CCF|RWA|M)\s*[:\-]\s*(?P<desc>[^.]+)",
        r"\*\*(?P<param>[A-Z]+)\*\*\s*\((?P<desc>[^)]+)\)",
        r"(?P<param>[A-Z]{2,})\s*=\s*(?P<value>[\d.%]+)",
    ]

    for pattern in definition_patterns:
        for match in re.finditer(pattern, content):
            param_data = match.groupdict()
            param_name = param_data.get("param", "")

            # Skip if already found
            if any(p["name"] == param_name and p.get("value") for p in parameters):
                continue

            line_num = content[:match.start()].count("\n") + 1

            parameters.append({
                "name": param_name,
                "description": param_data.get("desc", ""),
                "value": param_data.get("value"),
                "source_type": "text",
                "line": line_num,
            })

    # Extract percentage values
    percentage_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*%")
    for i, line in enumerate(lines):
        for key, pattern in param_patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                percentages = percentage_pattern.findall(line)
                for pct in percentages:
                    # Check if not already captured
                    exists = any(
                        p["name"] == key and p.get("value") == f"{pct}%"
                        for p in parameters
                    )
                    if not exists:
                        parameters.append({
                            "name": key,
                            "value": f"{pct}%",
                            "context": line.strip(),
                            "source_type": "inline",
                            "line": i + 1,
                        })

    return {
        "success": True,
        "parameter_count": len(parameters),
        "parameters": parameters,
    }


@tool(
    name="compare_methodologies",
    description="Compara dos documentos de metodología y genera un resumen de diferencias",
    parameters={
        "type": "object",
        "properties": {
            "methodology_a": {
                "type": "string",
                "description": "Contenido del primer documento de metodología",
            },
            "methodology_b": {
                "type": "string",
                "description": "Contenido del segundo documento de metodología",
            },
            "aspects": {
                "type": "array",
                "description": "Aspectos específicos a comparar",
                "items": {"type": "string"},
            },
        },
        "required": ["methodology_a", "methodology_b"],
    },
    category="methodology",
)
def compare_methodologies(
    methodology_a: str,
    methodology_b: str,
    aspects: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compare two methodology documents.

    Args:
        methodology_a: Content of first methodology.
        methodology_b: Content of second methodology.
        aspects: Specific aspects to compare.

    Returns:
        Comparison results.
    """
    # Extract parameters from both
    params_a = extract_parameters(methodology_a)["parameters"]
    params_b = extract_parameters(methodology_b)["parameters"]

    # Extract formulas from both
    formulas_a = extract_formulas(methodology_a)["formulas"]
    formulas_b = extract_formulas(methodology_b)["formulas"]

    # Compare parameters
    param_comparison = []
    all_params = set(p["name"] for p in params_a) | set(p["name"] for p in params_b)

    for param_name in all_params:
        a_vals = [p for p in params_a if p["name"] == param_name]
        b_vals = [p for p in params_b if p["name"] == param_name]

        param_comparison.append({
            "parameter": param_name,
            "in_methodology_a": len(a_vals) > 0,
            "in_methodology_b": len(b_vals) > 0,
            "values_a": [p.get("value") for p in a_vals],
            "values_b": [p.get("value") for p in b_vals],
            "different": (
                [p.get("value") for p in a_vals] != [p.get("value") for p in b_vals]
            ),
        })

    # Compare formulas
    formula_comparison = []
    all_formulas = set(f["name"] for f in formulas_a) | set(f["name"] for f in formulas_b)

    for formula_name in all_formulas:
        a_forms = [f for f in formulas_a if f["name"] == formula_name]
        b_forms = [f for f in formulas_b if f["name"] == formula_name]

        formula_comparison.append({
            "formula": formula_name,
            "in_methodology_a": len(a_forms) > 0,
            "in_methodology_b": len(b_forms) > 0,
            "expressions_a": [f.get("expression") for f in a_forms],
            "expressions_b": [f.get("expression") for f in b_forms],
        })

    # Summary
    differences = [p for p in param_comparison if p["different"]]
    missing_in_a = [p for p in param_comparison if not p["in_methodology_a"]]
    missing_in_b = [p for p in param_comparison if not p["in_methodology_b"]]

    return {
        "success": True,
        "parameter_comparison": param_comparison,
        "formula_comparison": formula_comparison,
        "summary": {
            "total_parameters": len(all_params),
            "different_values": len(differences),
            "missing_in_a": len(missing_in_a),
            "missing_in_b": len(missing_in_b),
            "total_formulas": len(all_formulas),
        },
        "differences": differences,
    }

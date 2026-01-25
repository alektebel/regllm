#!/usr/bin/env python3
"""
Code Analysis Tools

Tools for reading, parsing, and analyzing code implementations.
"""

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from ..tool_registry import tool


@dataclass
class FunctionInfo:
    """Information about a function in code."""
    name: str
    start_line: int
    end_line: int
    args: List[str]
    docstring: Optional[str]
    returns: Optional[str]
    body_preview: str


@dataclass
class CalculationInfo:
    """Information about a calculation in code."""
    variable: str
    expression: str
    line: int
    context: str


@tool(
    name="read_code_file",
    description="Lee un archivo de código fuente",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Ruta al archivo de código (.py, .js, etc.)",
            },
            "start_line": {
                "type": "integer",
                "description": "Línea inicial (opcional)",
            },
            "end_line": {
                "type": "integer",
                "description": "Línea final (opcional)",
            },
        },
        "required": ["file_path"],
    },
    category="code_analysis",
    examples=["read_code_file('src/training/model_setup.py')"],
)
def read_code_file(
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> Dict[str, Any]:
    """Read a source code file.

    Args:
        file_path: Path to the code file.
        start_line: Optional starting line number.
        end_line: Optional ending line number.

    Returns:
        Dict with file content and metadata.
    """
    path = Path(file_path)

    if not path.exists():
        return {
            "success": False,
            "error": f"Archivo no encontrado: {file_path}",
        }

    try:
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")
        total_lines = len(lines)

        # Apply line range if specified
        if start_line or end_line:
            start_idx = (start_line - 1) if start_line else 0
            end_idx = end_line if end_line else total_lines
            lines = lines[start_idx:end_idx]
            content = "\n".join(lines)

        # Detect language
        language = _detect_language(path.suffix)

        return {
            "success": True,
            "file_path": str(path.absolute()),
            "language": language,
            "content": content,
            "total_lines": total_lines,
            "lines_returned": len(lines),
            "start_line": start_line or 1,
            "end_line": end_line or total_lines,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error leyendo archivo: {str(e)}",
        }


def _detect_language(suffix: str) -> str:
    """Detect programming language from file extension."""
    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".rs": "rust",
        ".go": "go",
        ".rb": "ruby",
        ".r": "r",
    }
    return language_map.get(suffix.lower(), "unknown")


@tool(
    name="analyze_code_structure",
    description="Analiza la estructura de un archivo de código Python",
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Contenido del archivo de código Python",
            },
            "file_path": {
                "type": "string",
                "description": "Ruta del archivo (para contexto)",
            },
        },
        "required": ["content"],
    },
    category="code_analysis",
)
def analyze_code_structure(
    content: str,
    file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze the structure of a Python code file.

    Args:
        content: Python source code content.
        file_path: Optional file path for context.

    Returns:
        Dict with code structure analysis.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {
            "success": False,
            "error": f"Error de sintaxis en el código: {e}",
        }

    classes = []
    functions = []
    imports = []
    global_vars = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "methods": methods,
                "method_count": len(methods),
                "docstring": ast.get_docstring(node),
            })

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip methods (already captured in classes)
            is_method = False
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    if hasattr(parent, "body") and isinstance(parent.body, list):
                        if node in parent.body:
                            is_method = True
                            break
            if is_method:
                continue

            args = [arg.arg for arg in node.args.args]
            functions.append({
                "name": node.name,
                "line": node.lineno,
                "end_line": node.end_lineno,
                "args": args,
                "docstring": ast.get_docstring(node),
                "is_async": isinstance(node, ast.AsyncFunctionDef),
            })

        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "module": alias.name,
                    "alias": alias.asname,
                    "line": node.lineno,
                })

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append({
                    "module": f"{module}.{alias.name}" if module else alias.name,
                    "from_module": module,
                    "alias": alias.asname,
                    "line": node.lineno,
                })

        elif isinstance(node, ast.Assign) and isinstance(node, ast.Assign):
            # Top-level assignments
            if node.col_offset == 0:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        global_vars.append({
                            "name": target.id,
                            "line": node.lineno,
                        })

    return {
        "success": True,
        "file_path": file_path,
        "structure": {
            "classes": classes,
            "class_count": len(classes),
            "functions": functions,
            "function_count": len(functions),
            "imports": imports,
            "import_count": len(imports),
            "global_variables": global_vars,
        },
        "total_lines": len(content.split("\n")),
    }


@tool(
    name="extract_functions",
    description="Extrae funciones de un archivo de código Python con sus cuerpos completos",
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Contenido del archivo de código Python",
            },
            "function_names": {
                "type": "array",
                "description": "Lista de nombres de funciones a extraer (opcional, todas si no se especifica)",
                "items": {"type": "string"},
            },
        },
        "required": ["content"],
    },
    category="code_analysis",
)
def extract_functions(
    content: str,
    function_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extract functions from Python code with their complete bodies.

    Args:
        content: Python source code content.
        function_names: Optional list of specific functions to extract.

    Returns:
        Dict with extracted functions.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {
            "success": False,
            "error": f"Error de sintaxis: {e}",
        }

    lines = content.split("\n")
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Filter by name if specified
            if function_names and node.name not in function_names:
                continue

            # Get function body
            start_line = node.lineno - 1
            end_line = node.end_lineno if node.end_lineno else start_line + 1
            body = "\n".join(lines[start_line:end_line])

            # Get return type annotation
            returns = None
            if node.returns:
                returns = ast.unparse(node.returns)

            # Get arguments with type hints
            args = []
            for arg in node.args.args:
                arg_info = {"name": arg.arg}
                if arg.annotation:
                    arg_info["type"] = ast.unparse(arg.annotation)
                args.append(arg_info)

            functions.append({
                "name": node.name,
                "start_line": node.lineno,
                "end_line": end_line,
                "args": args,
                "returns": returns,
                "docstring": ast.get_docstring(node),
                "body": body,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "decorators": [
                    ast.unparse(d) for d in node.decorator_list
                ],
            })

    return {
        "success": True,
        "function_count": len(functions),
        "functions": functions,
    }


@tool(
    name="extract_calculations",
    description="Extrae cálculos matemáticos de código Python",
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Contenido del archivo de código Python",
            },
            "variable_patterns": {
                "type": "array",
                "description": "Patrones de nombres de variables a buscar (ej: ['rwa', 'pd', 'lgd'])",
                "items": {"type": "string"},
            },
        },
        "required": ["content"],
    },
    category="code_analysis",
)
def extract_calculations(
    content: str,
    variable_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extract mathematical calculations from Python code.

    Args:
        content: Python source code content.
        variable_patterns: Optional patterns for variable names to search.

    Returns:
        Dict with extracted calculations.
    """
    calculations = []
    lines = content.split("\n")

    # Default patterns for banking calculations
    if not variable_patterns:
        variable_patterns = [
            "rwa", "pd", "lgd", "ead", "capital", "k_",
            "exposure", "risk", "weight", "ccf", "maturity",
        ]

    # Pattern for assignments with calculations
    assignment_pattern = re.compile(
        r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$"
    )

    # Pattern for augmented assignments
    aug_assignment_pattern = re.compile(
        r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([+\-*/])=\s*(.+)$"
    )

    for i, line in enumerate(lines):
        # Skip comments and empty lines
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Check regular assignments
        match = assignment_pattern.match(line)
        if match:
            var_name = match.group(1)
            expression = match.group(2).strip()

            # Check if variable matches patterns
            matches_pattern = any(
                pat.lower() in var_name.lower()
                for pat in variable_patterns
            )

            # Check if expression is a calculation (contains operators)
            is_calculation = bool(re.search(r"[+\-*/%()]", expression))

            if matches_pattern or is_calculation:
                # Get context (surrounding lines)
                context_start = max(0, i - 2)
                context_end = min(len(lines), i + 3)
                context = "\n".join(lines[context_start:context_end])

                # Extract variables used in expression
                used_vars = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", expression)
                # Filter out functions and keywords
                used_vars = [
                    v for v in used_vars
                    if v not in {"True", "False", "None", "and", "or", "not", "in", "is"}
                ]

                calculations.append({
                    "variable": var_name,
                    "expression": expression,
                    "line": i + 1,
                    "context": context,
                    "used_variables": list(set(used_vars)),
                    "matches_pattern": matches_pattern,
                })

        # Check augmented assignments
        aug_match = aug_assignment_pattern.match(line)
        if aug_match:
            var_name = aug_match.group(1)
            operator = aug_match.group(2)
            expression = aug_match.group(3).strip()

            matches_pattern = any(
                pat.lower() in var_name.lower()
                for pat in variable_patterns
            )

            if matches_pattern:
                calculations.append({
                    "variable": var_name,
                    "expression": f"{var_name} {operator}= {expression}",
                    "line": i + 1,
                    "is_augmented": True,
                    "operator": operator,
                })

    return {
        "success": True,
        "calculation_count": len(calculations),
        "calculations": calculations,
        "patterns_searched": variable_patterns,
    }


@tool(
    name="find_pattern_in_code",
    description="Busca patrones específicos en código",
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Contenido del archivo de código",
            },
            "pattern": {
                "type": "string",
                "description": "Patrón regex a buscar",
            },
            "context_lines": {
                "type": "integer",
                "description": "Número de líneas de contexto (default: 2)",
            },
        },
        "required": ["content", "pattern"],
    },
    category="code_analysis",
)
def find_pattern_in_code(
    content: str,
    pattern: str,
    context_lines: int = 2,
) -> Dict[str, Any]:
    """Find pattern matches in code.

    Args:
        content: Code content to search.
        pattern: Regex pattern to find.
        context_lines: Number of context lines around matches.

    Returns:
        Dict with matches and their contexts.
    """
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return {
            "success": False,
            "error": f"Patrón regex inválido: {e}",
        }

    lines = content.split("\n")
    matches = []

    for i, line in enumerate(lines):
        for match in regex.finditer(line):
            # Get context
            context_start = max(0, i - context_lines)
            context_end = min(len(lines), i + context_lines + 1)
            context = "\n".join(lines[context_start:context_end])

            matches.append({
                "line": i + 1,
                "column": match.start(),
                "match": match.group(),
                "full_line": line,
                "context": context,
            })

    return {
        "success": True,
        "pattern": pattern,
        "match_count": len(matches),
        "matches": matches,
    }

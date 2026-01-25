"""
Tests for Code Analysis Tools
"""

import sys
from pathlib import Path

# Add project root to path to allow direct imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

# Direct imports to avoid src/__init__.py heavy dependencies
from src.agents.tools.code_analysis_tools import (
    read_code_file,
    analyze_code_structure,
    extract_functions,
    extract_calculations,
    find_pattern_in_code,
)


class TestReadCodeFile:
    """Tests for read_code_file tool."""

    def test_read_existing_file(self, sample_code_file):
        """Test reading an existing code file."""
        result = read_code_file(str(sample_code_file))

        assert result["success"] is True
        assert result["language"] == "python"
        assert "content" in result
        assert result["total_lines"] > 0

    def test_read_nonexistent_file(self):
        """Test reading a non-existent file."""
        result = read_code_file("/nonexistent/path.py")

        assert result["success"] is False
        assert "error" in result

    def test_read_with_line_range(self, sample_code_file):
        """Test reading specific line range."""
        result = read_code_file(str(sample_code_file), start_line=1, end_line=10)

        assert result["success"] is True
        assert result["lines_returned"] == 10
        assert result["start_line"] == 1
        assert result["end_line"] == 10

    def test_detects_language(self, temp_dir):
        """Test language detection from extension."""
        js_file = temp_dir / "test.js"
        js_file.write_text("const x = 1;", encoding="utf-8")

        result = read_code_file(str(js_file))

        assert result["success"] is True
        assert result["language"] == "javascript"


class TestAnalyzeCodeStructure:
    """Tests for analyze_code_structure tool."""

    def test_analyze_structure(self, sample_code_content):
        """Test analyzing code structure."""
        result = analyze_code_structure(sample_code_content)

        assert result["success"] is True
        assert "structure" in result

    def test_finds_functions(self, sample_code_content):
        """Test that functions are found."""
        result = analyze_code_structure(sample_code_content)

        functions = result["structure"]["functions"]
        func_names = [f["name"] for f in functions]

        assert "calculate_pd" in func_names
        assert "calculate_lgd" in func_names
        assert "calculate_rwa" in func_names

    def test_finds_imports(self, sample_code_content):
        """Test that imports are found."""
        result = analyze_code_structure(sample_code_content)

        imports = result["structure"]["imports"]
        assert len(imports) > 0

        # Should find numpy import
        import_modules = [i["module"] for i in imports]
        assert any("numpy" in m or "np" in str(m) for m in import_modules)

    def test_finds_global_variables(self, sample_code_content):
        """Test that global variables are found."""
        result = analyze_code_structure(sample_code_content)

        global_vars = result["structure"]["global_variables"]
        var_names = [v["name"] for v in global_vars]

        assert "LGD_SENIOR" in var_names
        assert "LGD_SUBORDINATED" in var_names

    def test_handles_syntax_error(self):
        """Test handling of syntax errors."""
        invalid_code = "def broken(:\n    pass"

        result = analyze_code_structure(invalid_code)

        assert result["success"] is False
        assert "error" in result
        assert "sintaxis" in result["error"].lower()


class TestExtractFunctions:
    """Tests for extract_functions tool."""

    def test_extract_all_functions(self, sample_code_content):
        """Test extracting all functions."""
        result = extract_functions(sample_code_content)

        assert result["success"] is True
        assert result["function_count"] > 0

    def test_extract_specific_functions(self, sample_code_content):
        """Test extracting specific functions by name."""
        result = extract_functions(
            sample_code_content,
            function_names=["calculate_pd", "calculate_rwa"],
        )

        assert result["success"] is True
        assert result["function_count"] == 2

        func_names = [f["name"] for f in result["functions"]]
        assert "calculate_pd" in func_names
        assert "calculate_rwa" in func_names

    def test_function_structure(self, sample_code_content):
        """Test that functions have complete structure."""
        result = extract_functions(sample_code_content)

        for func in result["functions"]:
            assert "name" in func
            assert "start_line" in func
            assert "end_line" in func
            assert "args" in func
            assert "body" in func

    def test_extracts_docstrings(self, sample_code_content):
        """Test that docstrings are extracted."""
        result = extract_functions(sample_code_content)

        # At least some functions should have docstrings
        funcs_with_docs = [f for f in result["functions"] if f.get("docstring")]
        assert len(funcs_with_docs) > 0

    def test_extracts_return_types(self, sample_code_content):
        """Test that return type annotations are extracted."""
        result = extract_functions(sample_code_content)

        # Some functions have return type annotations
        funcs_with_returns = [f for f in result["functions"] if f.get("returns")]
        assert len(funcs_with_returns) > 0


class TestExtractCalculations:
    """Tests for extract_calculations tool."""

    def test_extract_calculations(self, sample_code_content):
        """Test extracting calculations from code."""
        result = extract_calculations(sample_code_content)

        assert result["success"] is True
        assert result["calculation_count"] > 0

    def test_finds_rwa_calculation(self, sample_code_content):
        """Test that RWA calculation is found."""
        result = extract_calculations(sample_code_content)

        calc_vars = [c["variable"].lower() for c in result["calculations"]]
        assert any("rwa" in v for v in calc_vars)

    def test_finds_capital_calculation(self, sample_code_content):
        """Test that capital calculation is found."""
        result = extract_calculations(sample_code_content)

        calc_vars = [c["variable"].lower() for c in result["calculations"]]
        assert any("k" in v or "capital" in v for v in calc_vars)

    def test_calculation_structure(self, sample_code_content):
        """Test that calculations have correct structure."""
        result = extract_calculations(sample_code_content)

        for calc in result["calculations"]:
            assert "variable" in calc
            assert "expression" in calc
            assert "line" in calc

    def test_with_custom_patterns(self, sample_code_content):
        """Test with custom variable patterns."""
        result = extract_calculations(
            sample_code_content,
            variable_patterns=["lgd", "pd", "ead"],
        )

        assert result["success"] is True
        assert "lgd" in result["patterns_searched"]


class TestFindPatternInCode:
    """Tests for find_pattern_in_code tool."""

    def test_find_simple_pattern(self, sample_code_content):
        """Test finding a simple pattern."""
        result = find_pattern_in_code(
            sample_code_content,
            pattern="def calculate",
        )

        assert result["success"] is True
        assert result["match_count"] > 0

    def test_find_regex_pattern(self, sample_code_content):
        """Test finding a regex pattern."""
        result = find_pattern_in_code(
            sample_code_content,
            pattern=r"def calculate_\w+",
        )

        assert result["success"] is True
        assert result["match_count"] > 0

    def test_includes_context(self, sample_code_content):
        """Test that context lines are included."""
        result = find_pattern_in_code(
            sample_code_content,
            pattern="LGD_SENIOR",
            context_lines=2,
        )

        assert result["success"] is True
        for match in result["matches"]:
            assert "context" in match
            # Context should include multiple lines
            assert "\n" in match["context"]

    def test_invalid_pattern(self, sample_code_content):
        """Test handling of invalid regex pattern."""
        result = find_pattern_in_code(
            sample_code_content,
            pattern="[invalid(regex",
        )

        assert result["success"] is False
        assert "error" in result

    def test_no_matches(self, sample_code_content):
        """Test when pattern has no matches."""
        result = find_pattern_in_code(
            sample_code_content,
            pattern="definitely_not_in_code_xyz123",
        )

        assert result["success"] is True
        assert result["match_count"] == 0

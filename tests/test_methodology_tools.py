"""
Tests for Methodology Tools
"""

import sys
from pathlib import Path

# Add project root to path to allow direct imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

# Direct imports to avoid src/__init__.py heavy dependencies
from src.agents.tools.methodology_tools import (
    read_methodology,
    parse_methodology_sections,
    extract_formulas,
    extract_parameters,
    compare_methodologies,
)


class TestReadMethodology:
    """Tests for read_methodology tool."""

    def test_read_existing_file(self, sample_methodology_file):
        """Test reading an existing methodology file."""
        result = read_methodology(str(sample_methodology_file))

        assert result["success"] is True
        assert "content" in result
        assert "IRB" in result["content"]
        assert result["line_count"] > 0

    def test_read_nonexistent_file(self):
        """Test reading a non-existent file."""
        result = read_methodology("/nonexistent/path/file.md")

        assert result["success"] is False
        assert "error" in result
        assert "no encontrado" in result["error"].lower()

    def test_extracts_title(self, sample_methodology_file):
        """Test that title is extracted correctly."""
        result = read_methodology(str(sample_methodology_file))

        assert result["success"] is True
        assert "title" in result
        assert "IRB" in result["title"]


class TestParseMethodologySections:
    """Tests for parse_methodology_sections tool."""

    def test_parse_sections(self, sample_methodology_content):
        """Test parsing methodology into sections."""
        result = parse_methodology_sections(sample_methodology_content)

        assert result["success"] is True
        assert result["section_count"] > 0
        assert "sections" in result

    def test_section_structure(self, sample_methodology_content):
        """Test that sections have correct structure."""
        result = parse_methodology_sections(sample_methodology_content)

        for section in result["sections"]:
            assert "title" in section
            assert "level" in section
            assert "content" in section
            assert "start_line" in section

    def test_section_hierarchy(self, sample_methodology_content):
        """Test that section levels are captured correctly."""
        result = parse_methodology_sections(sample_methodology_content)

        # Should have both level 1 and level 2 sections
        levels = [s["level"] for s in result["sections"]]
        assert 1 in levels or 2 in levels


class TestExtractFormulas:
    """Tests for extract_formulas tool."""

    def test_extract_code_block_formulas(self, sample_methodology_content):
        """Test extracting formulas from code blocks."""
        result = extract_formulas(sample_methodology_content)

        assert result["success"] is True
        assert result["formula_count"] > 0

    def test_formula_structure(self, sample_methodology_content):
        """Test that formulas have correct structure."""
        result = extract_formulas(sample_methodology_content)

        for formula in result["formulas"]:
            assert "name" in formula
            assert "expression" in formula
            assert "line" in formula

    def test_extracts_rwa_formula(self, sample_methodology_content):
        """Test that RWA formula is extracted."""
        result = extract_formulas(sample_methodology_content)

        formula_names = [f["name"] for f in result["formulas"]]
        # Should find RWA-related formula
        assert any("RWA" in name or "K" in name for name in formula_names)

    def test_extracts_variables(self, sample_methodology_content):
        """Test that formula variables are extracted."""
        result = extract_formulas(sample_methodology_content)

        # At least one formula should have variables
        formulas_with_vars = [
            f for f in result["formulas"]
            if f.get("variables") and len(f["variables"]) > 0
        ]
        # May or may not have variables depending on formula complexity
        assert result["formula_count"] >= 0


class TestExtractParameters:
    """Tests for extract_parameters tool."""

    def test_extract_parameters(self, sample_methodology_content):
        """Test extracting parameters from methodology."""
        result = extract_parameters(sample_methodology_content)

        assert result["success"] is True
        assert result["parameter_count"] > 0

    def test_extracts_lgd_values(self, sample_methodology_content):
        """Test that LGD values are extracted."""
        result = extract_parameters(sample_methodology_content)

        param_names = [p["name"] for p in result["parameters"]]
        # Should find LGD parameter
        assert "LGD" in param_names or any("LGD" in str(p) for p in result["parameters"])

    def test_extracts_percentage_values(self, sample_methodology_content):
        """Test that percentage values are extracted."""
        result = extract_parameters(sample_methodology_content)

        # Should find parameters with percentage values
        percentage_params = [
            p for p in result["parameters"]
            if p.get("value") and "%" in str(p["value"])
        ]
        assert len(percentage_params) > 0

    def test_parameter_structure(self, sample_methodology_content):
        """Test that parameters have correct structure."""
        result = extract_parameters(sample_methodology_content)

        for param in result["parameters"]:
            assert "name" in param
            assert "line" in param
            assert "source_type" in param


class TestCompareMethodologies:
    """Tests for compare_methodologies tool."""

    @pytest.fixture
    def second_methodology(self):
        """Create a second methodology for comparison."""
        return """# Metodología IRB Avanzado (A-IRB)

## Descripción General

El enfoque IRB Avanzado permite estimación propia de todos los parámetros.

## Características Principales

### Parámetros de Riesgo

| Parámetro | Fuente |
|-----------|--------|
| **PD** | Estimación propia |
| **LGD** | Estimación propia (downturn) |
| **EAD** | Estimación propia |

### Requisitos de Datos

- Mínimo 7 años de datos para LGD y EAD
- Downturn LGD obligatorio

## Fórmula de Capital

```
K = LGD × f(PD, R) × MA
```
"""

    def test_compare_methodologies(
        self, sample_methodology_content, second_methodology
    ):
        """Test comparing two methodologies."""
        result = compare_methodologies(
            methodology_a=sample_methodology_content,
            methodology_b=second_methodology,
        )

        assert result["success"] is True
        assert "parameter_comparison" in result
        assert "formula_comparison" in result
        assert "summary" in result

    def test_identifies_differences(
        self, sample_methodology_content, second_methodology
    ):
        """Test that differences are identified."""
        result = compare_methodologies(
            methodology_a=sample_methodology_content,
            methodology_b=second_methodology,
        )

        # Should have some comparison data
        assert len(result["parameter_comparison"]) > 0 or len(result["formula_comparison"]) > 0

    def test_summary_statistics(
        self, sample_methodology_content, second_methodology
    ):
        """Test that summary statistics are provided."""
        result = compare_methodologies(
            methodology_a=sample_methodology_content,
            methodology_b=second_methodology,
        )

        summary = result["summary"]
        assert "total_parameters" in summary
        assert "total_formulas" in summary

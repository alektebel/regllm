"""
Pytest configuration and fixtures for RegLLM tests.
"""

import sys
from pathlib import Path

# Add project root to path to allow direct imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import tempfile
import shutil


@pytest.fixture
def sample_methodology_content():
    """Sample methodology document content."""
    return """# Metodología IRB Fundación (F-IRB)

## Descripción General

El enfoque IRB Fundación es una metodología para el cálculo de requisitos de capital.

## Características Principales

### Parámetros de Riesgo

| Parámetro | Fuente | Descripción |
|-----------|--------|-------------|
| **PD** (Probabilidad de Impago) | Estimación propia | La entidad desarrolla sus propios modelos de PD |
| **LGD** (Pérdida en caso de Impago) | Valores supervisores | Prescritos por el regulador (45% senior, 75% subordinado) |
| **EAD** (Exposición en caso de Impago) | Valores supervisores | Basados en factores de conversión regulatorios |

### Valores Supervisores de LGD

- **Exposiciones senior sin garantías**: 45%
- **Exposiciones subordinadas**: 75%

## Fórmula de RWA

Los activos ponderados por riesgo se calculan como:

```
RWA = K × 12.5 × EAD
```

Donde K es el requerimiento de capital:

```
K = LGD × N[(1-R)^-0.5 × G(PD) + (R/(1-R))^0.5 × G(0.999)] - PD × LGD
```

## Requisitos de Implementación

1. Sistema de calificación interno
2. Mínimo 5 años de datos para estimación de PD
3. Validación independiente anual
"""


@pytest.fixture
def sample_code_content():
    """Sample code implementation content."""
    return '''"""
IRB Foundation Implementation
"""

import numpy as np
from scipy.stats import norm


# Supervisory LGD values
LGD_SENIOR = 0.45
LGD_SUBORDINATED = 0.75

# Default CCF values
CCF_COMMITTED = 0.75
CCF_UNCONDITIONAL = 0.0


def calculate_pd(ratings_data: list, years: int = 5) -> float:
    """Calculate Probability of Default from historical data.

    Args:
        ratings_data: Historical ratings data
        years: Number of years of data (minimum 5 required)

    Returns:
        Estimated PD
    """
    if years < 5:
        raise ValueError("Minimum 5 years of data required")

    # Simplified PD calculation
    defaults = sum(1 for r in ratings_data if r.get("defaulted"))
    total = len(ratings_data)

    pd = defaults / total if total > 0 else 0.03
    return max(0.0003, min(pd, 1.0))  # Floor and cap


def calculate_lgd(exposure_type: str = "senior", has_collateral: bool = False) -> float:
    """Get LGD value based on exposure type.

    Args:
        exposure_type: "senior" or "subordinated"
        has_collateral: Whether exposure has collateral

    Returns:
        LGD value
    """
    if exposure_type == "senior":
        lgd = LGD_SENIOR
    else:
        lgd = LGD_SUBORDINATED

    if has_collateral:
        lgd = lgd * 0.8  # Simplified collateral adjustment

    return lgd


def calculate_ead(committed_amount: float, drawn_amount: float, ccf: float = None) -> float:
    """Calculate Exposure at Default.

    Args:
        committed_amount: Total committed credit
        drawn_amount: Currently drawn amount
        ccf: Credit conversion factor (optional)

    Returns:
        EAD value
    """
    if ccf is None:
        ccf = CCF_COMMITTED

    undrawn = committed_amount - drawn_amount
    ead = drawn_amount + (ccf * undrawn)

    return ead


def calculate_asset_correlation(pd: float) -> float:
    """Calculate asset correlation based on PD.

    Args:
        pd: Probability of default

    Returns:
        Asset correlation (R)
    """
    # Basel formula for corporate exposures
    exp_factor = np.exp(-50 * pd)
    r = 0.12 * (1 - exp_factor) / (1 - np.exp(-50)) + 0.24 * (1 - (1 - exp_factor) / (1 - np.exp(-50)))

    return r


def calculate_maturity_adjustment(pd: float, m: float = 2.5) -> float:
    """Calculate maturity adjustment.

    Args:
        pd: Probability of default
        m: Effective maturity (default 2.5 years)

    Returns:
        Maturity adjustment factor
    """
    b = (0.11852 - 0.05478 * np.log(pd)) ** 2
    ma = (1 + (m - 2.5) * b) / (1 - 1.5 * b)

    return ma


def calculate_capital_requirement(pd: float, lgd: float, m: float = 2.5) -> float:
    """Calculate capital requirement K.

    Args:
        pd: Probability of default
        lgd: Loss given default
        m: Effective maturity

    Returns:
        Capital requirement K
    """
    # Asset correlation
    r = calculate_asset_correlation(pd)

    # Capital requirement before maturity adjustment
    k_base = lgd * (
        norm.cdf(
            (1 - r) ** -0.5 * norm.ppf(pd) +
            (r / (1 - r)) ** 0.5 * norm.ppf(0.999)
        ) - pd
    )

    # Maturity adjustment
    ma = calculate_maturity_adjustment(pd, m)

    k = k_base * ma

    return k


def calculate_rwa(ead: float, pd: float, lgd: float, m: float = 2.5) -> float:
    """Calculate Risk-Weighted Assets.

    Args:
        ead: Exposure at default
        pd: Probability of default
        lgd: Loss given default
        m: Effective maturity

    Returns:
        RWA value
    """
    k = calculate_capital_requirement(pd, lgd, m)
    rwa = k * 12.5 * ead

    return rwa
'''


@pytest.fixture
def sample_inconsistent_code():
    """Sample code with inconsistencies."""
    return '''"""
IRB Implementation with inconsistencies
"""

# Wrong LGD values (should be 0.45 and 0.75)
LGD_SENIOR = 0.40  # Inconsistent!
LGD_SUBORDINATED = 0.80  # Inconsistent!


def calculate_pd(data):
    """Calculate PD - missing years requirement check."""
    # Missing validation for 5-year minimum
    return 0.02


def calculate_rwa(ead, pd, lgd):
    """Calculate RWA - wrong formula."""
    # Missing maturity adjustment
    k = lgd * pd  # Simplified, missing correlation
    rwa = k * 10 * ead  # Wrong multiplier (should be 12.5)
    return rwa
'''


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_methodology_file(temp_dir, sample_methodology_content):
    """Create a sample methodology file."""
    file_path = temp_dir / "methodology_test.md"
    file_path.write_text(sample_methodology_content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_code_file(temp_dir, sample_code_content):
    """Create a sample code file."""
    file_path = temp_dir / "irb_implementation.py"
    file_path.write_text(sample_code_content, encoding="utf-8")
    return file_path


@pytest.fixture
def tool_registry():
    """Create a fresh tool registry for testing."""
    from src.agents.tool_registry import ToolRegistry
    return ToolRegistry()


@pytest.fixture
def registered_registry(tool_registry):
    """Create a registry with all tools registered."""
    from src.agents.tools.registration import register_all_tools
    return register_all_tools(tool_registry)

"""Tests for formula grounding verifier."""

from __future__ import annotations

from src.agent.formula_guard import check_formula_grounding


def _tool_msg(name: str, payload: dict) -> dict:
    import json
    return {"role": "tool", "name": name, "content": json.dumps(payload)}


def test_grounded_formula_passes() -> None:
    messages = [
        _tool_msg("get_sas_formula", {
            "field": "MOC",
            "definitions": [{"data_step": "work.lgd_final", "expr": "0.05 * LGD_ESTIMADA"}],
        }),
    ]
    answer = "MoC = 0.05 * LGD_ESTIMADA según work.lgd_final."
    assert check_formula_grounding(answer, messages) == []


def test_hallucinated_coef_fails() -> None:
    messages = [
        _tool_msg("get_sas_formula", {
            "field": "MOC",
            "definitions": [{"expr": "0.05 * LGD_ESTIMADA"}],
        }),
    ]
    answer = "MoC = 0.30 × LGD_ESTIMADA + 0.70 × LGD_FLOOR para hipotecas."
    violations = check_formula_grounding(answer, messages)
    assert any("0.30" in v for v in violations)


def test_formula_without_tools_fails() -> None:
    answer = "ECL = PD_ESTIMADA * LGD_CON_MOC * EAD"
    violations = check_formula_grounding(answer, [])
    assert any("sin llamar" in v for v in violations)


def test_suspicious_dataset_fails() -> None:
    messages = [
        _tool_msg("trace_dependencies", {"target": "MOC", "edges": []}),
    ]
    answer = "MoC proviene de catalogo.ciclos_recuperacion."
    violations = check_formula_grounding(answer, messages)
    assert any("catalogo" in v.lower() for v in violations)

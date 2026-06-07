"""High-level orchestrator: explain why a target field differs between V2 and V3.

Combines the path-integrated gradient attribution (numeric, smooth) with
Shapley values (handles categoricals, branch flips, non-smoothness) into
a single ``DiscrepancyReport``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.sas_logic_tree import AnyNode, SASLogicTree

from .gradient_explainer import (
    BranchFlip,
    GradientReport,
    path_integrated_gradients,
)
from .shapley_explainer import ShapleyReport, shapley_values


Method = Literal["grad", "shapley", "both"]


@dataclass
class FieldDelta:
    field: str
    v2: Any
    v3: Any
    is_numeric: bool

    def to_dict(self) -> dict:
        return {"field": self.field, "v2": self.v2, "v3": self.v3, "is_numeric": self.is_numeric}


@dataclass
class DiscrepancyReport:
    target: str
    y_v2: float | None
    y_v3: float | None
    delta_y: float | None
    excluded_v2: bool
    excluded_v3: bool
    field_deltas: list[FieldDelta]
    branch_flips: list[BranchFlip]
    gradient: GradientReport | None
    shapley: ShapleyReport | None
    suspects: list[str]                    # ranked field names worth investigating

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "y_v2": self.y_v2,
            "y_v3": self.y_v3,
            "delta_y": self.delta_y,
            "excluded_v2": self.excluded_v2,
            "excluded_v3": self.excluded_v3,
            "field_deltas": [d.to_dict() for d in self.field_deltas],
            "branch_flips": [b.to_dict() for b in self.branch_flips],
            "gradient": self.gradient.to_dict() if self.gradient else None,
            "shapley": self.shapley.to_dict() if self.shapley else None,
            "suspects": self.suspects,
        }


def _norm(row: dict[str, Any]) -> dict[str, Any]:
    return {k.upper() if isinstance(k, str) else k: v for k, v in row.items()}


def _field_deltas(v2: dict[str, Any], v3: dict[str, Any]) -> list[FieldDelta]:
    out: list[FieldDelta] = []
    keys = sorted(set(v2) | set(v3))
    for k in keys:
        a, b = v2.get(k), v3.get(k)
        if a == b:
            continue
        if a is None and b is None:
            continue
        is_num = (
            isinstance(a, (int, float)) and not isinstance(a, bool)
            and isinstance(b, (int, float)) and not isinstance(b, bool)
        )
        out.append(FieldDelta(field=k, v2=a, v3=b, is_numeric=is_num))
    return out


def explain_field_diff(
    sas_code: str | None = None,
    nodes: list[AnyNode] | None = None,
    *,
    row_v2: dict[str, Any],
    row_v3: dict[str, Any],
    target: str,
    method: Method = "both",
    grad_steps: int = 32,
    shapley_n_samples: int = 256,
    shapley_exact_threshold: int = 12,
) -> DiscrepancyReport:
    """Run the full discrepancy analysis between two row versions.

    Parameters
    ----------
    sas_code:
        Raw SAS source. Either this or ``nodes`` must be provided.
    nodes:
        Pre-parsed AST. Takes precedence over ``sas_code`` if both are given.
    row_v2, row_v3:
        Two row dicts. Numeric fields are interpolated for the gradient
        method; all differing fields enter Shapley.
    target:
        Output column whose Δ we want to explain.
    method:
        ``"grad"``, ``"shapley"`` or ``"both"`` (default).
    """
    if nodes is None:
        if sas_code is None:
            raise ValueError("Either nodes or sas_code must be provided")
        nodes = SASLogicTree().parse(sas_code)

    target_u = target.upper()
    v2 = _norm(row_v2)
    v3 = _norm(row_v3)

    deltas = _field_deltas(v2, v3)

    grad_report: GradientReport | None = None
    shap_report: ShapleyReport | None = None

    if method in ("grad", "both"):
        grad_report = path_integrated_gradients(
            nodes, v2, v3, target_u, steps=grad_steps,
        )

    if method in ("shapley", "both"):
        shap_report = shapley_values(
            nodes, v2, v3, target_u,
            n_samples=shapley_n_samples,
            exact_threshold=shapley_exact_threshold,
        )

    # Build a unified suspect ranking
    scores: dict[str, float] = {}
    if grad_report:
        for a in grad_report.attributions:
            scores[a.field] = scores.get(a.field, 0.0) + abs(a.contribution)
    if shap_report:
        for a in shap_report.attributions:
            scores[a.field] = scores.get(a.field, 0.0) + abs(a.contribution)
    # Always elevate fields participating in branch flips
    flip_vars: set[str] = set()
    if grad_report:
        for fl in grad_report.branch_flips:
            for v in fl.vars_in_condition:
                flip_vars.add(v)
    suspects = sorted(scores, key=lambda k: scores[k], reverse=True)
    for v in flip_vars:
        if v not in suspects:
            suspects.insert(0, v)

    # Pull anchor values from whichever report is available
    if grad_report is not None:
        y_v2, y_v3, delta_y = grad_report.y_v2, grad_report.y_v3, grad_report.delta_y
        excl_v2, excl_v3 = grad_report.excluded_v2, grad_report.excluded_v3
        flips = grad_report.branch_flips
    elif shap_report is not None:
        y_v2, y_v3, delta_y = shap_report.y_v2, shap_report.y_v3, shap_report.delta_y
        excl_v2 = excl_v3 = False
        flips = []
    else:
        y_v2 = y_v3 = delta_y = None
        excl_v2 = excl_v3 = False
        flips = []

    return DiscrepancyReport(
        target=target_u,
        y_v2=y_v2,
        y_v3=y_v3,
        delta_y=delta_y,
        excluded_v2=excl_v2,
        excluded_v3=excl_v3,
        field_deltas=deltas,
        branch_flips=flips,
        gradient=grad_report,
        shapley=shap_report,
        suspects=suspects,
    )

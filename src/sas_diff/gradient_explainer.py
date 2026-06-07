"""Path-integrated gradient attribution (Aumann–Shapley) for SAS pipelines.

Given the SAS AST, an input row at version V2 and another at V3, and a
chosen target field ``Y``, this module attributes ``Y(V3) − Y(V2)`` to
each numeric input field ``x_i`` via the path integral

    phi_i = (x_i^V3 − x_i^V2) · ∫_0^1 ∂Y/∂x_i (X^V2 + t·(X^V3 − X^V2)) dt

approximated by the trapezoidal rule with ``steps`` sample points along
the line segment.

Properties
----------
- *Efficiency*: ``Σ_i phi_i ≈ Y(V3) − Y(V2)`` whenever no branch is flipped
  along the path. When branches flip, the residual is reported under
  ``unexplained_residual`` and the responsible predicate variables are
  surfaced via ``branch_flips``.
- *Linearity*: φ is linear in the function evaluated.
- Categorical / string inputs are NOT differentiable; their contribution
  is computed by the ShapleyExplainer, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from src.sas_logic_tree import AnyNode

from .tensor_evaluator import (
    BranchEvent,
    TensorEvaluator,
    TensorTrace,
)


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass
class GradAttribution:
    field: str
    delta_input: float           # x_v3 − x_v2
    contribution: float          # phi_i  (signed; sums to delta_y modulo branch flips)
    avg_gradient: float          # mean ∂Y/∂x_i along the path

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "delta_input": self.delta_input,
            "contribution": self.contribution,
            "avg_gradient": self.avg_gradient,
        }


@dataclass
class BranchFlip:
    condition: str
    data_step: str
    kind: str                    # "if" | "where" | "select"
    v2_taken: bool
    v3_taken: bool
    vars_in_condition: list[str]

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "data_step": self.data_step,
            "kind": self.kind,
            "v2_taken": self.v2_taken,
            "v3_taken": self.v3_taken,
            "vars_in_condition": self.vars_in_condition,
        }


@dataclass
class GradientReport:
    target: str
    y_v2: float | None
    y_v3: float | None
    delta_y: float | None
    excluded_v2: bool
    excluded_v3: bool
    attributions: list[GradAttribution] = field(default_factory=list)
    branch_flips: list[BranchFlip] = field(default_factory=list)
    unexplained_residual: float = 0.0
    steps: int = 32

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "y_v2": self.y_v2,
            "y_v3": self.y_v3,
            "delta_y": self.delta_y,
            "excluded_v2": self.excluded_v2,
            "excluded_v3": self.excluded_v3,
            "attributions": [a.to_dict() for a in self.attributions],
            "branch_flips": [b.to_dict() for b in self.branch_flips],
            "unexplained_residual": self.unexplained_residual,
            "steps": self.steps,
        }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _evaluate_at(
    nodes: list[AnyNode],
    values: dict[str, Any],
) -> TensorTrace:
    """Evaluate the AST with the given values, returning a trace."""
    env = TensorEvaluator.make_env(values, requires_grad=False)
    return TensorEvaluator().run(nodes, env)


def _branch_flips(
    log_v2: list[BranchEvent],
    log_v3: list[BranchEvent],
) -> list[BranchFlip]:
    """Pair up branch events by (kind, condition, data_step) and report mismatches."""
    flips: list[BranchFlip] = []
    # Index v2 events by key with a counter of occurrences; we pop as we match
    used_v2: list[bool] = [False] * len(log_v2)
    for ev3 in log_v3:
        key3 = (ev3.kind, ev3.condition, ev3.data_step)
        for j, ev2 in enumerate(log_v2):
            if used_v2[j]:
                continue
            if (ev2.kind, ev2.condition, ev2.data_step) == key3:
                used_v2[j] = True
                if ev2.taken != ev3.taken:
                    flips.append(BranchFlip(
                        condition=ev3.condition,
                        data_step=ev3.data_step,
                        kind=ev3.kind,
                        v2_taken=ev2.taken,
                        v3_taken=ev3.taken,
                        vars_in_condition=ev3.vars_in_condition,
                    ))
                break
        else:
            # Branch only seen in V3 — typically caused by a flip upstream
            flips.append(BranchFlip(
                condition=ev3.condition,
                data_step=ev3.data_step,
                kind=ev3.kind,
                v2_taken=False,
                v3_taken=ev3.taken,
                vars_in_condition=ev3.vars_in_condition,
            ))
    # Branches only seen in V2
    for j, ev2 in enumerate(log_v2):
        if not used_v2[j]:
            flips.append(BranchFlip(
                condition=ev2.condition,
                data_step=ev2.data_step,
                kind=ev2.kind,
                v2_taken=ev2.taken,
                v3_taken=False,
                vars_in_condition=ev2.vars_in_condition,
            ))
    return flips


# ── Main API ─────────────────────────────────────────────────────────────────


def path_integrated_gradients(
    nodes: list[AnyNode],
    row_v2: dict[str, Any],
    row_v3: dict[str, Any],
    target: str,
    steps: int = 32,
    branch_row: str = "v3",
) -> GradientReport:
    """Compute path-integrated gradient attributions for ``target`` between V2 and V3.

    Parameters
    ----------
    nodes:
        Parsed SAS AST (list of top-level nodes).
    row_v2, row_v3:
        Row dicts (column → value). Numeric fields are interpolated; categorical
        ones are held at the ``branch_row`` value.
    target:
        Column name whose Y(V3)−Y(V2) we want to explain (case-insensitive).
    steps:
        Number of trapezoidal sample points along the path. Higher = lower
        approximation error.
    branch_row:
        ``"v3"`` (default) or ``"v2"``: which row's categorical values are used
        when fixing the branch path during the path integral.
    """
    target = target.upper()

    # 1. Anchor evaluations to obtain Y(V2), Y(V3) and branch logs
    trace_v2 = _evaluate_at(nodes, row_v2)
    trace_v3 = _evaluate_at(nodes, row_v3)
    y_v2 = trace_v2.scalar(target)
    y_v3 = trace_v3.scalar(target)
    delta_y = (y_v3 - y_v2) if (y_v2 is not None and y_v3 is not None) else None
    flips = _branch_flips(trace_v2.branch_log, trace_v3.branch_log)

    report = GradientReport(
        target=target,
        y_v2=y_v2,
        y_v3=y_v3,
        delta_y=delta_y,
        excluded_v2=trace_v2.excluded,
        excluded_v3=trace_v3.excluded,
        branch_flips=flips,
        steps=steps,
    )

    # If either side is excluded or target was not produced, gradients are
    # undefined; return early with the bookkeeping we already have.
    if y_v2 is None or y_v3 is None:
        return report

    # 2. Compute the path integral via composite Simpson's rule
    # (exact for cubics — covers most multilinear SAS pipelines).
    # Simpson's needs an even number of intervals → odd number of nodes.
    if steps < 3:
        steps = 3
    if steps % 2 == 0:
        steps += 1
    n_intervals = steps - 1
    h_step = 1.0 / n_intervals
    ts = [k * h_step for k in range(steps)]
    # Simpson weights: 1, 4, 2, 4, 2, ..., 4, 1 — multiplied by h/3 for ∫_0^1
    weights = [1.0] * steps
    for i in range(1, steps - 1):
        weights[i] = 4.0 if i % 2 == 1 else 2.0
    h = h_step / 3.0

    # Track per-field running sum of gradients
    grad_sum: dict[str, float] = {}
    grad_count: dict[str, int] = {}

    target_seen_at_least_once = False

    for t, w in zip(ts, weights):
        env, knobs = TensorEvaluator.make_path_env(row_v2, row_v3, t, branch_row=branch_row)
        if not knobs:
            break
        trace = TensorEvaluator().run(nodes, env)
        y = trace.value(target)
        if not isinstance(y, torch.Tensor):
            # Target produced as a non-numeric or excluded; skip this t
            continue
        target_seen_at_least_once = True
        # Compute ∂y/∂x_i for every knob
        grads = torch.autograd.grad(
            y, list(knobs.values()),
            retain_graph=False, create_graph=False,
            allow_unused=True,
        )
        for (name, _), g in zip(knobs.items(), grads):
            g_val = 0.0 if g is None else float(g.detach().item())
            grad_sum[name] = grad_sum.get(name, 0.0) + w * h * g_val
            grad_count[name] = grad_count.get(name, 0) + 1

    if not target_seen_at_least_once:
        return report

    # 3. Build attributions: phi_i = (x_v3 − x_v2) · ∫ ∂Y/∂x_i dt
    attrs: list[GradAttribution] = []
    for name, integral in grad_sum.items():
        v2 = row_v2.get(name) if name in row_v2 else row_v2.get(name.lower())
        v3 = row_v3.get(name) if name in row_v3 else row_v3.get(name.lower())
        if not (isinstance(v2, (int, float)) and isinstance(v3, (int, float))):
            continue
        if isinstance(v2, bool) or isinstance(v3, bool):
            continue
        delta_x = float(v3) - float(v2)
        contribution = delta_x * integral
        avg_grad = integral  # already integrated over [0,1]
        if abs(delta_x) < 1e-15 and abs(contribution) < 1e-15:
            continue
        attrs.append(GradAttribution(
            field=name,
            delta_input=delta_x,
            contribution=contribution,
            avg_gradient=avg_grad,
        ))

    attrs.sort(key=lambda a: abs(a.contribution), reverse=True)
    report.attributions = attrs

    explained = sum(a.contribution for a in attrs)
    report.unexplained_residual = (delta_y or 0.0) - explained

    return report

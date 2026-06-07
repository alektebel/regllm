"""SAS field-diff explainer: gradient + Shapley attribution for V2→V3 deltas."""

from .discrepancy import DiscrepancyReport, FieldDelta, Method, explain_field_diff
from .gradient_explainer import (
    BranchFlip,
    GradAttribution,
    GradientReport,
    path_integrated_gradients,
)
from .shapley_explainer import ShapAttribution, ShapleyReport, shapley_values
from .tensor_evaluator import (
    BranchEvent,
    TensorEvaluator,
    TensorTrace,
    evaluate_tensor,
)

__all__ = [
    "DiscrepancyReport", "FieldDelta", "Method", "explain_field_diff",
    "GradAttribution", "GradientReport", "BranchFlip", "path_integrated_gradients",
    "ShapAttribution", "ShapleyReport", "shapley_values",
    "BranchEvent", "TensorEvaluator", "TensorTrace", "evaluate_tensor",
]

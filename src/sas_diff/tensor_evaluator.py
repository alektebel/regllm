"""Differentiable tensor-based evaluator for SAS DATA-step ASTs.

Re-uses the parser and AST nodes from ``src.sas_logic_tree`` but evaluates
numeric expressions with ``torch.tensor(..., requires_grad=True)`` so the
autograd engine can compute the gradient of any output field with respect
to any numeric input field.

Branches (``IF`` / ``SELECT`` / ``WHERE``) are evaluated eagerly: the
condition is detached from the autograd graph and the chosen body is
walked. Detecting branch flips between two rows (V2 / V3) is done at a
higher level by running the evaluator twice and diff-ing ``branch_log``.

Categorical / string fields are passed through unchanged. Non-smooth
operators (``floor``, ``ceil``, ``round``) detach from autograd to avoid
producing zero gradients silently.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import torch

from src.sas_logic_tree import (
    AnyNode,
    AssignNode,
    DataStepNode,
    DeleteNode,
    DoLoopNode,
    FilterNode,
    IfNode,
    MacroCallNode,
    MacroDefNode,
    MacroLetNode,
    MergeNode,
    ProcNode,
    RetainNode,
    SelectNode,
    _to_python,
)


# ── Branch trace ──────────────────────────────────────────────────────────────


@dataclass
class BranchEvent:
    """A single branch decision recorded during evaluation."""

    kind: str                       # "if" | "where" | "select" | "select_otherwise"
    condition: str
    taken: bool
    data_step: str
    vars_in_condition: list[str] = field(default_factory=list)


@dataclass
class TensorTrace:
    """Result of one evaluation pass."""

    final: dict[str, Any]
    branch_log: list[BranchEvent]
    excluded: bool

    def value(self, name: str) -> Any:
        return self.final.get(name.upper())

    def scalar(self, name: str) -> float | None:
        v = self.value(name)
        if isinstance(v, torch.Tensor):
            return float(v.detach().item())
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return None


# ── torch-aware versions of the SAS function library ─────────────────────────


def _as_tensor(x: Any) -> torch.Tensor | None:
    if isinstance(x, torch.Tensor):
        return x
    try:
        return torch.tensor(float(x), dtype=torch.float64)
    except (TypeError, ValueError):
        return None


def _has_tensor(args) -> bool:
    return any(isinstance(a, torch.Tensor) for a in args)


def _tmax(*args):
    if _has_tensor(args):
        ts = [_as_tensor(a) for a in args if a is not None]
        return torch.stack(ts).max() if ts else None
    return max(a for a in args if a is not None)


def _tmin(*args):
    if _has_tensor(args):
        ts = [_as_tensor(a) for a in args if a is not None]
        return torch.stack(ts).min() if ts else None
    return min(a for a in args if a is not None)


def _tabs(x):
    return torch.abs(x) if isinstance(x, torch.Tensor) else (abs(x) if x is not None else None)


def _tlog(x):
    if isinstance(x, torch.Tensor):
        return torch.log(x)
    return math.log(x) if x is not None and x > 0 else None


def _tlog2(x):
    if isinstance(x, torch.Tensor):
        return torch.log2(x)
    return math.log2(x) if x is not None and x > 0 else None


def _tlog10(x):
    if isinstance(x, torch.Tensor):
        return torch.log10(x)
    return math.log10(x) if x is not None and x > 0 else None


def _texp(x):
    return torch.exp(x) if isinstance(x, torch.Tensor) else (math.exp(x) if x is not None else None)


def _tsqrt(x):
    return torch.sqrt(x) if isinstance(x, torch.Tensor) else (math.sqrt(x) if x is not None else None)


def _tfloor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().floor()
    return math.floor(x) if x is not None else None


def _tceil(x):
    if isinstance(x, torch.Tensor):
        return x.detach().ceil()
    return math.ceil(x) if x is not None else None


def _tround(x, n=1):
    # Non-smooth — detach to keep gradients zero rather than NaN
    if isinstance(x, torch.Tensor):
        return torch.round(x.detach() / n) * n if n else x
    if x is None:
        return None
    return round(x / n) * n if n else x


def _tsign(x):
    if isinstance(x, torch.Tensor):
        return torch.sign(x.detach())
    if x is None:
        return None
    return 1 if x > 0 else -1 if x < 0 else 0


def _tcoalesce(*args):
    for a in args:
        if a is not None:
            return a
    return None


def _tcoalescec(*args):
    for a in args:
        if a is not None and a != "":
            return a
    return None


def _tifn(cond, t, f):
    if isinstance(cond, torch.Tensor):
        cond = bool(cond.detach().item())
    return t if bool(cond) else f


def _tnmiss(*args):
    return sum(1 for a in args if a is None)


def _tn(*args):
    return sum(1 for a in args if a is not None)


def _tsum(*args):
    vals = [a for a in args if a is not None]
    if not vals:
        return None
    if any(isinstance(v, torch.Tensor) for v in vals):
        return torch.stack([_as_tensor(v) for v in vals]).sum()
    return sum(vals)


def _tmean(*args):
    vals = [a for a in args if a is not None]
    if not vals:
        return None
    if any(isinstance(v, torch.Tensor) for v in vals):
        return torch.stack([_as_tensor(v) for v in vals]).mean()
    return sum(vals) / len(vals)


# torch tensors need ``__import__`` available (for lazy module loads inside
# the comparison/arithmetic dunder methods), so we cannot blank out
# ``__builtins__`` like the Python evaluator does. A trimmed-down import-safe
# builtin dict is the next-best thing.
import builtins as _builtins

_TENSOR_GLOBALS: dict[str, Any] = {"__builtins__": _builtins.__dict__}

_TENSOR_LOCALS: dict[str, Any] = {
    # Arithmetic
    "max": _tmax, "min": _tmin, "abs": _tabs,
    "round": _tround, "int": int, "float": float, "str": str, "len": len,
    "True": True, "False": False, "None": None,
    "log": _tlog, "log2": _tlog2, "log10": _tlog10,
    "exp": _texp, "sqrt": _tsqrt,
    "floor": _tfloor, "ceil": _tceil, "sign": _tsign,
    "mod": lambda a, b: a % b,
    # SAS string functions
    "upcase": lambda s: str(s).upper() if s is not None else None,
    "lowcase": lambda s: str(s).lower() if s is not None else None,
    "strip": lambda s: str(s).strip() if s is not None else None,
    "trim": lambda s: str(s).rstrip() if s is not None else None,
    "substr": lambda s, p, n=None: (str(s)[p - 1:p - 1 + n] if n else str(s)[p - 1:]) if s is not None else None,
    "index": lambda s, sub: (str(s).find(str(sub)) + 1) if s is not None else 0,
    "scan": lambda s, n, d=" ": str(s).split(d)[n - 1] if s is not None and len(str(s).split(d)) >= n else "",
    "compress": lambda s, c="": str(s).replace(c, "") if s is not None else None,
    "cats": lambda *args: "".join(str(a).strip() for a in args if a is not None),
    "cat": lambda *args: "".join(str(a) for a in args if a is not None),
    "catx": lambda sep, *args: sep.join(str(a).strip() for a in args if a is not None),
    "reverse": lambda s: str(s)[::-1] if s is not None else None,
    "repeat": lambda s, n: str(s) * (n + 1) if s is not None else None,
    # SAS date — placeholders
    "today": lambda: 23376, "year": lambda d: 2024, "month": lambda d: 1, "day": lambda d: 1,
    # Logic
    "coalesce": _tcoalesce, "coalescec": _tcoalescec,
    "missing": lambda x: x is None,
    "nmiss": _tnmiss, "n": _tn,
    "ifn": _tifn, "ifc": _tifn,
    "sum": _tsum, "mean": _tmean,
    # Lag — placeholder
    "lag": lambda x, n=1: None, "lag1": lambda x: None, "lag2": lambda x: None,
    # Conversions
    "input": lambda val, fmt=None: float(val) if val is not None else None,
    "put": lambda val, fmt=None: str(val) if val is not None else "",
    "probnorm": lambda x: 0.5,
}


def _mk_lookup(ref_data: dict[str, list[dict]]) -> Any:
    """Build lookup function that searches reference datasets."""
    def _lookup(table_name: str, key_field: str, key_value: Any, return_field: str) -> Any:
        table = ref_data.get(table_name, [])
        for row in table:
            if row.get(key_field) == key_value:
                return row.get(return_field)
        return None
    return _lookup


def _eval_tensor_expr(expr: str, env: dict, is_condition: bool = False) -> Any:
    """Evaluate a SAS expression; numeric vars in ``env`` propagate gradients."""
    py = _to_python(expr, is_condition=is_condition)
    ref_data = env.get("_REFERENCE_DATA", {})
    try:
        return eval(py, _TENSOR_GLOBALS,
                    {**_TENSOR_LOCALS, "_lookup": _mk_lookup(ref_data), **env})  # noqa: S307
    except Exception:
        return False if is_condition else None


# ── Variable extraction ──────────────────────────────────────────────────────

_SKIP_TOKENS = {
    "AND", "OR", "NOT", "EQ", "NE", "GT", "LT", "GE", "LE", "IN",
    "IF", "THEN", "ELSE", "TRUE", "FALSE", "NULL", "MISSING",
    "MAX", "MIN", "ABS", "LOG", "EXP", "SQRT", "FLOOR", "CEIL",
    "ROUND", "INT", "BETWEEN", "LIKE", "IS", "DO", "END", "WHEN", "WHERE",
    "OTHERWISE", "TO", "BY",
}


def _extract_vars(expr: str) -> list[str]:
    """Coarse identifier extraction from an expression."""
    if not expr:
        return []
    cleaned = re.sub(r"'[^']*'", " ", expr)
    cleaned = re.sub(r'"[^"]*"', " ", cleaned)
    cleaned = re.sub(r"&&?\w+\.?", " ", cleaned)
    tokens = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", cleaned)
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        u = t.upper()
        if u in _SKIP_TOKENS or u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


# ── Evaluator ────────────────────────────────────────────────────────────────


_MAX_LOOP_ITERS = 1000


class TensorEvaluator:
    """Walk a SAS AST, mutating ``env`` with torch tensors where possible."""

    def __init__(self) -> None:
        self.branch_log: list[BranchEvent] = []
        self._current_step: str = ""
        self._step_passes: bool = True
        self._row_deleted: bool = False
        self._first_filter_done: bool = False
        self._first_filter_passed: bool = True

    # ── env builders ──────────────────────────────────────────────────────

    @staticmethod
    def make_env(values: dict[str, Any], requires_grad: bool = True) -> dict[str, Any]:
        """Wrap numeric values as float64 tensors; pass strings through."""
        env: dict[str, Any] = {}
        for k, v in values.items():
            ku = k.upper() if isinstance(k, str) else k
            if isinstance(v, bool):
                env[ku] = v
            elif isinstance(v, (int, float)):
                env[ku] = torch.tensor(float(v), dtype=torch.float64, requires_grad=requires_grad)
            elif v is None:
                env[ku] = None
            else:
                env[ku] = v
        return env

    @staticmethod
    def make_path_env(
        values_v2: dict[str, Any],
        values_v3: dict[str, Any],
        t: float,
        branch_row: str = "v3",
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """Build an env on the line segment ``V2 + t·(V3 − V2)`` for numerics.

        Returns a tuple ``(env, knobs)`` where ``knobs`` are the leaf tensors
        with ``requires_grad=True`` — one per numeric field. Categoricals
        take the value of ``branch_row`` (V2 or V3) so the branch path is
        deterministic.
        """
        keys = set(values_v2) | set(values_v3)
        env: dict[str, Any] = {}
        knobs: dict[str, torch.Tensor] = {}
        for k in keys:
            ku = k.upper() if isinstance(k, str) else k
            v2 = values_v2.get(k)
            v3 = values_v3.get(k)
            num_v2 = isinstance(v2, (int, float)) and not isinstance(v2, bool)
            num_v3 = isinstance(v3, (int, float)) and not isinstance(v3, bool)
            if num_v2 and num_v3:
                interp = float(v2) + t * (float(v3) - float(v2))
                tens = torch.tensor(interp, dtype=torch.float64, requires_grad=True)
                env[ku] = tens
                knobs[ku] = tens
            elif num_v3:
                tens = torch.tensor(float(v3), dtype=torch.float64, requires_grad=True)
                env[ku] = tens
                knobs[ku] = tens
            elif num_v2:
                tens = torch.tensor(float(v2), dtype=torch.float64, requires_grad=True)
                env[ku] = tens
                knobs[ku] = tens
            else:
                env[ku] = v3 if branch_row == "v3" and v3 is not None else (v2 if v2 is not None else v3)
        return env, knobs

    # ── Top-level walk ────────────────────────────────────────────────────

    def run(self, nodes: list[AnyNode], env: dict[str, Any]) -> TensorTrace:
        for node in nodes:
            if isinstance(node, DataStepNode):
                self._current_step = node.output_dataset
                self._step_passes = True
                # Set MERGE IN= variables to 1 (assume row is present)
                for body_node in node.body:
                    if isinstance(body_node, MergeNode):
                        for v in body_node.in_vars:
                            env[v] = 1
                self._eval_body(node.body, env)
            elif isinstance(node, (ProcNode, MacroLetNode, MacroDefNode, MacroCallNode)):
                continue
            else:
                self._eval_node(node, env)
        return TensorTrace(
            final=env,
            branch_log=self.branch_log,
            excluded=(self._first_filter_done and not self._first_filter_passed) or self._row_deleted,
        )

    def _eval_body(self, nodes: list[AnyNode], env: dict[str, Any]) -> None:
        for node in nodes:
            if not self._step_passes or self._row_deleted:
                break
            self._eval_node(node, env)

    def _eval_node(self, node: AnyNode, env: dict[str, Any]) -> None:
        if isinstance(node, AssignNode):
            self._eval_assign(node, env)
        elif isinstance(node, IfNode):
            self._eval_if(node, env)
        elif isinstance(node, FilterNode):
            self._eval_filter(node, env)
        elif isinstance(node, DoLoopNode):
            self._eval_do_loop(node, env)
        elif isinstance(node, SelectNode):
            self._eval_select(node, env)
        elif isinstance(node, RetainNode):
            self._eval_retain(node, env)
        elif isinstance(node, DeleteNode):
            self._row_deleted = True

    # ── Per-node evaluators ───────────────────────────────────────────────

    def _eval_assign(self, node: AssignNode, env: dict[str, Any]) -> None:
        new = _eval_tensor_expr(node.expr, env)
        if new is not None:
            env[node.var.upper()] = new

    def _eval_if(self, node: IfNode, env: dict[str, Any]) -> None:
        cond = _eval_tensor_expr(node.condition, env, is_condition=True)
        taken = _to_bool(cond)
        self.branch_log.append(BranchEvent(
            kind="if",
            condition=node.condition,
            taken=taken,
            data_step=self._current_step,
            vars_in_condition=_extract_vars(node.condition),
        ))
        self._eval_body(node.then_branch if taken else node.else_branch, env)

    def _eval_filter(self, node: FilterNode, env: dict[str, Any]) -> None:
        cond = _eval_tensor_expr(node.condition, env, is_condition=True)
        passed = _to_bool(cond)
        self.branch_log.append(BranchEvent(
            kind="where",
            condition=node.condition,
            taken=passed,
            data_step=self._current_step,
            vars_in_condition=_extract_vars(node.condition),
        ))
        if not self._first_filter_done:
            self._first_filter_done = True
            self._first_filter_passed = passed
        if not passed:
            self._step_passes = False

    def _eval_do_loop(self, node: DoLoopNode, env: dict[str, Any]) -> None:
        # Bare DO group
        if not node.var and not node.while_cond and not node.until_cond:
            self._eval_body(node.body, env)
            return
        if node.var:
            start = _eval_tensor_expr(node.start, env)
            stop = _eval_tensor_expr(node.stop, env)
            step = _eval_tensor_expr(node.by_step, env) or 1
            s = _to_float(start, default=0.0)
            e = _to_float(stop, default=0.0)
            st = _to_float(step, default=1.0)
            i = s
            iters = 0
            while ((st > 0 and i <= e) or (st < 0 and i >= e)) and iters < _MAX_LOOP_ITERS:
                if node.while_cond and not _to_bool(_eval_tensor_expr(node.while_cond, env, True)):
                    break
                env[node.var.upper()] = torch.tensor(i, dtype=torch.float64)
                self._eval_body(node.body, env)
                if node.until_cond and _to_bool(_eval_tensor_expr(node.until_cond, env, True)):
                    break
                i += st
                iters += 1
        elif node.while_cond:
            iters = 0
            while iters < _MAX_LOOP_ITERS:
                if not _to_bool(_eval_tensor_expr(node.while_cond, env, True)):
                    break
                self._eval_body(node.body, env)
                iters += 1

    def _eval_select(self, node: SelectNode, env: dict[str, Any]) -> None:
        select_val = _eval_tensor_expr(node.select_expr, env) if node.select_expr else None
        for when in node.whens:
            matched = False
            for v in when.values:
                v_eval = _eval_tensor_expr(v, env, is_condition=not node.select_expr)
                if node.select_expr:
                    a = select_val.item() if isinstance(select_val, torch.Tensor) else select_val
                    b = v_eval.item() if isinstance(v_eval, torch.Tensor) else v_eval
                    if a == b or str(a) == str(b):
                        matched = True
                        break
                else:
                    if _to_bool(v_eval):
                        matched = True
                        break
            if matched:
                self.branch_log.append(BranchEvent(
                    kind="select",
                    condition=", ".join(when.values),
                    taken=True,
                    data_step=self._current_step,
                    vars_in_condition=(
                        _extract_vars(", ".join(when.values))
                        + _extract_vars(node.select_expr or "")
                    ),
                ))
                self._eval_body(when.body, env)
                return
        if node.otherwise:
            self.branch_log.append(BranchEvent(
                kind="select_otherwise",
                condition="OTHERWISE",
                taken=True,
                data_step=self._current_step,
            ))
            self._eval_body(node.otherwise, env)

    def _eval_retain(self, node: RetainNode, env: dict[str, Any]) -> None:
        if node.initial:
            init_val = _eval_tensor_expr(node.initial, env)
            for v in node.vars:
                if v.upper() not in env:
                    env[v.upper()] = init_val


# ── helpers ──────────────────────────────────────────────────────────────────


def _to_bool(x: Any) -> bool:
    if isinstance(x, torch.Tensor):
        return bool(x.detach().reshape(-1)[0].item())
    return bool(x)


def _to_float(x: Any, default: float = 0.0) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().item())
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# ── Convenience ──────────────────────────────────────────────────────────────


def evaluate_tensor(nodes: list[AnyNode], values: dict[str, Any]) -> TensorTrace:
    """One-shot evaluation: wrap values as tensors, walk the AST, return trace."""
    env = TensorEvaluator.make_env(values, requires_grad=True)
    return TensorEvaluator().run(nodes, env)

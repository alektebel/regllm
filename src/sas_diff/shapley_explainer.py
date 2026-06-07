"""Shapley value attribution for SAS-pipeline field discrepancies.

Treats the SAS pipeline as a black-box ``f: row → Y(row)``. For each input
field that differs between V2 and V3, computes the Shapley value of
"flipping that field from V2 to V3" with respect to the target output
``Y``.

Two modes:

- **Exact** (``n_diff_fields ≤ 12``): enumerate all ``2^n`` coalitions.
- **Permutation sampling** otherwise: average ``n_samples`` random orderings.

Both modes satisfy the *efficiency* axiom in expectation:
``Σ_i φ_i = Y(V3) − Y(V2)``.

Categoricals are handled natively (no special encoding) — that is the main
reason for using Shapley on top of the gradient explainer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import combinations
from math import factorial
from typing import Any

from src.sas_logic_tree import AnyNode, SASLogicTree


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass
class ShapAttribution:
    field: str
    delta_input: Any              # numeric: x_v3 − x_v2; categorical: (v2, v3) tuple
    is_numeric: bool
    contribution: float           # Shapley value

    def to_dict(self) -> dict:
        di: Any = self.delta_input
        if isinstance(di, tuple):
            di = list(di)
        return {
            "field": self.field,
            "delta_input": di,
            "is_numeric": self.is_numeric,
            "contribution": self.contribution,
        }


@dataclass
class ShapleyReport:
    target: str
    y_v2: float | None
    y_v3: float | None
    delta_y: float | None
    method: str                   # "exact" | "permutation"
    n_evaluations: int
    attributions: list[ShapAttribution] = field(default_factory=list)
    unexplained_residual: float = 0.0

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "y_v2": self.y_v2,
            "y_v3": self.y_v3,
            "delta_y": self.delta_y,
            "method": self.method,
            "n_evaluations": self.n_evaluations,
            "attributions": [a.to_dict() for a in self.attributions],
            "unexplained_residual": self.unexplained_residual,
        }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _norm(row: dict[str, Any]) -> dict[str, Any]:
    return {k.upper() if isinstance(k, str) else k: v for k, v in row.items()}


def _evaluate(tree: SASLogicTree, nodes: list[AnyNode], target: str, row: dict[str, Any]) -> float | None:
    trace = tree.evaluate(nodes, row)
    y = trace.final.get(target)
    if isinstance(y, bool) or y is None:
        return None
    if isinstance(y, (int, float)):
        return float(y)
    return None


def _coalition_row(base: dict[str, Any], target_row: dict[str, Any], coalition: frozenset[str]) -> dict[str, Any]:
    """Return base updated with target_row's values for keys in ``coalition``."""
    out = dict(base)
    for k in coalition:
        if k in target_row:
            out[k] = target_row[k]
    return out


def _diff_fields(row_v2: dict[str, Any], row_v3: dict[str, Any]) -> list[str]:
    keys = set(row_v2) | set(row_v3)
    diff: list[str] = []
    for k in keys:
        a, b = row_v2.get(k), row_v3.get(k)
        if a is None and b is None:
            continue
        if a != b:
            diff.append(k)
    return sorted(diff)


# ── Main API ─────────────────────────────────────────────────────────────────


def shapley_values(
    nodes: list[AnyNode],
    row_v2: dict[str, Any],
    row_v3: dict[str, Any],
    target: str,
    fields: list[str] | None = None,
    n_samples: int = 256,
    exact_threshold: int = 12,
    seed: int = 0,
) -> ShapleyReport:
    """Compute Shapley attributions for ``Y(V3) − Y(V2)``.

    Parameters
    ----------
    nodes:
        Parsed SAS AST.
    row_v2, row_v3:
        Two row dicts. Keys are case-insensitive.
    target:
        Field name to attribute.
    fields:
        Explicit list of fields to attribute. If ``None``, attributes every
        field whose value differs between V2 and V3.
    n_samples:
        Number of random permutations when falling back to sampling.
    exact_threshold:
        Maximum number of differing fields for which we run the exact
        Shapley enumeration.
    seed:
        RNG seed for the sampling mode.
    """
    target = target.upper()
    v2 = _norm(row_v2)
    v3 = _norm(row_v3)
    tree = SASLogicTree()

    diff = sorted({f.upper() for f in fields}) if fields else _diff_fields(v2, v3)

    # Anchor evaluations
    y_v2 = _evaluate(tree, nodes, target, v2)
    y_v3 = _evaluate(tree, nodes, target, v3)
    delta_y = (y_v3 - y_v2) if (y_v2 is not None and y_v3 is not None) else None

    n = len(diff)
    if n == 0 or y_v2 is None or y_v3 is None:
        return ShapleyReport(
            target=target,
            y_v2=y_v2, y_v3=y_v3, delta_y=delta_y,
            method="exact", n_evaluations=2,
        )

    cache: dict[frozenset, float | None] = {}

    def f(coalition: frozenset[str]) -> float | None:
        if coalition in cache:
            return cache[coalition]
        row = _coalition_row(v2, v3, coalition)
        out = _evaluate(tree, nodes, target, row)
        cache[coalition] = out
        return out

    if n <= exact_threshold:
        method = "exact"
        phi = _shapley_exact(diff, f, n)
    else:
        method = "permutation"
        phi = _shapley_permutation(diff, f, n_samples, seed)

    attributions: list[ShapAttribution] = []
    for k in diff:
        a, b = v2.get(k), v3.get(k)
        is_numeric = (
            isinstance(a, (int, float)) and not isinstance(a, bool)
            and isinstance(b, (int, float)) and not isinstance(b, bool)
        )
        if is_numeric:
            delta_input: Any = float(b) - float(a)
        else:
            delta_input = (a, b)
        attributions.append(ShapAttribution(
            field=k,
            delta_input=delta_input,
            is_numeric=is_numeric,
            contribution=phi.get(k, 0.0),
        ))

    attributions.sort(key=lambda x: abs(x.contribution), reverse=True)

    explained = sum(a.contribution for a in attributions)
    residual = (delta_y or 0.0) - explained

    return ShapleyReport(
        target=target,
        y_v2=y_v2, y_v3=y_v3, delta_y=delta_y,
        method=method,
        n_evaluations=len(cache) + 2,
        attributions=attributions,
        unexplained_residual=residual,
    )


# ── Algorithms ───────────────────────────────────────────────────────────────


def _shapley_exact(
    diff: list[str],
    f,
    n: int,
) -> dict[str, float]:
    """Enumerate all 2^n coalitions and compute the exact Shapley values."""
    phi = {k: 0.0 for k in diff}
    n_fact = factorial(n)
    for size in range(n):
        weight = factorial(size) * factorial(n - size - 1) / n_fact
        for combo in combinations(diff, size):
            S = frozenset(combo)
            fS = f(S)
            if fS is None:
                continue
            for k in diff:
                if k in S:
                    continue
                fS_k = f(S | {k})
                if fS_k is None:
                    continue
                phi[k] += weight * (fS_k - fS)
    return phi


def _shapley_permutation(
    diff: list[str],
    f,
    n_samples: int,
    seed: int,
) -> dict[str, float]:
    """Castro et al. permutation sampling (unbiased estimator of Shapley)."""
    rng = random.Random(seed)
    phi = {k: 0.0 for k in diff}
    counts = {k: 0 for k in diff}
    fields = list(diff)
    for _ in range(n_samples):
        rng.shuffle(fields)
        prev = f(frozenset())
        if prev is None:
            continue
        S: set[str] = set()
        for k in fields:
            S.add(k)
            cur = f(frozenset(S))
            if cur is None:
                continue
            phi[k] += cur - prev
            counts[k] += 1
            prev = cur
    return {k: (phi[k] / counts[k]) if counts[k] else 0.0 for k in diff}

"""Adaptive procedural bug sampling for GRPO.

Adjusts mutation type mix and depth based on recent reward component
failures — easier bugs when the model collapses, harder when it succeeds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from training.curriculum import StageConfig

# Easier → harder
MUTATION_LADDER = [
    "wrong_literal",
    "wrong_op",
    "swap_vars",
    "remove_guard",
    "wrong_agg",
]

COMPONENT_TO_MUTATION_HINT = {
    "correct_step": ["swap_vars", "wrong_op"],
    "correct_var": ["wrong_literal", "wrong_op"],
    "functional_fix": ["remove_guard", "wrong_agg"],
    "trace_path": ["swap_vars", "wrong_op", "remove_guard"],
}


@dataclass
class AdaptiveProfile:
    """Runtime difficulty controller updated from reward traces."""

    difficulty: float = 0.35
    mutation_weights: dict[str, float] = field(default_factory=dict)
    weak_components: dict[str, float] = field(default_factory=dict)
    recent_mean_reward: float = 0.0
    recent_reward_std: float = 0.0
    round_idx: int = 0

    def update(self, traces: list[dict[str, Any]]) -> None:
        """Update profile from a batch of reward breakdowns."""
        if not traces:
            return

        rewards = [t["reward"] for t in traces]
        self.recent_mean_reward = sum(rewards) / len(rewards)
        if len(rewards) > 1:
            mean = self.recent_mean_reward
            self.recent_reward_std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
        else:
            self.recent_reward_std = 0.0

        # Component failure rates (0 = always fails)
        comp_fail: dict[str, list[float]] = {}
        for t in traces:
            for k, v in t.get("components", {}).items():
                comp_fail.setdefault(k, []).append(v)

        self.weak_components = {
            k: sum(vals) / len(vals) for k, vals in comp_fail.items()
        }

        # Difficulty: low reward + low variance → stuck → ease off
        if self.recent_mean_reward < 0.0 and self.recent_reward_std < 0.08:
            self.difficulty = max(0.1, self.difficulty - 0.08)
        elif self.recent_mean_reward > 0.4 and self.recent_reward_std > 0.15:
            self.difficulty = min(0.95, self.difficulty + 0.06)
        elif self.recent_mean_reward > 0.25:
            self.difficulty = min(0.85, self.difficulty + 0.03)
        else:
            self.difficulty = max(0.15, self.difficulty - 0.01)

        # Mutation weights: boost types that train weak components
        self.mutation_weights = {m: 0.2 for m in MUTATION_LADDER}
        for comp, rate in self.weak_components.items():
            if rate >= 0.5:
                continue
            for m in COMPONENT_TO_MUTATION_HINT.get(comp, []):
                self.mutation_weights[m] = self.mutation_weights.get(m, 0.2) + 0.35

        # Harder ladder rungs get weight from difficulty
        cutoff = max(1, int(len(MUTATION_LADDER) * self.difficulty))
        allowed = set(MUTATION_LADDER[:cutoff])
        for m in MUTATION_LADDER:
            if m not in allowed:
                self.mutation_weights[m] = 0.05
            elif m in allowed:
                self.mutation_weights[m] = max(self.mutation_weights.get(m, 0.2), 0.25)

        self.round_idx += 1

    def allowed_mutation_types(self) -> list[str] | None:
        cutoff = max(1, int(len(MUTATION_LADDER) * self.difficulty))
        return MUTATION_LADDER[:cutoff]

    def adaptive_stage(self, stage: StageConfig) -> StageConfig:
        """Return a stage config with adaptive depth/type filters."""
        allowed = self.allowed_mutation_types()
        min_depth = stage.min_depth
        max_depth = stage.max_depth

        if self.difficulty < 0.25:
            max_depth = 0 if max_depth is None else min(max_depth, 0)
        elif self.difficulty > 0.7 and stage.name == "multi_depth":
            min_depth = max(min_depth, 1)

        mutation_types = allowed
        if stage.mutation_types:
            mutation_types = [m for m in allowed if m in stage.mutation_types] or allowed

        return replace(
            stage,
            mutation_types=mutation_types,
            min_depth=min_depth,
            max_depth=max_depth,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "difficulty": round(self.difficulty, 4),
            "mutation_weights": self.mutation_weights,
            "weak_components": {k: round(v, 4) for k, v in self.weak_components.items()},
            "recent_mean_reward": round(self.recent_mean_reward, 4),
            "recent_reward_std": round(self.recent_reward_std, 4),
            "round_idx": self.round_idx,
            "allowed_mutation_types": self.allowed_mutation_types(),
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

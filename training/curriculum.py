"""VibeThinker-style curriculum for SAS debugging RL.

Three stages of increasing difficulty:

Stage 1 — Tool Selection (spectrum):
  The model just needs to call the right tool. Given a question about
  lineage → call trace_dependencies. Given a question about docs →
  call search_docs. Reward: did it call the right tool?

Stage 2 — Single-Depth Debugging (signal):
  The model must call tools AND identify the bug in a single DATA step.
  Mutations are in the final step (depth=0). Reward: did it find the
  right step + variable?

Stage 3 — Multi-Depth Debugging (deep signal):
  The model must trace bugs that propagate through multiple steps.
  Mutations in upstream steps (depth≥1). Reward: full functional
  verification — does the proposed fix restore correct values?

Progression: advance when mean reward over last N episodes > threshold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageConfig:
    """Configuration for one curriculum stage."""
    name: str
    description: str
    max_depth: int | None        # mutation depth limit (None = any)
    min_depth: int               # minimum mutation depth
    reward_weights: dict[str, float]  # how to weight reward components
    advance_threshold: float     # mean reward to advance to next stage
    advance_window: int          # episodes over which to measure
    mutation_types: list[str] | None = None  # None = all types


STAGES: list[StageConfig] = [
    # ── Phase A: Generalized debugging ─────────────────────────────
    StageConfig(
        name="tool_selection",
        description="Learn to call the right tool (trace_dependencies vs search_docs)",
        max_depth=0,
        min_depth=0,
        reward_weights={
            "tool_call": 0.70,       # did it call any tool?
            "correct_tool": 0.30,    # did it call the RIGHT tool?
        },
        advance_threshold=0.6,
        advance_window=20,
        mutation_types=["wrong_literal", "swap_vars", "wrong_op", "remove_guard", "tool_selection"],
    ),
    StageConfig(
        name="single_depth",
        description="Find bugs in the final DATA step (depth=0)",
        max_depth=0,
        min_depth=0,
        reward_weights={
            "tool_call": 0.10,
            "correct_tool": 0.10,
            "trace_path": 0.20,
            "correct_step": 0.20,
            "correct_var": 0.20,
            "no_hallucination": 0.10,
            "functional_fix": 0.10,
        },
        advance_threshold=0.25,
        advance_window=30,
    ),
    StageConfig(
        name="multi_depth",
        description="Trace bugs that propagate through multiple steps (depth≥1)",
        max_depth=None,
        min_depth=1,
        reward_weights={
            "tool_call": 0.10,
            "correct_tool": 0.05,
            "trace_path": 0.30,       # gold BFS path from symptom to root
            "correct_step": 0.15,
            "correct_var": 0.15,
            "no_hallucination": 0.10,
            "functional_fix": 0.15,
        },
        advance_threshold=0.5,
        advance_window=40,
    ),

    # ── Phase B: Experience-accelerated debugging ──────────────────
    StageConfig(
        name="experience_intro",
        description="Multi-depth debugging with search_experience available (gentle bonus)",
        max_depth=None,
        min_depth=0,
        reward_weights={
            "tool_call": 0.10,
            "correct_tool": 0.05,
            "trace_path": 0.25,
            "correct_step": 0.15,
            "correct_var": 0.15,
            "no_hallucination": 0.10,
            "functional_fix": 0.10,
            "experience_used": 0.10,  # gentle intro, not 0.30
        },
        advance_threshold=0.30,
        advance_window=30,
    ),
    StageConfig(
        name="experience_accelerated",
        description="Use experience KB to find bugs faster (query past insights first)",
        max_depth=None,
        min_depth=0,
        reward_weights={
            "tool_call": 0.05,
            "correct_tool": 0.05,
            "correct_step": 0.15,
            "correct_var": 0.15,
            "no_hallucination": 0.10,
            "functional_fix": 0.20,
            "experience_used": 0.30,  # did it query search_experience?
        },
        advance_threshold=0.35,
        advance_window=30,
    ),
    StageConfig(
        name="experience_efficiency",
        description="Same correctness, fewer tool calls — leverage experience for shortcuts",
        max_depth=None,
        min_depth=0,
        reward_weights={
            "correct_step": 0.15,
            "correct_var": 0.15,
            "no_hallucination": 0.10,
            "functional_fix": 0.20,
            "experience_used": 0.20,  # query experience KB
            "efficiency": 0.20,       # fewer tool calls = better
        },
        advance_threshold=0.35,
        advance_window=40,
    ),
]


@dataclass
class CurriculumState:
    """Tracks training progression across stages."""
    current_stage: int = 0
    episode_rewards: list[float] = field(default_factory=list)
    stage_history: list[dict] = field(default_factory=list)

    @property
    def stage(self) -> StageConfig:
        return STAGES[min(self.current_stage, len(STAGES) - 1)]

    @property
    def is_complete(self) -> bool:
        return self.current_stage >= len(STAGES)

    def record_reward(self, reward: float) -> bool:
        """Record a reward and return True if stage should advance."""
        self.episode_rewards.append(reward)
        window = self.stage.advance_window
        if len(self.episode_rewards) < window:
            return False
        recent = self.episode_rewards[-window:]
        mean_reward = sum(recent) / len(recent)
        if mean_reward >= self.stage.advance_threshold:
            self.stage_history.append({
                "stage": self.stage.name,
                "episodes": len(self.episode_rewards),
                "final_mean_reward": round(mean_reward, 4),
            })
            self.current_stage += 1
            self.episode_rewards = []
            return True
        return False

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "current_stage": self.current_stage,
            "episode_rewards": self.episode_rewards,
            "stage_history": self.stage_history,
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CurriculumState:
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(
            current_stage=data["current_stage"],
            episode_rewards=data["episode_rewards"],
            stage_history=data["stage_history"],
        )

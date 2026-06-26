"""RL environment for procedural memory scenarios (sleep cycle).

Mirrors ``rl_env.py`` + ``bug_generator.py`` pattern for learn/organize/retrieve.

Usage:
    from training.memory_rl_env import generate_memory_batch, compute_memory_reward

    batch = generate_memory_batch(16, phase="retrieve")
    for scenario, messages in batch:
        reward = compute_memory_reward(completion, scenario)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.memory_scenario_generator import MemoryPhase, MemoryScenario, MemoryScenarioGenerator
from training.memory_reward import MemoryRewardBreakdown, compute_memory_reward

_SYSTEM = {
    "learn": "Modo sueño: decide qué persistir (save_insight/save_quirk) vs. omitir ruido.",
    "organize": "Modo sueño: organiza el grafo (merge_nodes, add_tags, link_nodes, flag_conflict).",
    "consolidate": (
        "Modo sueño: consolida insights en Concepts (create_concept) para reducir entropía."
    ),
    "retrieve": "Modo sueño: recupera la experiencia exacta (search_experience).",
}


def _messages_for(scenario: MemoryScenario) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM[scenario.phase]},
        {"role": "user", "content": scenario.user_prompt},
    ]


def generate_memory_batch(
    n: int,
    phase: MemoryPhase | None = None,
    *,
    seed: int | None = None,
) -> list[tuple[MemoryScenario, list[dict]]]:
    """Return (scenario, chat_messages) pairs for GRPO / eval."""
    gen = MemoryScenarioGenerator(seed=seed or 42)
    if phase:
        scenarios = [gen.generate_one(phase) for _ in range(n)]
        for i, s in enumerate(scenarios):
            s.scenario_id = f"memrl_{phase}_{i:04d}_{s.subtype}"
    else:
        scenarios = gen.generate_batch(n)
    return [(s, _messages_for(s)) for s in scenarios]


__all__ = [
    "MemoryScenario",
    "MemoryRewardBreakdown",
    "generate_memory_batch",
    "compute_memory_reward",
]

"""Health diagnostics for GRPO RL sessions — collapse, stagnation, weak components."""

from __future__ import annotations

import json
import statistics as stats
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SessionHealth:
    ok: bool = True
    collapsed: bool = False
    stagnant: bool = False
    mean_reward: float = 0.0
    mean_group_std: float = 0.0
    zero_variance_pct: float = 0.0
    weak_components: dict[str, float] = field(default_factory=dict)
    recommendation: str = "continue"
    details: dict[str, Any] = field(default_factory=dict)


def diagnose_session(session_dir: Path) -> SessionHealth:
    """Analyze a completed RL session directory."""
    h = SessionHealth()
    base = Path(session_dir)
    groups_path = base / "group_stats.jsonl"
    metrics_path = base / "metrics.jsonl"
    summary_path = base / "summary.json"

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        h.mean_reward = float(summary.get("mean_reward") or 0)
        h.mean_group_std = float(summary.get("mean_group_std") or 0)
        h.zero_variance_pct = float(summary.get("zero_variance_groups_pct") or 0)
        h.weak_components = summary.get("adaptive_profile", {}).get("weak_components", {})

    groups = _load_jsonl(groups_path)
    metrics = _load_jsonl(metrics_path)

    if groups:
        stds = [g["std"] for g in groups]
        h.mean_group_std = stats.mean(stds)
        h.zero_variance_pct = 100 * sum(1 for s in stds if s < 1e-6) / len(stds)
        if not h.weak_components and len(groups) >= 10:
            comps: dict[str, list[float]] = {}
            for g in groups[-20:]:
                for k, v in g.get("mean_components", {}).items():
                    comps.setdefault(k, []).append(v)
            h.weak_components = {k: stats.mean(v) for k, v in comps.items()}

    if metrics:
        rewards = [m["reward"] for m in metrics if m.get("reward") is not None]
        if rewards:
            q = max(1, len(rewards) // 4)
            first, last = stats.mean(rewards[:q]), stats.mean(rewards[-q:])
            h.details["reward_first_q"] = round(first, 4)
            h.details["reward_last_q"] = round(last, 4)
            h.mean_reward = stats.mean(rewards)
            if last < first - 0.15:
                h.stagnant = True

    # Collapse: most groups identical reward → no GRPO gradient
    h.collapsed = h.zero_variance_pct >= 65.0
    if h.collapsed:
        h.ok = False
        h.recommendation = "rollback_and_sft"
    elif h.stagnant and h.mean_reward < 0.1:
        h.ok = False
        h.recommendation = "sft_warmup"
    elif any(v < 0.25 for k, v in h.weak_components.items() if k not in ("tool_call", "correct_tool", "no_hallucination")):
        h.recommendation = "component_warmup"

    return h


def diagnose_curriculum_state(state_path: Path) -> dict[str, Any]:
    """Quick stats from curriculum_state.json episode tail."""
    if not state_path.exists():
        return {}
    data = json.loads(state_path.read_text())
    tail = data.get("episode_rewards", [])[-40:]
    if not tail:
        return {"n": 0}
    return {
        "n": len(tail),
        "mean": round(sum(tail) / len(tail), 4),
        "std": round(stats.pstdev(tail), 4) if len(tail) > 1 else 0.0,
        "stage": data.get("current_stage"),
    }


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

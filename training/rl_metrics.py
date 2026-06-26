"""Metrics logger for GRPO RL sessions — traces, variance, KL, gradients."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RLSessionLogger:
    """Persists step metrics, group reward variance, and completion traces."""

    session_dir: Path
    num_generations: int = 4
    started_at: float = field(default_factory=time.time)
    step: int = 0
    round_idx: int = 0

    def __post_init__(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "plots").mkdir(exist_ok=True)
        self._metrics_path = self.session_dir / "metrics.jsonl"
        self._groups_path = self.session_dir / "group_stats.jsonl"
        self._traces_path = self.session_dir / "traces.jsonl"
        self._meta_path = self.session_dir / "session_meta.json"

    def write_meta(self, config: dict[str, Any]) -> None:
        self._meta_path.write_text(json.dumps({
            "started_at": datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat(),
            "config": config,
        }, indent=2))

    def record_reward_group(
        self,
        rewards: list[float],
        components: list[dict[str, float]],
        *,
        round_idx: int,
        prompt_key: str,
        mutation_type: str = "",
        step_name: str = "",
    ) -> None:
        """Log within-group reward variance (critical GRPO signal)."""
        if not rewards:
            return
        arr = np.array(rewards, dtype=float)
        entry = {
            "ts": time.time(),
            "round": round_idx,
            "step": self.step,
            "prompt_key": prompt_key[:80],
            "mutation_type": mutation_type,
            "step_name": step_name,
            "n": len(rewards),
            "mean": float(arr.mean()),
            "std": float(arr.std()) if len(arr) > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "range": float(arr.max() - arr.min()),
            "rewards": [round(r, 4) for r in rewards],
            "mean_components": {
                k: round(float(np.mean([c.get(k, 0) for c in components])), 4)
                for k in (components[0].keys() if components else [])
            },
        }
        with self._groups_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def record_trace(
        self,
        *,
        round_idx: int,
        prompt_key: str,
        completion: str,
        reward: float,
        components: dict[str, float],
        mutation_type: str = "",
        bug_step: str = "",
    ) -> None:
        """Save a sampled completion trace."""
        entry = {
            "ts": time.time(),
            "round": round_idx,
            "step": self.step,
            "prompt_key": prompt_key[:80],
            "reward": round(reward, 4),
            "components": components,
            "mutation_type": mutation_type,
            "bug_step": bug_step,
            "completion_preview": completion[:600],
        }
        with self._traces_path.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def record_trainer_log(self, logs: dict[str, Any], *, round_idx: int) -> None:
        """Capture loss, KL, grad_norm from GRPOTrainer.on_log."""
        self.step += 1
        entry = {
            "ts": time.time(),
            "step": self.step,
            "round": round_idx,
            "loss": _f(logs.get("loss")),
            "grad_norm": _f(logs.get("grad_norm")),
            "learning_rate": _f(logs.get("learning_rate")),
            "reward": _f(logs.get("reward")),
            "reward_std": _f(logs.get("reward_std")),
            "kl": _f(logs.get("kl")),
            "entropy": _f(logs.get("entropy")),
            "clip_ratio": _f(logs.get("clip_ratio")),
            "epoch": _f(logs.get("epoch")),
        }
        with self._metrics_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def finalize(self, adaptive_profile: dict | None = None, extra: dict | None = None) -> Path:
        """Write summary + generate plots."""
        metrics = _load_jsonl(self._metrics_path)
        groups = _load_jsonl(self._groups_path)

        summary = {
            "duration_sec": round(time.time() - self.started_at, 1),
            "total_steps": len(metrics),
            "total_groups": len(groups),
            "total_traces": sum(1 for _ in open(self._traces_path)) if self._traces_path.exists() else 0,
        }

        if metrics:
            summary["final_loss"] = metrics[-1].get("loss")
            summary["final_kl"] = metrics[-1].get("kl")
            summary["final_grad_norm"] = metrics[-1].get("grad_norm")
            summary["mean_reward"] = _mean([m["reward"] for m in metrics if m.get("reward") is not None])
            summary["mean_kl"] = _mean([m["kl"] for m in metrics if m.get("kl") is not None])
            summary["mean_grad_norm"] = _mean([m["grad_norm"] for m in metrics if m.get("grad_norm") is not None])

        if groups:
            summary["mean_group_std"] = _mean([g["std"] for g in groups])
            summary["mean_group_range"] = _mean([g["range"] for g in groups])
            summary["zero_variance_groups_pct"] = round(
                100 * sum(1 for g in groups if g["std"] < 1e-6) / len(groups), 1
            )

        if adaptive_profile:
            summary["adaptive_profile"] = adaptive_profile
        if extra:
            summary.update(extra)

        summary_path = self.session_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        try:
            _plot_session(self.session_dir, metrics, groups)
        except Exception as e:
            (self.session_dir / "plots" / "plot_error.txt").write_text(str(e))

        return summary_path


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
        return None if x != x else round(x, 6)  # NaN check
    except (TypeError, ValueError):
        return None


def _mean(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _plot_session(session_dir: Path, metrics: list[dict], groups: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = session_dir / "plots"
    if not metrics and not groups:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"RL Session — {session_dir.name}", fontsize=12, fontweight="bold")

    if metrics:
        steps = [m["step"] for m in metrics]
        for ax, key, title, color in [
            (axes[0, 0], "reward", "Mean Reward (trainer)", "steelblue"),
            (axes[0, 1], "kl", "KL Divergence", "mediumpurple"),
            (axes[1, 0], "grad_norm", "Gradient Norm", "coral"),
            (axes[1, 1], "loss", "Loss", "gray"),
        ]:
            vals = [m.get(key) for m in metrics]
            clean_steps = [s for s, v in zip(steps, vals) if v is not None]
            clean_vals = [v for v in vals if v is not None]
            if clean_vals:
                ax.plot(clean_steps, clean_vals, color=color, lw=1.2, alpha=0.7)
                ax.set_title(title)
                ax.grid(True, alpha=0.2)
                if key in ("kl", "loss", "grad_norm"):
                    ax.set_yscale("log")

    if groups:
        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 4))
        stds = [g["std"] for g in groups]
        ax2.plot(range(len(stds)), stds, color="seagreen", lw=1.0, alpha=0.6)
        ax2.axhline(0.08, color="red", ls="--", alpha=0.5, label="collapse threshold")
        ax2.set_title("Within-Group Reward Std (GRPO variance signal)")
        ax2.set_xlabel("Group index")
        ax2.set_ylabel("Reward std")
        ax2.legend()
        ax2.grid(True, alpha=0.2)
        fig2.tight_layout()
        fig2.savefig(plots_dir / "reward_variance.png", dpi=130)
        plt.close(fig2)

    fig.tight_layout()
    fig.savefig(plots_dir / "training_evolution.png", dpi=130)
    plt.close(fig)

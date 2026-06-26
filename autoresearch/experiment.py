"""Experiment tracker — records research iterations and outcomes.

Each experiment (one research loop iteration) is tracked with:
  - Configuration snapshot
  - Synthetic data generated
  - Training parameters
  - Evaluation results
  - Insights extracted

Stored as local JSON logs, optional MLflow runs, and as Experience/Insight
nodes in the KG.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MLFLOW_CFG = {
    "enabled": True,
    "experiment_name": "regllm-tool-calling",
    "tracking_uri": "sqlite:///autoresearch/mlflow.db",
}


def _load_mlflow_config() -> dict[str, Any]:
    cfg_path = _PROJECT_ROOT / "autoresearch" / "config.yaml"
    if not cfg_path.exists():
        return dict(_DEFAULT_MLFLOW_CFG)
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        return {**_DEFAULT_MLFLOW_CFG, **cfg.get("mlflow", {})}
    except Exception as e:
        logger.warning("Could not load MLflow config: %s", e)
        return dict(_DEFAULT_MLFLOW_CFG)


def log_to_mlflow(
    *,
    run_name: str,
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    artifacts: dict[str, Path | str] | None = None,
) -> str | None:
    """Log one training/eval run to MLflow. Returns run_id or None if disabled."""
    ml_cfg = _load_mlflow_config()
    if not ml_cfg.get("enabled", True):
        return None
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed — skipping experiment log")
        return None

    tracking_uri = ml_cfg.get("tracking_uri", _DEFAULT_MLFLOW_CFG["tracking_uri"])
    if tracking_uri.startswith("sqlite:///"):
        db_rel = tracking_uri.removeprefix("sqlite:///")
        db_path = (_PROJECT_ROOT / db_rel).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    elif tracking_uri.startswith("file:"):
        rel = tracking_uri.removeprefix("file:")
        track_path = (_PROJECT_ROOT / rel).resolve()
        track_path.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(str(track_path))
    else:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(ml_cfg.get("experiment_name", "regllm-tool-calling"))

    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics(metrics)
        if artifacts:
            for name, path in artifacts.items():
                p = Path(path)
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path=name)
        logger.info("MLflow run logged: %s (%s)", run_name, run.info.run_id)
        return run.info.run_id


@dataclass
class ExperimentResult:
    iteration: int
    timestamp: str
    config_snapshot: dict[str, Any]
    synthetic_data: dict[str, int]
    training: dict[str, Any]
    evaluation: dict[str, Any]
    insights: list[dict[str, Any]]
    metrics: dict[str, float]


class ExperimentTracker:
    """Records and manages research experiments."""

    def __init__(self, log_dir: str | Path) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._history: list[ExperimentResult] = self._load_history()

    def _load_history(self) -> list[ExperimentResult]:
        history_path = self._log_dir / "history.json"
        if history_path.exists():
            try:
                raw = json.loads(history_path.read_text())
                return [ExperimentResult(**r) for r in raw]
            except Exception as e:
                logger.warning("Failed to load experiment history: %s", e)
        return []

    def save(self, result: ExperimentResult) -> None:
        self._history.append(result)
        history_path = self._log_dir / "history.json"
        history_path.write_text(
            json.dumps([asdict(r) for r in self._history], indent=2, ensure_ascii=False)
        )
        iter_dir = self._log_dir / f"iter_{result.iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "result.json").write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False)
        )
        logger.info("Experiment %d saved → %s", result.iteration, iter_dir)

        # MLflow (shared with autoresearch loop)
        by_kind = result.evaluation.get("by_kind") or result.metrics.get("by_kind") or {}
        log_to_mlflow(
            run_name=f"iter_{result.iteration:03d}",
            metrics={
                "tool_selection_accuracy": result.metrics.get("tool_selection_accuracy", 0.0) * 100
                if result.metrics.get("tool_selection_accuracy", 0.0) <= 1.0
                else result.metrics.get("tool_selection_accuracy", 0.0),
                **{f"accuracy_{k}": float(v) for k, v in by_kind.items()},
            },
            params={
                "iteration": result.iteration,
                "adapter": result.training.get("adapter", ""),
                "epochs": result.training.get("epochs", ""),
                "n_train_examples": result.synthetic_data.get("train", ""),
            },
            tags={"source": "autoresearch_loop"},
            artifacts={"result": iter_dir / "result.json"},
        )

    @property
    def history(self) -> list[ExperimentResult]:
        return list(self._history)

    @property
    def latest(self) -> ExperimentResult | None:
        return self._history[-1] if self._history else None

    def best_accuracy(self, metric: str = "tool_selection_accuracy") -> float:
        accuracies = [r.metrics.get(metric, 0.0) for r in self._history]
        return max(accuracies) if accuracies else 0.0

    def accuracy_trend(self, metric: str = "tool_selection_accuracy") -> list[float]:
        return [r.metrics.get(metric, 0.0) for r in self._history]

    def summary(self) -> dict[str, Any]:
        if not self._history:
            return {"experiments": 0, "status": "no_experiments"}
        last = self._history[-1]
        return {
            "experiments": len(self._history),
            "last_iteration": last.iteration,
            "last_accuracy": last.metrics.get("tool_selection_accuracy", 0.0),
            "best_accuracy": self.best_accuracy(),
            "trend": self.accuracy_trend(),
        }

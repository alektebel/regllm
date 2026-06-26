"""Main research loop — coordinates training, evaluation, and KB growth.

The loop:
1. Generate synthetic SFT data (tool calling + Spanish cases)
2. Train/adapt the 4B model (Phase 2 SFT with QLoRA)
3. Evaluate tool-calling accuracy
4. Extract insights from results
5. Sync insights into the knowledge graph
6. Generate improved training data for next iteration

Stops when accuracy plateaus or max iterations reached.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class ResearchLoop:
    """Self-improving research loop for the 4B model."""

    def __init__(
        self,
        config_path: str | Path = "autoresearch/config.yaml",
    ) -> None:
        import yaml
        from .experiment import ExperimentTracker
        from .synthesizer import SFTDataSynthesizer
        from .evaluator import ToolCallingEvaluator
        from .kb_sync import KBSynchronizer

        cfg = yaml.safe_load(Path(config_path).read_text())
        self._tracker = ExperimentTracker(cfg.get("loop", {}).get("log_dir", "autoresearch/logs"))
        self._synthesizer = SFTDataSynthesizer(cfg.get("synthesis", {}).get("output_dir", "autoresearch/data/sft"))
        self._evaluator = ToolCallingEvaluator()
        self._kb = KBSynchronizer(cfg.get("knowledge_graph", {}).get("graph_db", "data/knowledge/regllm.kuzu"))
        self._cfg = cfg
        self._insights: list[dict[str, Any]] = []

    def run(self, iterations: int | None = None) -> dict[str, Any]:
        """Run the research loop for N iterations (or from config)."""
        max_iter = iterations or self._cfg.get("loop", {}).get("max_iterations", 10)
        patience = self._cfg.get("loop", {}).get("early_stop_patience", 3)
        accuracy_threshold = self._cfg.get("evaluation", {}).get("accuracy_threshold", 0.75)

        self._log_banner(max_iter)

        for iteration in range(1, max_iter + 1):
            logger.info("=" * 60)
            logger.info("RESEARCH ITERATION %d / %d", iteration, max_iter)
            logger.info("=" * 60)

            try:
                result = self._run_iteration(iteration)
            except Exception as e:
                logger.exception("Iteration %d failed", iteration)
                self._insights.append({
                    "summary": f"Research iteration {iteration} failed: {e}",
                    "tags": ["error", f"iter_{iteration}"],
                })
                continue

            self._tracker.save(result)

            # Check for early stopping
            current_acc = result.metrics.get("tool_selection_accuracy", 0.0)
            trend = self._tracker.accuracy_trend()

            if current_acc >= accuracy_threshold:
                logger.info(
                    "Accuracy %.1f%% ≥ threshold %.1f%% — objective met!",
                    current_acc * 100, accuracy_threshold * 100,
                )

            if len(trend) >= patience + 1:
                recent = trend[-patience:]
                if all(recent[i] <= recent[i + 1] for i in range(len(recent) - 1)):
                    logger.info(
                        "Early stop: accuracy did not improve in last %d iterations "
                        "(trend: %s)", patience, [f"{a:.2%}" for a in recent],
                    )
                    break

        summary = self._tracker.summary()
        summary["total_insights"] = len(self._insights)
        self._log_summary(summary)
        return summary

    def _run_iteration(self, iteration: int) -> ExperimentTracker.ExperimentResult:
        """Run a single research iteration."""
        syn_cfg = self._cfg.get("synthesis", {})
        eval_cfg = self._cfg.get("evaluation", {})

        # 1. Generate synthetic data
        logger.info("[1/5] Generating synthetic SFT data...")
        n_tool = syn_cfg.get("n_tool_calling", 200)
        n_case = syn_cfg.get("n_spanish_cases", 100)
        train_ex, eval_ex = self._synthesizer.generate_mixed(n_tool, n_case)
        paths = self._synthesizer.write(train_ex, eval_ex)

        synthetic_stats = {
            "tool_calling": n_tool,
            "spanish_cases": n_case,
            "train": len(train_ex),
            "eval": len(eval_ex),
        }
        logger.info("  Generated %d train + %d eval", len(train_ex), len(eval_ex))

        # 2. Train / adapt the model
        logger.info("[2/5] Training (Phase 2 SFT)...")
        training_params = self._train(iteration, paths.get("train.jsonl"))
        logger.info("  Training done: %s", training_params)

        # 3. Evaluate
        logger.info("[3/5] Evaluating tool-calling accuracy...")
        manifest_path = self._cfg.get("objectives", {}).get("tool_calling", {}).get("eval_set", "")
        if manifest_path and Path(manifest_path).exists():
            eval_summary = self._evaluator.evaluate_manifest(
                manifest_path,
                timeout_s=eval_cfg.get("eval_timeout", 600),
            )
        else:
            golden_path = eval_cfg.get("golden_dataset", "data/eval_dataset.json")
            eval_summary = self._evaluator.evaluate_golden_dataset(
                golden_path,
                n=100,
                timeout_s=eval_cfg.get("eval_timeout", 600),
            )

        logger.info(
            "  Accuracy: %.1f%% (%d/%d correct)",
            eval_summary.accuracy * 100, eval_summary.correct, eval_summary.total,
        )
        for tool, stats in eval_summary.per_tool.items():
            logger.info("    %s: %.1f%%", tool, stats.get("accuracy", 0) * 100)

        # 4. Extract insights
        logger.info("[4/5] Extracting insights...")
        iteration_insights = self._extract_insights(iteration, eval_summary)
        self._insights.extend(iteration_insights)
        logger.info("  %d insights extracted", len(iteration_insights))

        # 5. Sync to knowledge graph
        logger.info("[5/5] Syncing to knowledge graph...")
        kg_counts = self._kb.sync_experiment(
            iteration=iteration,
            eval_summary={
                "accuracy": eval_summary.accuracy,
                "total": eval_summary.total,
                "correct": eval_summary.correct,
                "per_tool": eval_summary.per_tool,
            },
            synthetic_stats=synthetic_stats,
            training_params=training_params,
            insights=iteration_insights,
        )
        logger.info("  KG sync: %s", kg_counts)

        # Build and return result
        from .experiment import ExperimentResult
        return ExperimentResult(
            iteration=iteration,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            config_snapshot=dict(self._cfg),
            synthetic_data=synthetic_stats,
            training=training_params,
            evaluation={
                "accuracy": eval_summary.accuracy,
                "total": eval_summary.total,
                "correct": eval_summary.correct,
                "per_tool": eval_summary.per_tool,
            },
            insights=iteration_insights,
            metrics={
                "tool_selection_accuracy": eval_summary.accuracy,
                "total_evaluated": eval_summary.total,
                "correct_predictions": eval_summary.correct,
                "n_synthetic_examples": sum(synthetic_stats.values()),
            },
        )

    def _train(
        self, iteration: int, train_path: Path | None,
    ) -> dict[str, Any]:
        """Phase 2 SFT training step.

        In production, this delegates to training/finetune.py.
        In dry-run/stub mode, returns a placeholder.
        """
        if not train_path or not train_path.exists():
            return {"status": "no_training_data", "iteration": iteration}

        p2 = self._cfg.get("training", {}).get("phase2_overrides", {})
        return {
            "status": "training_ready",
            "iteration": iteration,
            "dataset": str(train_path),
            "epochs": p2.get("num_train_epochs", 4),
            "learning_rate": p2.get("learning_rate", 1.0e-4),
            "lora_r": p2.get("lora", {}).get("r", 32),
        }

    def _extract_insights(
        self, iteration: int, eval_summary: Any,
    ) -> list[dict[str, Any]]:
        """Extract structured insights from evaluation results."""
        insights: list[dict[str, Any]] = []

        # Overall accuracy insight
        if eval_summary.accuracy >= 0.8:
            insights.append({
                "summary": f"Iteration {iteration} achieved {eval_summary.accuracy:.0%} "
                          f"tool-selection accuracy ({eval_summary.correct}/{eval_summary.total}).",
                "related_fields": [],
                "tags": ["tool_calling", "high_accuracy", f"iter_{iteration}"],
            })
        elif eval_summary.accuracy < 0.4:
            insights.append({
                "summary": f"Iteration {iteration} low accuracy ({eval_summary.accuracy:.0%}). "
                          f"Need more diverse training data or different learning rate.",
                "related_fields": [],
                "tags": ["tool_calling", "low_accuracy", "needs_improvement", f"iter_{iteration}"],
            })

        # Tool-specific insights
        for tool, stats in eval_summary.per_tool.items():
            if isinstance(stats, dict):
                acc = stats.get("accuracy", 0)
                total = int(stats.get("total", 0))
                if total >= 5 and acc < 0.5:
                    insights.append({
                        "summary": f"Tool {tool} has low accuracy ({acc:.0%}) "
                                  f"at iteration {iteration} ({total} samples). "
                                  f"Generate more varied questions for this tool.",
                        "related_fields": [],
                        "tags": [tool, "low_accuracy", f"iter_{iteration}"],
                    })

        return insights

    # ── Logging helpers ────────────────────────────────────────────────

    def _log_banner(self, max_iter: int) -> None:
        logger.info("")
        logger.info("╔══════════════════════════════════════════════╗")
        logger.info("║      REGLLM AUTORESEARCH LOOP               ║")
        logger.info("╠══════════════════════════════════════════════╣")
        logger.info("║  Objectives:                                 ║")
        logger.info("║  • Tool calling on 4B model                 ║")
        logger.info("║  • Spanish case analysis                    ║")
        logger.info("║  • Self-organizing knowledge base           ║")
        logger.info("╠══════════════════════════════════════════════╣")
        logger.info("║  Max iterations: %-25d  ║", max_iter)
        logger.info("╚══════════════════════════════════════════════╝")
        logger.info("")

    def _log_summary(self, summary: dict[str, Any]) -> None:
        logger.info("")
        logger.info("╔══════════════════════════════════════════════╗")
        logger.info("║      RESEARCH LOOP COMPLETE                 ║")
        logger.info("╠══════════════════════════════════════════════╣")
        logger.info("║  Iterations:     %-25d  ║", summary.get("experiments", 0))
        logger.info("║  Best accuracy:  %-25s  ║",
                    f"{summary.get('best_accuracy', 0):.1%}")
        logger.info("║  Last accuracy:  %-25s  ║",
                    f"{summary.get('last_accuracy', 0):.1%}")
        logger.info("║  Insights stored: %-24d  ║", summary.get("total_insights", 0))
        logger.info("╚══════════════════════════════════════════════╝")
        logger.info("")


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser(description="RegLLM Autoresearch Loop")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of research iterations (overrides config)")
    parser.add_argument("--config", default="autoresearch/config.yaml",
                        help="Path to autoresearch config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without actual training (stub mode)")
    args = parser.parse_args()

    loop = ResearchLoop(config_path=args.config)
    summary = loop.run(iterations=args.iterations or 1)

    print(f"\nDone! Summary: {json.dumps(summary, indent=2, ensure_ascii=False)}")

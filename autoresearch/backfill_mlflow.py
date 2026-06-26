#!/usr/bin/env python3
"""Backfill MLflow with tool-calling SFT results already on disk.

Reads result JSON from training/output/ and data/ — does not re-run eval.

Usage:
    .venv/bin/python autoresearch/backfill_mlflow.py
    .venv/bin/python autoresearch/backfill_mlflow.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autoresearch.experiment import log_to_mlflow


def _load(path: Path) -> dict | list | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def _runs_from_iterate_log(data: list) -> list[dict]:
    runs = []
    for row in data:
        it = row.get("iteration", 0)
        by_kind = row.get("by_kind", {})
        runs.append({
            "run_name": f"tool_sft_iter{it}" if it else "tool_sft_phase2",
            "metrics": {
                "tool_selection_accuracy": float(row["accuracy_pct"]),
                "accuracy_lineage": float(by_kind.get("lineage", 0)),
                "accuracy_rag": float(by_kind.get("rag", 0)),
                "accuracy_both": float(by_kind.get("both", 0)),
                "n_failures": float(row.get("n_failures", 0)),
            },
            "params": {
                "adapter": row.get("adapter", ""),
                "note": row.get("note", ""),
                "iteration": it,
            },
            "tags": {"source": "training_iterate", "stage": "sft"},
            "artifacts": {"iterate_log": PROJECT_ROOT / "training/output/iterate_log.json"},
        })
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    planned: list[dict] = []

    # Baseline (qwen3:4b, prompt-tuned)
    baseline = _load(PROJECT_ROOT / "data/tool_selection_baseline.json")
    if isinstance(baseline, dict) and "accuracy_pct" in baseline:
        planned.append({
            "run_name": "baseline_qwen3_4b",
            "metrics": {
                "tool_selection_accuracy": float(baseline["accuracy_pct"]),
                "accuracy_lineage": float(baseline.get("by_kind", {}).get("lineage", 0)),
                "accuracy_rag": float(baseline.get("by_kind", {}).get("rag", 0)),
                "accuracy_both": float(baseline.get("by_kind", {}).get("both", 0)),
            },
            "params": {"model": baseline.get("model", "qwen3:4b"), "stage": "baseline"},
            "tags": {"source": "eval_tool_selection"},
            "artifacts": {"results": PROJECT_ROOT / "data/tool_selection_baseline.json"},
        })

    # iterate_log.json (phase2-sft + iter1)
    iterate = _load(PROJECT_ROOT / "training/output/iterate_log.json")
    if isinstance(iterate, list):
        for i, row in enumerate(iterate):
            row = dict(row)
            row.setdefault("iteration", i)
            planned.extend(_runs_from_iterate_log([row]))

    # Best adapter pointer
    best = PROJECT_ROOT / "training/output/best_adapter.txt"
    if best.exists():
        planned.append({
            "run_name": "best_adapter_pointer",
            "metrics": {"tool_selection_accuracy": max(
                (r["metrics"]["tool_selection_accuracy"] for r in planned), default=0.0
            )},
            "params": {"best_adapter": best.read_text().strip()},
            "tags": {"source": "best_adapter"},
            "artifacts": {"best_adapter": best},
        })

    print(f"Planned {len(planned)} MLflow runs:")
    for r in planned:
        print(f"  {r['run_name']:30s}  acc={r['metrics'].get('tool_selection_accuracy', 0):.1f}%")

    if args.dry_run:
        return

    run_ids = []
    for r in planned:
        rid = log_to_mlflow(
            run_name=r["run_name"],
            metrics=r["metrics"],
            params=r.get("params"),
            tags=r.get("tags"),
            artifacts=r.get("artifacts"),
        )
        if rid:
            run_ids.append(rid)

    print(f"\nLogged {len(run_ids)} runs to MLflow experiment 'regllm-tool-calling'")
    print(f"  tracking: {PROJECT_ROOT / 'autoresearch/mlflow.db'}")
    print("  view:     mlflow ui --backend-store-uri sqlite:///autoresearch/mlflow.db")


if __name__ == "__main__":
    main()

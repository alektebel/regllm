#!/usr/bin/env python3
"""Train Phase 2 (sleep / self-organizing KB) and sync research results into the KG.

Uses existing datasets:
  - data/memory_eval_dataset.json
  - data/experience_eval_dataset.json
  - data/fragile_retrieval_dataset.json

Built via training/build_phase_datasets.py → phase2.1 / 2.2 / 2.3 JSONL.

Usage:
    .venv/bin/python autoresearch/run_kb_training.py
    .venv/bin/python autoresearch/run_kb_training.py --skip-train  # KG sync only
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autoresearch.experiment import log_to_mlflow
from autoresearch.kb_sync import KBSynchronizer

KB_DATA = PROJECT_ROOT / "autoresearch" / "data" / "kb"
PHASE_DIR = PROJECT_ROOT / "training" / "data"
DEFAULT_ADAPTER = PROJECT_ROOT / "training/output/tool-sft-iter1"
ITERATE_LOG = PROJECT_ROOT / "training/output/iterate_log.json"


def _free_gpu() -> None:
    import os
    import signal

    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/ps", timeout=5)
        for m in r.json().get("models", []):
            name = m.get("name", "")
            if name:
                httpx.post(
                    "http://localhost:11434/api/generate",
                    json={"model": name, "keep_alive": 0},
                    timeout=10,
                )
                print(f"  unloaded ollama model: {name}", flush=True)
    except Exception as e:
        print(f"  (ollama unload skipped: {e})", flush=True)

    my_pid = os.getpid()
    try:
        import subprocess as sp
        out = sp.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"],
            text=True,
        )
        for line in out.strip().splitlines():
            pid_s, mem = [p.strip() for p in line.split(",")]
            pid = int(pid_s)
            if pid == my_pid:
                continue
            print(f"  stopping GPU process pid={pid} ({mem}) to free VRAM", flush=True)
            os.kill(pid, signal.SIGTERM)
    except Exception as e:
        print(f"  (GPU cleanup skipped: {e})", flush=True)


def _rebuild_phase_datasets() -> None:
    print(">>> Rebuilding phase datasets...", flush=True)
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "training/build_phase_datasets.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _merge_phase2_jsonl(out_train: Path, out_eval: Path, eval_ratio: float = 0.12) -> dict:
    """Combine phase 2.1 + 2.2 + 2.3 into train/eval JSONL."""
    files = [
        PHASE_DIR / "phase2.1_learning.jsonl",
        PHASE_DIR / "phase2.2_organizing.jsonl",
        PHASE_DIR / "phase2.3_retrieval.jsonl",
    ]
    rows: list[dict] = []
    counts: dict[str, int] = {}
    for f in files:
        if not f.exists():
            continue
        phase = f.stem.split("_")[0].replace("phase", "2.")
        with open(f) as fh:
            for line in fh:
                ex = json.loads(line)
                rows.append(ex)
                p = ex.get("phase", phase)
                counts[p] = counts.get(p, 0) + 1

    rng = random.Random(42)
    rng.shuffle(rows)
    n_eval = max(1, int(len(rows) * eval_ratio))
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]

    out_train.parent.mkdir(parents=True, exist_ok=True)
    for path, data in [(out_train, train_rows), (out_eval, eval_rows)]:
        with open(path, "w", encoding="utf-8") as fh:
            for ex in data:
                fh.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")

    return {"total": len(rows), "train": len(train_rows), "eval": len(eval_rows), "by_phase": counts}


def _sync_tool_calling_to_kg(kb: KBSynchronizer) -> int:
    """Backfill KG with tool-calling SFT experiment results."""
    if not ITERATE_LOG.exists():
        print("  (no iterate_log.json — skipping KB backfill)", flush=True)
        return 0
    data = json.loads(ITERATE_LOG.read_text())
    total_nodes = 0
    for i, row in enumerate(data):
        iteration = row.get("iteration", i)
        acc = float(row.get("accuracy_pct", 0)) / 100.0
        by_kind = row.get("by_kind", {})
        per_tool = {
            "trace_dependencies": {"accuracy": by_kind.get("lineage", 0) / 100, "total": 10},
            "search_regulation": {"accuracy": by_kind.get("rag", 0) / 100, "total": 10},
        }
        insights = [{
            "summary": row.get("note", f"Tool-calling iter {iteration}: {acc:.0%} accuracy"),
            "tags": ["tool_calling", f"iter_{iteration}"],
            "related_fields": [],
        }]
        if acc < 0.9:
            insights.append({
                "summary": f"At iter {iteration}, dual-tool (both) accuracy was "
                           f"{by_kind.get('both', 0)}% — needs multi-step training.",
                "tags": ["both", "multi_tool", "needs_improvement"],
            })
        counts = kb.sync_experiment(
            iteration=100 + iteration,  # offset from live research iters
            eval_summary={"accuracy": acc, "per_tool": per_tool, "total": 74, "correct": int(acc * 74)},
            synthetic_stats={"train": 254, "eval": 30},
            training_params={"adapter": row.get("adapter", ""), "note": row.get("note", "")},
            insights=insights,
        )
        total_nodes += counts.get("nodes", 0)
        kb.store_model_artifact(
            iteration=100 + iteration,
            artifact_path=row.get("adapter", ""),
            artifact_type="qlora_adapter",
            metrics={"tool_selection_accuracy": acc * 100},
        )
    return total_nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 KB sleep training + KG sync")
    parser.add_argument("--adapter-in", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--output", default="autoresearch/output/kb-sleep-sft")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("RegLLM — Phase 2: Self-organizing KB training")
    print("=" * 60)

    _rebuild_phase_datasets()

    train_path = KB_DATA / "kb_sleep_train.jsonl"
    eval_path = KB_DATA / "kb_sleep_eval.jsonl"
    stats = _merge_phase2_jsonl(train_path, eval_path)
    print(f"KB sleep dataset: {stats}")

    print("\n>>> Syncing tool-calling results into knowledge graph...", flush=True)
    kb = KBSynchronizer()
    n_kb = _sync_tool_calling_to_kg(kb)
    print(f"  KG nodes synced: {n_kb}")

    adapter_out = args.output
    if not args.skip_train:
        print("\n>>> Freeing GPU...", flush=True)
        _free_gpu()

        print(f">>> SFT on KB sleep data (base={args.adapter_in})...", flush=True)
        from training.sft_run import run_sft
        adapter_out = run_sft(
            adapter_in=args.adapter_in if Path(args.adapter_in).exists() else None,
            output_dir=adapter_out,
            train_file=str(train_path.relative_to(PROJECT_ROOT)),
            eval_file=str(eval_path.relative_to(PROJECT_ROOT)),
            epochs=args.epochs,
            max_seq_length=1536,
            eval_strategy="no",
        )

    log_to_mlflow(
        run_name="kb_sleep_phase2",
        metrics={
            "kb_train_examples": float(stats["train"]),
            "kb_eval_examples": float(stats["eval"]),
            "kb_phase2_1_count": float(stats["by_phase"].get("2.1", 0)),
            "kb_phase2_2_count": float(stats["by_phase"].get("2.2", 0)),
            "kb_phase2_3_count": float(stats["by_phase"].get("2.3", 0)),
            "kg_nodes_synced": float(n_kb),
        },
        params={
            "adapter_in": args.adapter_in,
            "adapter_out": adapter_out,
            "epochs": args.epochs,
            "phase": "2_sleep_kb",
        },
        tags={"source": "autoresearch_kb", "stage": "phase2"},
        artifacts={
            "kb_train": train_path,
            "iterate_log": ITERATE_LOG,
        },
    )

    kb.store_model_artifact(
        iteration=200,
        artifact_path=adapter_out,
        artifact_type="kb_sleep_qlora",
        metrics={"kb_train_examples": stats["train"]},
    )

    print("\nDone.")
    print(f"  KB adapter: {adapter_out}")
    print(f"  KG DB:      data/knowledge/regllm.kuzu")
    print(f"  MLflow:     experiment regllm-tool-calling")


if __name__ == "__main__":
    main()

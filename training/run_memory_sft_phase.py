#!/usr/bin/env python3
"""Memory SFT phase — run AFTER reasoning/tool-calling SFT succeeds.

Builds procedural memory data (learn/organize/consolidate/retrieve) and
fine-tunes on top of the reasoning adapter.

Usage:
    # 1. Build datasets (includes EverMemOS-style consolidate phase)
    python training/run_memory_sft_phase.py --build-only

    # 2. Train on top of reasoning adapter (local or Vast)
    python training/run_memory_sft_phase.py \\
        --adapter-in training/output/phase2-qwen3-8b-sft \\
        --output training/output/memory-qwen3-8b-sft \\
        --min-tool-accuracy 0.90

    # 3. Upload adapter to Vast (after training)
    scp -i ~/.ssh/regllm_vast2 -P PORT -r training/output/memory-qwen3-8b-sft root@HOST:/workspace/regllm/training/output/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _run(cmd: list[str]) -> None:
    print(f"\n→ {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _latest_tool_accuracy() -> float | None:
    """Read tool accuracy from iterate_log or eval artifacts if present."""
    for path in (
        PROJECT_ROOT / "training/output/iterate_log.json",
        PROJECT_ROOT / "data/eval_results_v2.json",
        PROJECT_ROOT / "data/eval_results.json",
    ):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                data = data[-1]
            for key in ("tool_accuracy", "tool_recall", "accuracy"):
                if key in data:
                    return float(data[key])
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Memory SFT after reasoning phase")
    p.add_argument("--build-only", action="store_true", help="Only regenerate memory JSONL")
    p.add_argument("--n", type=int, default=480, help="Procedural scenarios to generate")
    p.add_argument("--adapter-in", default="training/output/phase2-qwen3-8b-sft")
    p.add_argument("--output", default="training/output/memory-qwen3-8b-sft")
    p.add_argument("--min-tool-accuracy", type=float, default=0.90,
                   help="Gate: skip SFT if reasoning eval below this (None to skip gate)")
    p.add_argument("--skip-gate", action="store_true", help="Train even if eval unknown/low")
    p.add_argument("--epochs", type=int, default=2)
    args = p.parse_args()

    _run([
        sys.executable, "training/build_memory_sft.py",
        "--n", str(args.n),
        "--rl-prompts",
    ])

    if args.build_only:
        print("Build complete.")
        return

    if not args.skip_gate and args.min_tool_accuracy is not None:
        acc = _latest_tool_accuracy()
        if acc is None:
            print(
                "WARNING: no reasoning eval found — use --skip-gate to train anyway, "
                "or run tool eval first."
            )
            sys.exit(1)
        if acc < args.min_tool_accuracy:
            print(
                f"Reasoning tool accuracy {acc:.1%} < {args.min_tool_accuracy:.1%} — "
                "fix reasoning before memory SFT. Use --skip-gate to override."
            )
            sys.exit(1)
        print(f"Reasoning gate passed: tool accuracy {acc:.1%}")

    adapter = PROJECT_ROOT / args.adapter_in
    if not adapter.exists():
        print(f"Adapter not found: {adapter}")
        sys.exit(1)

    _run([
        sys.executable, "training/sft_run.py",
        "--adapter-in", str(adapter.relative_to(PROJECT_ROOT)),
        "--output", args.output,
        "--train-file", "training/data/memory_sft_proc.jsonl",
        "--eval-file", "training/data/phase2.4_consolidating_proc.jsonl",
        "--epochs", str(args.epochs),
    ])
    print(f"\nDone. Memory adapter: {args.output}")
    print("Upload to Vast when ready:")
    print(f"  scp -r {args.output} root@VAST:/workspace/regllm/training/output/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train Olmo3-7B and Qwen3-8B adapters, then eval tool-selection accuracy.

Usage:
    .venv/bin/python training/train_midsize.py --free-gpu
    .venv/bin/python training/train_midsize.py --models olmo3-7b --epochs 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODELS = ["olmo3-7b", "qwen3-8b"]
RESULTS = PROJECT_ROOT / "training/output/midsize_compare.json"


def _free_gpu() -> None:
    try:
        import httpx
        for m in httpx.get("http://localhost:11434/api/ps", timeout=5).json().get("models", []):
            name = m.get("name", "")
            if name:
                httpx.post(
                    "http://localhost:11434/api/generate",
                    json={"model": name, "keep_alive": 0},
                    timeout=10,
                )
                print(f"  unloaded ollama: {name}", flush=True)
    except Exception as e:
        print(f"  (ollama cleanup: {e})", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--free-gpu", action="store_true")
    args = p.parse_args()
    py = sys.executable

    if not args.skip_data:
        print(">>> build_tool_sft", flush=True)
        subprocess.run([py, "training/build_tool_sft.py"], cwd=PROJECT_ROOT, check=True)
        print(">>> build_agent_sft", flush=True)
        subprocess.run([py, "training/build_agent_sft.py"], cwd=PROJECT_ROOT, check=True)

    if args.skip_train:
        return

    if args.free_gpu:
        _free_gpu()

    from training.sft_run import run_sft
    from training.tool_utils import eval_adapter

    report: dict = {"models": {}}
    for key in args.models:
        print(f"\n{'='*60}\nTRAIN {key}\n{'='*60}", flush=True)
        if args.free_gpu and key != args.models[0]:
            _free_gpu()
        out = run_sft(
            key,
            train_file="training/data/agent_sft_train.jsonl",
            eval_file="training/data/agent_sft_eval.jsonl",
            epochs=args.epochs,
        )
        print(f"\n>>> eval {key}", flush=True)
        result = eval_adapter(out, verbose=True)
        report["models"][key] = {"adapter": out, **result}
        print(f"\n{key}: {result['accuracy_pct']}%  by_kind={result['by_kind']}", flush=True)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n→ {RESULTS}")


if __name__ == "__main__":
    main()

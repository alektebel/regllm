#!/usr/bin/env python3
"""Automatically iterate: augment data → SFT → eval until tool accuracy target.

Usage:
    .venv/bin/python training/iterate_tool_calling.py --target 95 --max-iters 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.tool_utils import eval_adapter

LOG = PROJECT_ROOT / "training/output/iterate_log.json"
FAILURES = PROJECT_ROOT / "training/output/last_failures.json"


def _free_gpu() -> None:
    """Unload Ollama models so training fits in VRAM."""
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/ps", timeout=5)
        for m in r.json().get("models", []):
            name = m.get("name", "")
            if name:
                httpx.post("http://localhost:11434/api/generate",
                           json={"model": name, "keep_alive": 0}, timeout=10)
                print(f"  unloaded {name}", flush=True)
    except Exception as e:
        print(f"  (could not unload ollama: {e})", flush=True)
    time.sleep(15)


def _run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, default=95.0)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--base-adapter", default="training/output/phase2-sft")
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    py = str(PROJECT_ROOT / ".venv/bin/python")
    history: list[dict] = []
    adapter = args.base_adapter if Path(PROJECT_ROOT / args.base_adapter).exists() else None
    failure_file = ""

    for it in range(args.max_iters):
        print(f"\n{'='*60}\nITERATION {it+1}/{args.max_iters}\n{'='*60}", flush=True)

        build_cmd = [py, "training/build_tool_sft.py", "--seed", str(42 + it)]
        if failure_file:
            build_cmd += ["--failure-file", failure_file, "--failure-repeats", "10"]
        _run(build_cmd)

        print(">>> Freeing GPU (unload Ollama models)...", flush=True)
        _free_gpu()

        out_dir = f"training/output/tool-sft-iter{it+1}"
        from training.sft_run import run_sft
        adapter = run_sft(
            adapter_in=adapter,
            output_dir=out_dir,
            epochs=args.epochs + it,  # more epochs each round
        )

        print("\n>>> Evaluating...", flush=True)
        result = eval_adapter(adapter, verbose=False)
        result["iteration"] = it + 1
        result["adapter"] = adapter
        history.append(result)

        print(f"\nAccuracy: {result['accuracy_pct']}%  by_kind={result['by_kind']}", flush=True)

        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(json.dumps(history, indent=2, ensure_ascii=False))

        if result["accuracy_pct"] >= args.target:
            print(f"\n✓ Target {args.target}% reached!", flush=True)
            break

        failures = result.get("failures", [])
        FAILURES.write_text(json.dumps(failures, indent=2, ensure_ascii=False))
        failure_file = str(FAILURES)
        print(f"  {len(failures)} failures saved for next iteration", flush=True)
        time.sleep(2)
    else:
        print(f"\nStopped after {args.max_iters} iterations. Best: {max(h['accuracy_pct'] for h in history)}%", flush=True)

    best = max(history, key=lambda h: h["accuracy_pct"])
    print(f"\nBest: {best['accuracy_pct']}% @ iter {best['iteration']}  adapter={best['adapter']}")
    (PROJECT_ROOT / "training/output/best_adapter.txt").write_text(best["adapter"])


if __name__ == "__main__":
    main()

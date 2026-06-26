#!/usr/bin/env python3
"""Qwen3-8B: SFT → curricular GRPO on rl_env (AgentWorld / Qwen-style loop).

Pipeline:
  1. Build agent SFT data (tool traces + golden)
  2. QLoRA SFT on unsloth/Qwen3-8B-Instruct
  3. Tool-selection eval
  4. iterate_rl: oracle warmup → GRPO on procedural SAS bugs → eval

Usage:
    .venv/bin/python training/run_qwen3_rl_pipeline.py --skip-rl
    .venv/bin/python training/run_qwen3_rl_pipeline.py --skip-sft --max-iters 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_KEY = "qwen3-8b"
SFT_OUT = PROJECT_ROOT / "training/output/sft-qwen3-8b"
GRPO_OUT = PROJECT_ROOT / "training/output/grpo-qwen3-8b"
CURRICULUM = PROJECT_ROOT / "training/output/curriculum_state_qwen3.json"
LOG = PROJECT_ROOT / "training/output/qwen3_rl_pipeline.json"


def _py() -> str:
    return sys.executable


def _run(cmd: list[str], *, check: bool = True) -> int:
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=check).returncode


def _free_gpu() -> None:
    try:
        import httpx
        for m in httpx.get("http://localhost:11434/api/ps", timeout=5).json().get("models", []):
            if m.get("name"):
                httpx.post("http://localhost:11434/api/generate",
                           json={"model": m["name"], "keep_alive": 0}, timeout=10)
    except Exception:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-8B SFT + RL pipeline")
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--skip-sft", action="store_true")
    p.add_argument("--skip-rl", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-iters", type=int, default=3, help="iterate_rl GRPO rounds")
    p.add_argument("--session-minutes", type=float, default=90.0)
    p.add_argument("--reset-curriculum", action="store_true", default=True)
    p.add_argument("--no-reset-curriculum", dest="reset_curriculum", action="store_false")
    p.add_argument("--free-gpu", action="store_true")
    args = p.parse_args()
    py = _py()
    report: dict = {"model_key": MODEL_KEY}

    if not args.skip_data:
        _run([py, "training/build_tool_sft.py"])
        _run([py, "training/build_agent_sft.py"])

    if args.free_gpu:
        _free_gpu()

    if not args.skip_sft:
        adapter = SFT_OUT
        from training.sft_run import run_sft
        run_sft(
            MODEL_KEY,
            str(SFT_OUT),
            train_file="training/data/agent_sft_train.jsonl",
            eval_file="training/data/agent_sft_eval.jsonl",
            epochs=args.epochs,
        )
        from training.tool_utils import eval_adapter
        ev = eval_adapter(adapter)
        report["sft_eval"] = {k: v for k, v in ev.items() if k != "failures"}
        print(f"\nSFT tool accuracy: {ev['accuracy_pct']}%", flush=True)
    elif SFT_OUT.exists():
        adapter = SFT_OUT
    else:
        raise FileNotFoundError(f"No SFT adapter at {SFT_OUT} — run without --skip-sft")

    if args.skip_rl:
        LOG.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nDone (SFT only). Adapter → {SFT_OUT}")
        return

    if args.reset_curriculum and CURRICULUM.exists():
        CURRICULUM.unlink()

    rc = _run([
        py, "training/iterate_rl.py",
        "--max-iters", str(args.max_iters),
        "--session-minutes", str(args.session_minutes),
        "--base-adapter", str(SFT_OUT if not args.skip_sft else adapter),
        "--grpo-dir", str(GRPO_OUT),
        "--curriculum-state", str(CURRICULUM),
    ], check=False)
    report["iterate_rl_rc"] = rc
    report["sft_adapter"] = str(SFT_OUT if SFT_OUT.exists() else adapter)
    report["grpo_dir"] = str(GRPO_OUT)
    LOG.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPipeline log → {LOG}")


if __name__ == "__main__":
    main()

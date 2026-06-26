#!/usr/bin/env python3
"""Validate GRPO pipeline: reward sanity, pre/post eval, session health.

Usage:
    .venv/bin/python training/validate_rl.py
    .venv/bin/python training/validate_rl.py --adapter training/output/sft-olmo3-7b --grpo-minutes 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SESSIONS = PROJECT_ROOT / "training/output/rl_sessions"
REPORT = PROJECT_ROOT / "training/output/rl_validation_report.json"


def _free_gpu() -> None:
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        import httpx
        for m in httpx.get("http://localhost:11434/api/ps", timeout=5).json().get("models", []):
            name = m.get("name", "")
            if name:
                httpx.post("http://localhost:11434/api/generate",
                           json={"model": name, "keep_alive": 0}, timeout=10)
    except Exception:
        pass


def _eval_stage(adapter: Path, stage_idx: int, n: int = 12) -> dict:
    from training.curriculum import STAGES
    from training.rl_env import generate_training_batch, compute_reward
    from training.tool_utils import load_adapter, release_gpu

    manifest_path = PROJECT_ROOT / "training/data/rl_eval_manifest.jsonl"
    if manifest_path.exists():
        from training.rl_eval import eval_adapter_manifest
        return eval_adapter_manifest(
            adapter, manifest_path, stage_idx=stage_idx,
            temperature=0.0, max_new_tokens=400,
        )

    stage = STAGES[stage_idx]
    model, tokenizer = load_adapter(str(adapter), max_seq_length=4096)
    batch = generate_training_batch(n, stage, seed=42)

    import torch
    rewards = []
    for bug, messages in batch:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=400, temperature=0.0, do_sample=False)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        bd = compute_reward(text, bug, stage)
        rewards.append(bd.total)

    del model
    torch.cuda.empty_cache()
    mean = sum(rewards) / max(len(rewards), 1)
    return {"mean_reward": round(mean, 4), "n": len(rewards), "rewards": [round(r, 3) for r in rewards]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="training/output/sft-olmo3-7b")
    p.add_argument("--stage", type=int, default=3)
    p.add_argument("--grpo-minutes", type=float, default=8.0)
    p.add_argument("--skip-grpo", action="store_true")
    args = p.parse_args()
    py = sys.executable
    adapter = PROJECT_ROOT / args.adapter

    from training.rl_health import diagnose_session, diagnose_curriculum_state
    from training.curriculum import CurriculumState

    report: dict = {"ts": datetime.now(tz=timezone.utc).isoformat(), "adapter": str(adapter)}

    # 1) Reward function sanity
    print(">>> [1/4] Reward function eval-only", flush=True)
    r = subprocess.run([py, "training/train_rl.py", "--eval-only"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    report["reward_eval_rc"] = r.returncode
    report["reward_eval_ok"] = r.returncode == 0 and "good: reward=" in r.stdout
    if not report["reward_eval_ok"]:
        print(r.stderr[-500:] if r.stderr else r.stdout[-500:])

    # 2) Historical session health
    print(">>> [2/4] Session health diagnosis", flush=True)
    sessions = sorted(SESSIONS.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]
    report["sessions"] = {}
    for sd in sessions:
        h = diagnose_session(sd)
        report["sessions"][sd.name] = {
            "ok": h.ok,
            "collapsed": h.collapsed,
            "mean_reward": h.mean_reward,
            "zero_variance_pct": h.zero_variance_pct,
            "recommendation": h.recommendation,
            "weak_components": h.weak_components,
        }
        print(f"  {sd.name}: ok={h.ok} collapsed={h.collapsed} "
              f"mean_r={h.mean_reward:.3f} zero_var={h.zero_variance_pct:.0f}% → {h.recommendation}")

    report["curriculum"] = diagnose_curriculum_state(PROJECT_ROOT / "training/output/curriculum_state.json")

    # 3) Pre-eval on adapter
    print(f">>> [3/4] Pre-eval stage {args.stage} on {adapter.name}", flush=True)
    _free_gpu()
    pre = _eval_stage(adapter, args.stage, n=12)
    report["pre_eval"] = pre
    print(f"  pre mean_reward={pre['mean_reward']}", flush=True)

    # 4) Short GRPO smoke test
    if not args.skip_grpo:
        session = f"validate_{adapter.name}_{datetime.now(tz=timezone.utc).strftime('%H%M')}"
        print(f">>> [4/4] GRPO smoke test ({args.grpo_minutes} min) → {session}", flush=True)
        _free_gpu()
        rc = subprocess.run([
            py, "training/train_rl.py",
            "--base-adapter", str(adapter),
            "--output", str(PROJECT_ROOT / "training/output/grpo-validate"),
            "--batch-size", "6",
            "--num-generations", "4",
            "--max-completion-length", "384",
            "--max-minutes", str(args.grpo_minutes),
            "--adaptive-bugs",
            "--session-name", session,
        ], cwd=PROJECT_ROOT)
        report["grpo_rc"] = rc.returncode
        sd = SESSIONS / session
        if sd.exists():
            h = diagnose_session(sd)
            report["grpo_health"] = h.__dict__
            summary_path = sd / "summary.json"
            if summary_path.exists():
                report["grpo_summary"] = json.loads(summary_path.read_text())
            print(f"  GRPO health: ok={h.ok} collapsed={h.collapsed} "
                  f"zero_var={h.zero_variance_pct:.0f}% mean_group_std={h.mean_group_std:.4f}")
        _free_gpu()
        post = _eval_stage(adapter, args.stage, n=12)
        report["post_eval"] = post
        report["delta"] = round(post["mean_reward"] - pre["mean_reward"], 4)
        print(f"  post mean_reward={post['mean_reward']}  delta={report['delta']}", flush=True)

    # Verdict
    grpo_ok = report.get("grpo_health", {}).get("ok", True)
    grpo_collapsed = report.get("grpo_health", {}).get("collapsed", False)
    delta = report.get("delta", 0)
    if report["reward_eval_ok"] and grpo_ok and not grpo_collapsed and delta >= -0.05:
        report["verdict"] = "RL_PIPELINE_OK"
    elif grpo_collapsed:
        report["verdict"] = "GRPO_COLLAPSED"
    elif delta < -0.15:
        report["verdict"] = "GRPO_REGRESSED"
    else:
        report["verdict"] = "NEEDS_ATTENTION"

    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== VERDICT: {report['verdict']} ===")
    print(f"Report → {REPORT}")


if __name__ == "__main__":
    main()

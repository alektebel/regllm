#!/usr/bin/env python3
"""Automatic RL iteration loop — AgentWorld-style SFT warm-up → GRPO → self-distill.

Inspired by Qwen-AgentWorld (arxiv:2606.24597):
  - Simulated environment RL with rule-based verifiers
  - SFT activates behavior patterns before RL sharpens them
  - Online self-distillation from good rollouts
  - Autoresearch: diagnose → intervene → train → evaluate → repeat

Usage:
    .venv/bin/python training/iterate_rl.py
    .venv/bin/python training/iterate_rl.py --max-iters 10 --session-minutes 45
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG = PROJECT_ROOT / "training/output/rl_iterate_log.json"
CURRICULUM = PROJECT_ROOT / "training/output/curriculum_state.json"
GRPO_DIR = PROJECT_ROOT / "training/output/grpo-curriculum"
SESSIONS = PROJECT_ROOT / "training/output/rl_sessions"
BASE_ADAPTER = PROJECT_ROOT / "training/output/tool-sft-antihalluc"


def _free_gpu() -> None:
    try:
        import httpx
        for m in httpx.get("http://localhost:11434/api/ps", timeout=5).json().get("models", []):
            name = m.get("name", "")
            if name:
                httpx.post("http://localhost:11434/api/generate",
                           json={"model": name, "keep_alive": 0}, timeout=10)
                print(f"  unloaded {name}", flush=True)
    except Exception as e:
        print(f"  (gpu cleanup: {e})", flush=True)
    time.sleep(5)


def _run(cmd: list[str], timeout: float | None = None) -> int:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=PROJECT_ROOT, timeout=timeout)
    return r.returncode


def _current_adapter() -> Path:
    from training.curriculum import CurriculumState, STAGES
    state = CurriculumState.load(CURRICULUM)
    if state.is_complete:
        return GRPO_DIR / "final"
    stage = STAGES[state.current_stage]
    ckpt = GRPO_DIR / f"stage_{state.current_stage}_{stage.name}"
    if ckpt.exists() and (ckpt / "adapter_config.json").exists():
        return ckpt
    return BASE_ADAPTER


def _save_best(stage_idx: int, adapter: Path, score: float) -> None:
    best_dir = GRPO_DIR / f"best_stage{stage_idx}"
    meta = GRPO_DIR / f"best_stage{stage_idx}.json"
    prev = json.loads(meta.read_text()) if meta.exists() else {"score": -999}
    if score > prev.get("score", -999):
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(adapter, best_dir)
        meta.write_text(json.dumps({"score": score, "adapter": str(adapter), "ts": time.time()}, indent=2))
        print(f"  ★ New best for stage {stage_idx}: score={score:.3f}", flush=True)


def _rollback(stage_idx: int) -> bool:
    best = GRPO_DIR / f"best_stage{stage_idx}"
    if not best.exists():
        return False
    target = GRPO_DIR / f"stage_{stage_idx}_{_stage_name(stage_idx)}"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(best, target)
    print(f"  ↩ Rolled back stage {stage_idx} → {best}", flush=True)
    return True


def _stage_name(idx: int) -> str:
    from training.curriculum import STAGES
    return STAGES[idx].name


def _eval_stage(adapter: Path, stage_idx: int, n: int = 15) -> dict:
    """Quick eval in a subprocess so the parent never holds GPU VRAM."""
    import tempfile

    manifest_path = PROJECT_ROOT / "training/data/rl_eval_manifest.jsonl"
    if manifest_path.exists():
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out_path = tmp.name
        py = sys.executable
        rc = subprocess.run(
            [
                py, "training/rl_eval.py",
                "--adapter", str(adapter),
                "--manifest", str(manifest_path),
                "--stage", str(stage_idx),
                "--output", out_path,
            ],
            cwd=PROJECT_ROOT,
            timeout=600,
        )
        if rc.returncode == 0:
            subset = json.loads(Path(out_path).read_text(encoding="utf-8"))
            Path(out_path).unlink(missing_ok=True)
            return {
                "mean_reward": subset["mean_reward"],
                "n": subset["n"],
                "rewards": subset["rewards"],
                "manifest": True,
            }
        Path(out_path).unlink(missing_ok=True)
        print(f"  manifest eval subprocess failed (rc={rc.returncode})", flush=True)

    from training.curriculum import STAGES
    from training.rl_env import generate_training_batch, compute_reward
    from training.tool_utils import load_adapter, release_gpu

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

    mean = sum(rewards) / max(len(rewards), 1)
    del model
    release_gpu()
    return {"mean_reward": round(mean, 4), "n": len(rewards), "rewards": [round(r, 3) for r in rewards]}


def _intervention(
    health,
    stage_idx: int,
    adapter: Path,
    history: list[dict],
) -> Path:
    """Apply SFT warm-up or rollback based on session health."""
    rec = health.recommendation
    weak = health.weak_components

    if rec == "rollback_and_sft" and _rollback(stage_idx):
        adapter = GRPO_DIR / f"stage_{stage_idx}_{_stage_name(stage_idx)}"

    # Component-targeted warm-up
    steps = 100 if rec == "rollback_and_sft" else 60
    n = 100 if stage_idx >= 2 else 80

    print(f"  Intervention: {rec} → oracle SFT warmup (stage {stage_idx})", flush=True)
    _free_gpu()
    py = sys.executable
    rc = subprocess.run(
        [
            py, "training/build_rl_sft.py",
            "--stage", str(stage_idx),
            "--adapter", str(adapter),
            "--n", str(n),
            "--steps", str(steps),
        ],
        cwd=PROJECT_ROOT,
        timeout=600,
    )
    if rc.returncode != 0:
        print(f"  Intervention warmup failed (rc={rc.returncode})", flush=True)
    return adapter


def main() -> None:
    global BASE_ADAPTER, GRPO_DIR, CURRICULUM

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--session-minutes", type=float, default=45.0,
                        help="GRPO minutes per iteration")
    parser.add_argument("--target-stage", type=int, default=4,
                        help="Stop when curriculum reaches this stage index")
    parser.add_argument("--warmup-every", action="store_true", default=False,
                        help="Oracle SFT warmup at start of every iteration")
    parser.add_argument("--skip-pre-eval", action="store_true",
                        help="Skip pre-GRPO eval (saves one model load)")
    parser.add_argument("--base-adapter", default=str(BASE_ADAPTER))
    parser.add_argument("--grpo-dir", default=str(GRPO_DIR))
    parser.add_argument("--curriculum-state", default=str(CURRICULUM))
    args = parser.parse_args()

    BASE_ADAPTER = Path(args.base_adapter)
    GRPO_DIR = Path(args.grpo_dir)
    CURRICULUM = Path(args.curriculum_state)
    GRPO_DIR.mkdir(parents=True, exist_ok=True)
    CURRICULUM.parent.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    history: list[dict] = []

    print(f"\n{'='*60}\nRL AUTO-ITERATE — AgentWorld-style loop\n{'='*60}", flush=True)

    for it in range(args.max_iters):
        from training.curriculum import CurriculumState, STAGES
        from training.rl_health import diagnose_curriculum_state, diagnose_session

        state = CurriculumState.load(CURRICULUM)
        if state.current_stage >= args.target_stage or state.is_complete:
            print(f"\n✓ Target stage {args.target_stage} reached (current={state.current_stage})", flush=True)
            break

        stage_idx = state.current_stage
        stage = STAGES[stage_idx]
        adapter = _current_adapter()

        print(f"\n{'='*60}", flush=True)
        print(f"ITER {it+1}/{args.max_iters}  stage={stage_idx}:{stage.name}  adapter={adapter.name}", flush=True)
        print(f"  curriculum tail: {diagnose_curriculum_state(CURRICULUM)}", flush=True)
        print(f"{'='*60}", flush=True)

        # Pre-train eval (subprocess — must not hold GPU while GRPO runs)
        pre_eval = {"mean_reward": 0.0}
        if not args.skip_pre_eval:
            _free_gpu()
            try:
                pre_eval = _eval_stage(adapter, stage_idx, n=12)
                print(f"  Pre-eval mean reward: {pre_eval['mean_reward']}", flush=True)
            except Exception as e:
                print(f"  Pre-eval skipped: {e}", flush=True)

        # Warm-up SFT in subprocess so GPU is free before GRPO (8B OOM otherwise)
        if args.warmup_every or pre_eval["mean_reward"] < stage.advance_threshold:
            _free_gpu()
            rc = _run([
                py, "training/build_rl_sft.py",
                "--stage", str(stage_idx),
                "--adapter", str(adapter),
                "--n", "80",
                "--steps", "60",
            ])
            if rc != 0:
                print(f"  Warmup SFT failed (rc={rc})", flush=True)

        # Diagnose last session if exists
        last_sessions = sorted(SESSIONS.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        health = None
        if last_sessions:
            health = diagnose_session(last_sessions[0])
            if it > 0 or not health.ok:
                print(f"  Last session health: collapsed={health.collapsed}  "
                      f"zero_var={health.zero_variance_pct:.0f}%  rec={health.recommendation}", flush=True)
            if not health.ok:
                adapter = _intervention(health, stage_idx, adapter, history)

        from training.tool_utils import grpo_memory_profile, release_gpu
        release_gpu()

        # GRPO session — memory profile scales batch/generations for 8B on 24GB
        mem = grpo_memory_profile(adapter)
        print(f"  GRPO memory profile: {mem}", flush=True)
        session_name = f"iter{it+1:02d}_stage{stage_idx}_{datetime.now(tz=timezone.utc).strftime('%H%M')}"
        _free_gpu()
        rc = _run([
            py, "training/train_rl.py",
            "--base-adapter", str(adapter),
            "--output", str(GRPO_DIR),
            "--curriculum-state", str(CURRICULUM),
            "--batch-size", mem["batch_size"],
            "--num-generations", mem["num_generations"],
            "--max-completion-length", mem["max_completion_length"],
            "--max-prompt-length", mem["max_prompt_length"],
            "--max-minutes", str(args.session_minutes),
            "--adaptive-bugs",
            "--session-name", session_name,
        ], timeout=int(args.session_minutes * 60 + 600))

        session_dir = SESSIONS / session_name
        post_health = diagnose_session(session_dir) if session_dir.exists() else None

        # Self-distill: good traces logged in session; oracle warmup handles SFT
        if session_dir.exists() and post_health and post_health.mean_reward > 0.3:
            from training.build_rl_sft import collect_session_traces
            n_good = len(collect_session_traces(session_dir, threshold=0.4))
            print(f"  Session had {n_good} good traces (for analysis)", flush=True)

        # Post-eval
        _free_gpu()
        try:
            post_eval = _eval_stage(adapter, stage_idx, n=15)
        except Exception as e:
            post_eval = {"mean_reward": 0.0, "error": str(e)}

        state = CurriculumState.load(CURRICULUM)
        record = {
            "iteration": it + 1,
            "stage_idx": stage_idx,
            "stage": stage.name,
            "session": session_name,
            "grpo_rc": rc,
            "pre_eval": pre_eval,
            "post_eval": post_eval,
            "health": post_health.__dict__ if post_health else None,
            "curriculum_stage_after": state.current_stage,
            "advanced": state.current_stage > stage_idx,
        }
        history.append(record)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(json.dumps(history, indent=2, ensure_ascii=False))

        _save_best(stage_idx, adapter, post_eval.get("mean_reward", 0))

        print(f"\n  Post-eval: {post_eval.get('mean_reward')}  "
              f"stage now: {state.current_stage}  advanced={record['advanced']}", flush=True)

        if post_health and post_health.collapsed and not record["advanced"]:
            print("  ⚠ Session collapsed without advance — rollback next iter", flush=True)

        time.sleep(3)

    print(f"\n{'='*60}\nDone. Log → {LOG}\n{'='*60}", flush=True)
    if history:
        best = max(history, key=lambda h: h.get("post_eval", {}).get("mean_reward", -999))
        print(f"Best post-eval: {best['post_eval']['mean_reward']} @ iter {best['iteration']} stage {best['stage']}")


if __name__ == "__main__":
    main()

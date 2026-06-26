#!/usr/bin/env python3
"""Full Qwen3-8B pipeline: CPT → SFT → curricular GRPO (SAS bug env + analysis reward).

Designed for Vast 3090 with local clone at /workspace/models/Qwen3-8B.

Usage:
    python training/run_qwen3_full_pipeline.py
    python training/run_qwen3_full_pipeline.py --skip-cpt --skip-data
    python training/run_qwen3_full_pipeline.py --wait-for-clone
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

CONFIG = PROJECT_ROOT / "training/config_qwen3_8b_vast.yaml"
CLONE_DIR = Path("/workspace/models/Qwen3-8B")
SFT_OUT = PROJECT_ROOT / "training/output/phase2-qwen3-8b-sft"
GRPO_OUT = PROJECT_ROOT / "training/output/grpo-qwen3-8b"
CURRICULUM = PROJECT_ROOT / "training/output/curriculum_state_qwen3.json"
LOG = PROJECT_ROOT / "training/output/qwen3_full_pipeline.json"


def _py() -> str:
    return sys.executable


def _run(cmd: list[str], *, check: bool = True) -> int:
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=check).returncode


def _clone_ready(path: Path, min_gb: float = 14.0) -> bool:
    if not (path / "config.json").exists():
        return False
    idx = path / "model.safetensors.index.json"
    if not idx.exists() and not any(path.glob("model*.safetensors")):
        return False
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        return False
    if total < min_gb * 1e9:
        return False
    # all shards from index present
    if idx.exists():
        data = json.loads(idx.read_text())
        for name in data.get("weight_map", {}).values():
            if not (path / name).exists():
                return False
    return True


def wait_for_clone(path: Path, poll_s: int = 60) -> None:
    print(f"Waiting for clone at {path} (≥14 GB + all shards)...", flush=True)
    while True:
        try:
            sz = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
        except OSError:
            sz = 0.0
        git_busy = subprocess.run(
            ["pgrep", "-f", f"git.*Qwen3-8B"],
            capture_output=True,
        ).returncode == 0
        print(f"  {sz:.1f} GB  git_active={git_busy}", flush=True)
        if _clone_ready(path) and not git_busy:
            print("Clone ready.", flush=True)
            return
        time.sleep(poll_s)


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-8B CPT → SFT → RL")
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--skip-cpt", action="store_true")
    p.add_argument("--skip-sft", action="store_true")
    p.add_argument("--skip-rl", action="store_true")
    p.add_argument("--wait-for-clone", action="store_true")
    p.add_argument("--clone-dir", default=str(CLONE_DIR))
    p.add_argument("--max-iters", type=int, default=3)
    p.add_argument("--session-minutes", type=float, default=90.0)
    p.add_argument("--reset-curriculum", action="store_true", default=True)
    p.add_argument("--no-reset-curriculum", dest="reset_curriculum", action="store_false")
    args = p.parse_args()
    py = _py()
    clone = Path(args.clone_dir)
    report: dict = {"clone_dir": str(clone)}

    if args.wait_for_clone:
        wait_for_clone(clone)

    if not args.skip_data:
        _run([py, "training/build_corpus.py"])
        _run([py, "training/build_tool_sft.py"])
        _run([py, "training/build_agent_sft.py"])

    cpt_merged: str | None = None
    if not args.skip_cpt:
        rc = _run([py, "training/finetune.py", "--config", args.config, "--phase", "1"], check=False)
        report["cpt_rc"] = rc
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text())
        cpt_merged = str(PROJECT_ROOT / cfg["phase1"]["output_dir"]) + "-merged"
        report["cpt_merged"] = cpt_merged
    else:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text())
        cpt_merged = str(PROJECT_ROOT / cfg["phase1"]["output_dir"]) + "-merged"
        if not Path(cpt_merged).exists():
            cpt_merged = cfg.get("base_model", str(clone))

    if not args.skip_sft:
        sft_base = cpt_merged if Path(cpt_merged).exists() else cfg.get("base_model", str(clone))
        from training.sft_run import run_sft
        run_sft(
            "qwen3-8b",
            str(SFT_OUT),
            train_file="training/data/agent_sft_train.jsonl",
            eval_file="training/data/agent_sft_eval.jsonl",
            epochs=3,
            base_model_override=sft_base,
        )
        from training.tool_utils import eval_adapter
        ev = eval_adapter(SFT_OUT)
        report["sft_eval"] = {k: v for k, v in ev.items() if k != "failures"}
        print(f"SFT tool accuracy: {ev['accuracy_pct']}%", flush=True)

    if args.skip_rl:
        LOG.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Done (no RL). Log → {LOG}")
        return

    if args.reset_curriculum and CURRICULUM.exists():
        CURRICULUM.unlink()

    rc = _run([
        py, "training/iterate_rl.py",
        "--max-iters", str(args.max_iters),
        "--session-minutes", str(args.session_minutes),
        "--base-adapter", str(SFT_OUT),
        "--grpo-dir", str(GRPO_OUT),
        "--curriculum-state", str(CURRICULUM),
    ], check=False)
    report["rl_rc"] = rc
    report["sft_adapter"] = str(SFT_OUT)
    report["grpo_dir"] = str(GRPO_OUT)
    LOG.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull pipeline log → {LOG}")


if __name__ == "__main__":
    main()

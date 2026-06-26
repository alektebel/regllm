#!/usr/bin/env python3
"""Build fixed RL eval manifest for stable pre/post metrics.

Usage:
    PYTHONPATH=. .venv/bin/python training/build_rl_eval_manifest.py
    PYTHONPATH=. .venv/bin/python training/build_rl_eval_manifest.py --n 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUT = PROJECT_ROOT / "training/data/rl_eval_manifest.jsonl"


def bug_fingerprint(bug) -> str:
    parts = [
        bug.source,
        bug.level_id or "",
        bug.mutation.mutation_type,
        bug.mutation.step_name,
        bug.mutation.target_var,
        ",".join(bug.changed_fields[:4]),
        ",".join(bug.expected_tools),
        bug.task_kind,
    ]
    return "|".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50, help="Examples per stage in manifest")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    from training.curriculum import STAGES
    from training.rl_env import build_prompt, generate_training_batch

    rows: list[dict] = []
    for stage_idx, stage in enumerate(STAGES):
        batch = generate_training_batch(args.n, stage, seed=args.seed + stage_idx * 1000)
        for i, (bug, messages) in enumerate(batch):
            rows.append({
                "id": f"eval_s{stage_idx}_{stage.name}_{i:03d}",
                "stage_idx": stage_idx,
                "stage": stage.name,
                "fingerprint": bug_fingerprint(bug),
                "source": bug.source,
                "level_id": bug.level_id,
                "mutation_type": bug.mutation.mutation_type,
                "expected_tools": bug.expected_tools,
                "task_kind": bug.task_kind,
                "changed_fields": bug.changed_fields[:6],
                "messages": messages,
                "ground_truth": {
                    "step_name": bug.mutation.step_name,
                    "target_var": bug.mutation.target_var,
                    "original_expr": bug.mutation.original_expr,
                    "correct_values": {k: bug.correct_values.get(k) for k in bug.changed_fields[:3]},
                    "buggy_values": {k: bug.buggy_values.get(k) for k in bug.changed_fields[:3]},
                },
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_stage = {}
    for r in rows:
        by_stage[r["stage"]] = by_stage.get(r["stage"], 0) + 1
    print(f"Wrote {len(rows)} eval examples → {out}")
    print(f"  by stage: {by_stage}")


if __name__ == "__main__":
    main()

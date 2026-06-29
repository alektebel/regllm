#!/usr/bin/env python3
"""Simulate GRPO groups for depth-wise trace-to-bug rewards.

Usage:
    PYTHONPATH=. .venv/bin/python training/simulate_trace_grpo.py
    PYTHONPATH=. .venv/bin/python training/simulate_trace_grpo.py --stage multi_depth
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _grpo_advantages(rewards: list[float], eps: float = 1e-4) -> list[float]:
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    mean = statistics.mean(rewards)
    std = statistics.pstdev(rewards)
    if std < eps:
        return [0.0] * len(rewards)
    return [(r - mean) / (std + eps) for r in rewards]


def _variants(bug, oracle: str) -> dict[str, str]:
    gt = bug.gold_trace
    root = gt.get("root_step", bug.mutation.step_name)
    root_var = gt.get("root_var", bug.mutation.target_var)
    symptom = gt.get("symptom_field", "ECL")

    wrong_step = oracle.replace(root, "work.final").replace(root_var, "STAGE_IFRS9")
    no_tools = (
        f"PASO: work.final\nVARIABLE: {symptom}\n"
        f"CAUSA: parece un problema downstream sin trazar."
    )
    one_hop = (
        f'<tool_call>{{"name": "trace_dependencies", "arguments": {{"target": "{symptom}"}}}}</tool_call>\n'
        f"PASO: {root}\nVARIABLE: {root_var}\nCAUSA: identificado sin recorrer todo el path."
    )
    hallucinated = (
        f"OR_EADIL es la causa según mi conocimiento. PASO: work.final VARIABLE: OR_EADIL"
    )
    return {
        "oracle_trace": oracle,
        "one_hop_only": one_hop,
        "wrong_step": wrong_step,
        "no_tools": no_tools,
        "hallucinated": hallucinated,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["single_depth", "multi_depth"], default="multi_depth")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from training.curriculum import STAGES
    from training.rl_env import compute_reward, generate_training_batch
    from training.gold_trace import oracle_trace_completion

    stage = next(s for s in STAGES if s.name == args.stage)
    batch = generate_training_batch(args.n, stage, seed=args.seed)
    bugs = [b for b, _ in batch if b.gold_trace.get("path_fields")]

    print("=" * 72)
    print(f"Trace-path GRPO simulation  stage={args.stage}  n={len(bugs)}")
    print("=" * 72)

    group_stds: list[float] = []
    rows: list[dict] = []

    for i, bug in enumerate(bugs):
        gt = bug.gold_trace
        oracle = oracle_trace_completion(bug)
        variants = _variants(bug, oracle)
        breakdowns = []
        rewards: list[float] = []

        print(f"\n--- Bug {i+1} depth={bug.depth}  {bug.mutation.mutation_type} @ {bug.mutation.step_name} ---")
        print(f"  symptom={gt.get('symptom_field')}  path={gt.get('path_fields', [])[:4]}  min_hops={gt.get('min_hops')}")
        msgs = __import__("training.rl_env", fromlist=["build_prompt"]).build_prompt(bug, stage)
        print(f"  prompt has SAS code: {'```sas' in msgs[1]['content']}")

        for name, completion in variants.items():
            bd = compute_reward(completion, bug, stage)
            breakdowns.append((name, bd))
            rewards.append(bd.total)

        adv = _grpo_advantages(rewards)
        std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        group_stds.append(std)

        print(f"\n{'variant':<16} {'total':>7} {'trace':>6} {'step':>6} {'var':>6} {'advantage':>9}")
        print("-" * 60)
        for (name, bd), a in zip(breakdowns, adv):
            c = bd.components
            mark = " ←" if abs(a) > 0.4 else ""
            print(
                f"{name:<16} {bd.total:7.3f} "
                f"{c.get('trace_path', 0):6.2f} "
                f"{c.get('correct_step', 0):6.2f} "
                f"{c.get('correct_var', 0):6.2f} "
                f"{a:+9.3f}{mark}"
            )

        print(f"  group std={std:.3f}  range={max(rewards)-min(rewards):.3f}")
        rows.append({
            "depth": bug.depth,
            "mutation": bug.mutation.mutation_type,
            "group_std": round(std, 4),
            "oracle_total": breakdowns[0][1].total,
            "trace_path_oracle": breakdowns[0][1].components.get("trace_path", 0),
        })

    mean_std = statistics.mean(group_stds) if group_stds else 0
    print("\n" + "=" * 72)
    print(f"Mean group std: {mean_std:.4f}  (want >0.15)")
    print(f"Mean oracle trace_path: {statistics.mean(r['trace_path_oracle'] for r in rows):.3f}")

    out = PROJECT_ROOT / "training/output/trace_grpo_simulation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stage": args.stage, "groups": rows, "mean_std": mean_std}, indent=2))
    print(f"Wrote → {out}")


if __name__ == "__main__":
    main()

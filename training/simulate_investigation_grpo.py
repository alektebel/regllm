#!/usr/bin/env python3
"""Simulate Phase 1 investigation GRPO groups — reward spread and advantages.

No GPU: scores oracle / partial / bad completions and prints GRPO group statistics
(the signal that drives policy gradients).

Usage:
    PYTHONPATH=. .venv/bin/python training/simulate_investigation_grpo.py
    PYTHONPATH=. .venv/bin/python training/simulate_investigation_grpo.py --n-prompts 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _grpo_advantages(rewards: list[float], eps: float = 1e-4) -> list[float]:
    """Group-relative advantages (simplified GRPO / REINFORCE baseline)."""
    import statistics
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    mean = statistics.mean(rewards)
    std = statistics.pstdev(rewards)
    if std < eps:
        return [0.0] * len(rewards)
    return [(r - mean) / (std + eps) for r in rewards]


def _completion_variants(scenario, oracle: str) -> dict[str, str]:
    """Synthetic completions mimicking GRPO sample diversity."""
    gt = scenario.ground_truth
    fields = gt.get("must_mention_fields") or ["ECL"]
    phrases = gt.get("must_contain_phrases") or []
    steps = scenario.tool_steps
    tool_blocks = [
        f'<tool_call>{{"name": "{t}", "arguments": {json.dumps(a)}}}</tool_call>'
        for t, a in steps
    ]

    good = oracle

    partial_tools = "\n".join(tool_blocks[: max(1, len(tool_blocks) - 1)])
    partial = (
        partial_tools
        + f"\n\nHay un problema con {fields[0]}. Falta detalle de causa raíz."
    )

    wrong_tool = (
        '<tool_call>{"name": "search_experience", "arguments": {"query": "bug"}}</tool_call>\n'
        f"Según experiencia previa, el campo {fields[0]} puede fallar."
    )

    hallucinated = (
        "El problema es OR_EADIL = MAX(EAD, 0) según mi conocimiento. "
        f"La causa es un error en {fields[0]} sin trazar lineage."
    )

    tools_only = "\n".join(tool_blocks)

    no_tools = oracle.split("\n")[-1] if "\n" in oracle else oracle

    return {
        "oracle": good,
        "partial_tools": partial,
        "experience_only": wrong_tool,
        "hallucinated": hallucinated,
        "tools_no_analysis": tools_only,
        "analysis_no_tools": no_tools,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-prompts", type=int, default=5)
    p.add_argument("--manifest", default=str(PROJECT_ROOT / "training/data/investigation_eval_manifest.jsonl"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        from training.build_investigation_eval_manifest import main as build_main
        import sys as _sys
        _sys.argv = ["build_investigation_eval_manifest.py"]
        build_main()

    from training.investigation_env import (
        generate_investigation_batch,
        oracle_completion,
        compute_investigation_reward,
    )

    batch = generate_investigation_batch(args.n_prompts, seed=args.seed, manifest_path=manifest)

    print("=" * 72)
    print("Phase 1 Investigation GRPO Simulation (CPU — reward / advantage signal)")
    print("=" * 72)

    all_group_stds: list[float] = []
    zero_var_count = 0
    summary_rows: list[dict] = []

    for idx, (scenario, _messages) in enumerate(batch):
        oracle = oracle_completion(scenario)
        variants = _completion_variants(scenario, oracle)
        rewards_grpo: list[float] = []
        breakdowns = []

        print(f"\n--- Prompt {idx + 1}: {scenario.scenario_id} ({scenario.scenario_type}) ---")
        print(f"Q: {scenario.question[:90]}...")
        print(f"Required tools: {scenario.required_tools}")

        for name, completion in variants.items():
            bd = compute_investigation_reward(completion, scenario)
            breakdowns.append((name, bd))
            rewards_grpo.append(bd.grpo_reward)

        advantages = _grpo_advantages(rewards_grpo)
        import statistics
        std = statistics.pstdev(rewards_grpo) if len(rewards_grpo) > 1 else 0.0
        all_group_stds.append(std)
        if std < 1e-4:
            zero_var_count += 1

        print(f"\n{'variant':<22} {'total':>6} {'grpo_r':>7}  {'advantage':>9}  components")
        print("-" * 72)
        for (name, bd), adv in zip(breakdowns, advantages):
            comp = " ".join(f"{k}={v:.2f}" for k, v in bd.components.items())
            flag = " ← learn" if abs(adv) > 0.3 else ""
            print(f"{name:<22} {bd.total:6.3f} {bd.grpo_reward:7.3f}  {adv:+9.3f}  {comp}{flag}")
            if bd.ungrounded_claims and name in ("hallucinated", "analysis_no_tools"):
                print(f"  ungrounded: {bd.ungrounded_claims[:3]}")

        print(f"\n  group mean={statistics.mean(rewards_grpo):+.3f}  std={std:.3f}  "
              f"range={max(rewards_grpo)-min(rewards_grpo):.3f}")
        if std < 0.05:
            print("  ⚠ LOW VARIANCE — GRPO would emit ~zero gradients for this prompt group")

        summary_rows.append({
            "id": scenario.scenario_id,
            "type": scenario.scenario_type,
            "group_std": round(std, 4),
            "group_range": round(max(rewards_grpo) - min(rewards_grpo), 4),
            "oracle_total": breakdowns[0][1].total,
            "hallucinated_total": next(b[1].total for b in breakdowns if b[0] == "hallucinated"),
        })

    print("\n" + "=" * 72)
    print("AGGREGATE GRPO SIGNAL")
    print("=" * 72)
    n = len(batch)
    mean_std = sum(all_group_stds) / n if n else 0
    print(f"  Prompt groups:     {n}")
    print(f"  Mean group std:    {mean_std:.4f}  (want >0.15 for useful GRPO)")
    print(f"  Zero-var groups:   {zero_var_count}/{n}")
    print(f"  Oracle mean total: {sum(r['oracle_total'] for r in summary_rows)/n:.3f}")
    print(f"  Halluc mean total: {sum(r['hallucinated_total'] for r in summary_rows)/n:.3f}")

    out = PROJECT_ROOT / "training/output/investigation_grpo_simulation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"groups": summary_rows, "mean_group_std": mean_std}, indent=2))
    print(f"\n  Wrote → {out}")


if __name__ == "__main__":
    main()

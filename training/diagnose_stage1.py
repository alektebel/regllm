#!/usr/bin/env python3
"""Diagnose Stage 1 reward collapse — sample completions from a GRPO checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="training/output/grpo-curriculum/stage_1_single_depth_checkpoint_r11")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    from training.curriculum import STAGES
    from training.rl_env import generate_training_batch, compute_reward, build_prompt
    from training.tool_utils import load_adapter

    stage = STAGES[1]  # single_depth
    batch = generate_training_batch(args.n, stage, seed=99)
    print(f"Stage: {stage.name}  bugs={len(batch)}\n")

    model, tokenizer = load_adapter(args.adapter)

    for i, (bug, messages) in enumerate(batch):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=400, temperature=0.9, do_sample=True,
            use_cache=True,
        )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        bd = compute_reward(text, bug, stage)

        print("=" * 60)
        print(f"Bug {i}: mutation step={bug.mutation.step_name!r} var={bug.mutation.target_var!r}")
        print(f"  type={bug.mutation.mutation_type}  changed={bug.changed_fields[:3]}")
        print(f"  reward={bd.total}  components={bd.components}")
        print(f"  tools={bd.tool_calls_found}")
        print(f"  user prompt tail: {messages[1]['content'][-300:]}")
        print(f"  completion preview:\n{text[:600]}")
        print()


if __name__ == "__main__":
    main()

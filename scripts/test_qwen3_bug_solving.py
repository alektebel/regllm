#!/usr/bin/env python3
"""Test Qwen3-8B on procedurally generated SAS bugs.

Usage:
    python3 scripts/test_qwen3_bug_solving.py --stages 2 3  --n 20

Reports per-mutation-type accuracy: step, variable, fix, and RL reward.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from training.curriculum import STAGES
from training.rl_env import compute_reward, generate_training_batch, build_prompt

MODEL_NAME = "Qwen/Qwen3-8B"


@dataclass
class BugResult:
    mutation_type: str = ""
    correct_step: bool = False
    correct_var: bool = False
    correct_fix: bool = False
    reward: float = 0.0
    stage: str = ""
    step_name: str = ""
    target_var: str = ""
    original_expr: str = ""
    model_step: str = ""
    model_var: str = ""
    model_fix: str = ""
    duration_ms: float = 0.0


def load_model():
    print(f"Loading {MODEL_NAME} in 4-bit...")
    t0 = time.time()
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s, device: {model.device}")
    return model, tokenizer


def extract_sas_variables(text: str) -> set[str]:
    """Extract likely SAS variable names (UPPERCASE identifiers)."""
    return set(re.findall(r'\b[A-Z][A-Z_0-9]{2,}\b', text))


def extract_step_name(text: str) -> str:
    """Try to find a DATA step name in the model output."""
    patterns = [
        r'(?:paso|step|data)\s*(?:es|:)?\s*`?(\w+)`?',
        r'en\s+el\s+paso\s+`?(\w+)`?',
        r'`(\w+)`',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            candidate = m.group(1)
            if candidate.upper() != candidate:
                continue  # skip non-SAS names
            return candidate
    return ""


def extract_var_name(text: str) -> str:
    """Try to find the buggy variable in the model output."""
    patterns = [
        r'(?:variable|campo)\s*(?:es|:)?\s*`?(\w+)`?',
        r'variable\s+(\w+)',
        r'el\s+error\s+est[áa]\s+en\s+`?(\w+)`?',
        r'`(\w+)`\s*(?:es|:).*(?:variable|campo)',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return ""


def extract_fix(text: str) -> str:
    """Try to extract the proposed fix expression."""
    patterns = [
        r'(?:correcci[oó]n|fix|arreglo|deber[íi]a\s+ser)\s*(?:es|:)?\s*`?([^`\n]+)`?',
        r'deber[íi]a\s+ser\s+`?([^`\n]+)`?',
        r'cambiarlo\s+a\s+`?([^`\n]+)`?',
        r'CORRECCI[OÓ]N:\s*`?([^`\n]+)`?',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()[:200]
    return ""


def run_bug_test(
    model,
    tokenizer,
    bug: Any,
    messages: list[dict],
) -> BugResult:
    """Run model on one bug prompt and evaluate."""
    result = BugResult()
    result.mutation_type = bug.mutation.mutation_type
    result.stage = ""
    result.step_name = bug.mutation.step_name
    result.target_var = bug.mutation.target_var
    result.original_expr = bug.mutation.original_expr[:100]

    t0 = time.perf_counter()

    # Build prompt
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    result.duration_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Evaluate
    reward = compute_reward(completion, bug, STAGES[2])  # stage 2 = single_depth
    result.reward = reward.total if hasattr(reward, 'total') else reward

    result.model_step = extract_step_name(completion)
    result.model_var = extract_var_name(completion)
    result.model_fix = extract_fix(completion)

    result.correct_step = bug.mutation.step_name in completion or (
        result.model_step and result.model_step.upper() == bug.mutation.step_name.upper()
    )
    result.correct_var = bug.mutation.target_var.upper() in completion.upper() or (
        result.model_var and result.model_var.upper() == bug.mutation.target_var.upper()
    )
    # Fix is correct if original expression appears in completion
    result.correct_fix = any(
        expr in completion.replace(" ", "").replace("\n", " ")
        for expr in [bug.mutation.original_expr.replace(" ", ""), bug.mutation.original_expr]
    ) or (result.model_fix and result.model_fix.replace(" ", "") == bug.mutation.original_expr.replace(" ", ""))

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=int, nargs="+", default=[2, 3], help="Curriculum stages to test")
    parser.add_argument("--n", type=int, default=10, help="Bugs per stage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    model, tokenizer = load_model()

    all_results: list[BugResult] = []

    for stage_idx in args.stages:
        stage = STAGES[stage_idx]
        print(f"\n── Stage {stage_idx}: {stage.name} ──")
        batch = generate_training_batch(args.n, stage, seed=args.seed + stage_idx * 7)

        print(f"  Generating {len(batch)} bugs...")
        for bug, messages in batch:
            result = run_bug_test(model, tokenizer, bug, messages)
            all_results.append(result)

            step_ok = "✓" if result.correct_step else "✗"
            var_ok = "✓" if result.correct_var else "✗"
            fix_ok = "✓" if result.correct_fix else "✗"
            print(f"  [{result.mutation_type:20s}] step={step_ok} var={var_ok} fix={fix_ok} "
                  f"R={result.reward:+.3f} ({result.duration_ms:.0f}ms)")

    # ── Report ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Qwen3-8B Bug Solving Report")
    print("=" * 60)

    # By mutation type
    by_type: dict[str, list[BugResult]] = defaultdict(list)
    for r in all_results:
        by_type[r.mutation_type].append(r)

    print(f"\n  Overall ({len(all_results)} bugs):")
    overall_acc = {
        k: sum(1 for r in all_results if getattr(r, k)) / max(len(all_results), 1)
        for k in ("correct_step", "correct_var", "correct_fix")
    }
    avg_reward = sum(r.reward for r in all_results) / max(len(all_results), 1)
    print(f"    Step acc:  {overall_acc['correct_step']:.1%}")
    print(f"    Var acc:   {overall_acc['correct_var']:.1%}")
    print(f"    Fix acc:   {overall_acc['correct_fix']:.1%}")
    print(f"    Avg reward: {avg_reward:+.3f}")

    for mtype, results in sorted(by_type.items()):
        acc = sum(1 for r in results if r.correct_var) / max(len(results), 1)
        step_acc = sum(1 for r in results if r.correct_step) / max(len(results), 1)
        avg_r = sum(r.reward for r in results) / max(len(results), 1)
        print(f"  {mtype:20s}: {len(results):2d} bugs  "
              f"step={step_acc:.0%} var={acc:.0%} R={avg_r:+.3f}")

    # Save results
    out_path = Path("scripts/qwen3_bug_solving_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "stages": args.stages,
            "n": args.n,
            "results": [
                {k: v for k, v in r.__dict__.items()}
                for r in all_results
            ],
            "summary": {
                "step_accuracy": overall_acc["correct_step"],
                "var_accuracy": overall_acc["correct_var"],
                "fix_accuracy": overall_acc["correct_fix"],
                "avg_reward": avg_reward,
            }
        }, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()

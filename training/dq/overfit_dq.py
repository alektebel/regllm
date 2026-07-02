#!/usr/bin/env python3
"""Overfit GRPO on a single DQ coherence rule — sanity check for the DQ env.

Loads Qwen3-8B (default) with QLoRA, picks the ADJUDICACION rule (the user's
example: valor de adjudicación > 0 sin flag activo), and runs GRPO until the
model reliably emits a check scoring 5/5 reward components.

If this fits in VRAM and reward climbs, the full DQ training run will work.

Usage:
    .venv/bin/python training/dq/overfit_dq.py
    .venv/bin/python training/dq/overfit_dq.py --model Qwen/Qwen3-8B --steps 30
    .venv/bin/python training/dq/overfit_dq.py --rule-id dq_coh_cerrado_sin_terminacion
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

# Unsloth MUST be imported before transformers to apply GRPO memory patches.
import unsloth  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Compatibility shim for transformers 5.5.0
import transformers.utils.hub as _hub
if not hasattr(_hub, "TRANSFORMERS_CACHE"):
    _hub.TRANSFORMERS_CACHE = os.path.expanduser("~/.cache/huggingface/hub")


def _free_gpu() -> None:
    """Unload any Ollama models hogging VRAM (mirrors overfit_single.py)."""
    try:
        import httpx
        for m in httpx.get("http://localhost:11434/api/ps", timeout=5).json().get("models", []):
            name = m.get("name", "")
            if name:
                httpx.post("http://localhost:11434/api/generate",
                           json={"model": name, "keep_alive": 0}, timeout=10)
                print(f"  unloaded {name}")
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit GRPO on one DQ rule")
    parser.add_argument("--model", default="Qwen/Qwen3-8B",
                        help="HuggingFace model id or local path")
    parser.add_argument("--rule-id", default="dq_coh_adjudicacion_valor_sin_flag",
                        help="Coherence rule id (see training/dq/coherence_rules.py)")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=2)
    # Sized from the task, not the stopwatch: Qwen3 thinking-mode needs
    # ~300-600 tokens to plan a non-trivial WHERE predicate, plus ~80-150
    # for the <check>{...}</check> JSON. 768 is the floor; bump to 1024
    # for cross-table rules with JOINs. (~75s/step on 8B, ~12 GB peak.)
    parser.add_argument("--max-completion-length", type=int, default=768)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--tiled-mlp", action="store_true")
    args = parser.parse_args()

    _free_gpu()

    import torch
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    total_gb = torch.cuda.mem_get_info()[1] / 1e9
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"GPU: {torch.cuda.get_device_name()} — {total_gb:.1f} GB total, {free_gb:.1f} GB free")

    # ── Step 1: pick the rule + build prompt ────────────────────────────
    print(f"\n[1/4] Loading DQ rule {args.rule_id}...")
    from training.dq.dq_env import get_rule, build_prompt
    from training.dq.dq_reward import grpo_reward, reset_caches

    rule = get_rule(args.rule_id)
    messages = build_prompt(rule)
    print(f"  rule: {rule.rule_id}")
    print(f"  category={rule.category} severity={rule.severity} visible={rule.visible}")
    print(f"  columns: {rule.columns}")

    # ── Step 2: load model with QLoRA ───────────────────────────────────
    print(f"\n[2/4] Loading {args.model} with QLoRA (r={args.lora_r})...")
    from unsloth import FastLanguageModel

    max_seq = args.max_prompt_length + args.max_completion_length
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=max_seq, dtype=None, load_in_4bit=True,
    )

    peft_kwargs = dict(
        r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", use_gradient_checkpointing="unsloth",
    )
    if args.tiled_mlp:
        peft_kwargs["unsloth_tiled_mlp"] = True
    model = FastLanguageModel.get_peft_model(model, **peft_kwargs)

    cfg = getattr(model, "generation_config", None)
    if cfg is not None:
        cfg.max_length = None

    used = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after load: {used:.1f} GB allocated")

    # ── Step 3: dataset (1 sample repeated) ─────────────────────────────
    print("\n[3/4] Building dataset...")
    from datasets import Dataset

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    toks = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
    print(f"  prompt tokens: {toks}")

    n_repeats = max(args.steps, 4)
    dataset = Dataset.from_list([{"prompt": prompt_text}] * n_repeats)

    reward_fn = grpo_reward(rule)

    # ── Step 4: GRPO ───────────────────────────────────────────────────
    print(f"\n[4/4] Starting GRPO overfit ({args.steps} steps, gen={args.num_generations})...")
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback

    output_dir = str(PROJECT_ROOT / "training/output/dq-overfit-test")
    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=args.steps,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        temperature=1.0,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        seed=42,
        report_to="none",
        generation_kwargs={"min_new_tokens": 32, "top_p": 0.95},
    )

    class VRAMCallback(TrainerCallback):
        def on_step_end(self, _args, state, control, **kwargs):
            used = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  step {state.global_step}: VRAM {used:.1f} GB (peak {peak:.1f} GB)")

    # TRL expects warnings_issued dict on model; PEFT doesn't forward it.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, reward_funcs=reward_fn,
        train_dataset=dataset, args=config,
    )
    trainer.add_callback(VRAMCallback())

    t0 = time.time()
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n  OOM! Peak VRAM: {peak:.1f} GB")
        print("  Try: --num-generations 1, --max-completion-length 256, --max-prompt-length 768")
        return

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    free = torch.cuda.mem_get_info()[0] / 1e9
    print(f"\n{'='*50}")
    print(f"Done in {elapsed:.0f}s ({elapsed/args.steps:.1f}s/step)")
    print(f"Peak VRAM: {peak:.1f} GB / {total_gb:.1f} GB  (headroom {total_gb - peak:.1f} GB)")
    print(f"Free VRAM: {free:.1f} GB")


if __name__ == "__main__":
    main()

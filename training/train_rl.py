#!/usr/bin/env python3
"""Curricular GRPO training for SAS debugging — VibeThinker style.

Three-stage curriculum:
  Stage 1 (tool_selection): learn to call the right tool
  Stage 2 (single_depth):   find bugs in the final DATA step
  Stage 3 (multi_depth):    trace bugs through multiple steps

Each stage uses procedurally generated bugs (BugGenerator) with
functionally verifiable rewards. Self-distillation: collect good
traces, SFT on them, then GRPO on harder cases.

Usage:
    # Full curricular training
    .venv/bin/python training/train_rl.py

    # Start from a specific stage
    .venv/bin/python training/train_rl.py --start-stage 1

    # Self-distillation only (collect traces, no GRPO)
    .venv/bin/python training/train_rl.py --distill-only

    # Eval reward function on procedural bugs
    .venv/bin/python training/train_rl.py --eval-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Must be set before Unsloth import — torch.compile workspace can OOM on 24GB (3090).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_ADAPTER = PROJECT_ROOT / "training/output/tool-sft-antihalluc"
OUT_DIR = PROJECT_ROOT / "training/output/grpo-curriculum"
DISTILL_DIR = PROJECT_ROOT / "training/output/distill-traces"
CURRICULUM_STATE = PROJECT_ROOT / "training/output/curriculum_state.json"

# Legacy prompts file (for backward compatibility)
PROMPTS_FILE = PROJECT_ROOT / "training/data/rl_prompts.jsonl"


def _free_gpu() -> None:
    try:
        import httpx
        for m in httpx.get("http://localhost:11434/api/ps", timeout=5).json().get("models", []):
            name = m.get("name", "")
            if name:
                httpx.post(
                    "http://localhost:11434/api/generate",
                    json={"model": name, "keep_alive": 0},
                    timeout=10,
                )
                print(f"  unloaded {name}", flush=True)
    except Exception as e:
        print(f"  (gpu cleanup: {e})", flush=True)


def main() -> None:
    global BASE_ADAPTER, OUT_DIR, CURRICULUM_STATE

    parser = argparse.ArgumentParser(description="Curricular GRPO for SAS debugging")
    parser.add_argument("--base-adapter", default=str(BASE_ADAPTER))
    parser.add_argument("--output", default=str(OUT_DIR))
    parser.add_argument("--curriculum-state", default=str(CURRICULUM_STATE))
    parser.add_argument("--start-stage", type=int, default=None,
                        help="Start from this stage (0-2). Default: resume from saved state.")
    parser.add_argument("--num-generations", type=int, default=8,
                        help="Completions per prompt for GRPO (min 4, ideally 8+)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Bugs per training batch")
    parser.add_argument("--epochs-per-stage", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--eval-only", action="store_true",
                        help="Just eval the reward function on procedural bugs")
    parser.add_argument("--distill-only", action="store_true",
                        help="Collect good traces for self-distillation SFT")
    parser.add_argument("--distill-threshold", type=float, default=0.3,
                        help="Min reward to keep a trace for distillation")
    parser.add_argument("--distill-n", type=int, default=100,
                        help="Number of traces to collect for distillation")
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy fixed prompts instead of procedural bugs")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="Stop after N minutes (timed instrumented session)")
    parser.add_argument("--session-name", default=None,
                        help="Session folder name under training/output/rl_sessions/")
    parser.add_argument("--adaptive-bugs", action="store_true",
                        help="Adapt mutation difficulty from recent reward components")
    args = parser.parse_args()

    BASE_ADAPTER = Path(args.base_adapter)
    OUT_DIR = Path(args.output)
    CURRICULUM_STATE = Path(args.curriculum_state)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CURRICULUM_STATE.parent.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        _eval_procedural()
        return

    if args.distill_only:
        _self_distill(args)
        return

    if args.legacy:
        _train_legacy(args)
        return

    _train_curricular(args)


# ── Curricular GRPO ────────────────────────────────────────────────────

def _train_curricular(args) -> None:
    """Run curricular GRPO: iterate through stages, advance on reward threshold."""
    from training.curriculum import CurriculumState, STAGES
    from training.rl_env import generate_training_batch, compute_reward

    # Optional timed instrumented session
    session = None
    adaptive = None
    session_start = time.time()
    if args.max_minutes or args.adaptive_bugs:
        from training.adaptive_bugs import AdaptiveProfile
        from training.rl_metrics import RLSessionLogger

        name = args.session_name or datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        session_dir = PROJECT_ROOT / "training/output/rl_sessions" / name
        session = RLSessionLogger(session_dir, num_generations=args.num_generations)
        session.write_meta({
            "max_minutes": args.max_minutes,
            "adaptive_bugs": args.adaptive_bugs,
            "batch_size": args.batch_size,
            "num_generations": args.num_generations,
            "base_adapter": args.base_adapter,
        })
        print(f"  Session log → {session_dir}", flush=True)
        if args.adaptive_bugs:
            adaptive = AdaptiveProfile()
            print(f"  Adaptive procedural bugs: ON", flush=True)

    def _time_up() -> bool:
        return args.max_minutes is not None and (time.time() - session_start) >= args.max_minutes * 60

    # Load or create curriculum state
    state = CurriculumState.load(CURRICULUM_STATE)
    if args.start_stage is not None:
        state.current_stage = args.start_stage
        state.episode_rewards = []

    if state.is_complete:
        print("All curriculum stages complete!")
        return

    _free_gpu()
    _autotune_grpo_memory(args)

    # Load model
    adapter = args.base_adapter
    stage_idx = state.current_stage
    stage_ckpt = Path(args.output) / f"stage_{stage_idx}_{STAGES[stage_idx].name}"
    # Only auto-resume stage checkpoint when using default base (not a custom SFT adapter)
    using_default_base = Path(args.base_adapter).resolve() == BASE_ADAPTER.resolve()
    if (using_default_base and stage_ckpt.exists()
            and (stage_ckpt / "adapter_config.json").exists()):
        adapter = str(stage_ckpt)
        print(f"  Resuming from stage checkpoint: {adapter}")
    model, tokenizer = _load_model(adapter, max_seq_length=args.max_prompt_length + args.max_completion_length)

    train_round = 0
    while not state.is_complete:
        if _time_up():
            print(f"\n  ⏱ Session time limit ({args.max_minutes} min) reached.", flush=True)
            break

        stage = state.stage
        stage_idx = state.current_stage
        train_round += 1
        if session:
            session.round_idx = train_round

        # Adaptive stage overrides
        active_stage = adaptive.adaptive_stage(stage) if adaptive else stage

        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx}: {stage.name}")
        print(f"  {stage.description}")
        print(f"  depth: {active_stage.min_depth}–{active_stage.max_depth or '∞'}")
        if adaptive:
            print(f"  adaptive difficulty: {adaptive.difficulty:.2f}  types: {adaptive.allowed_mutation_types()}")
        print(f"  advance: reward > {stage.advance_threshold} over {stage.advance_window} eps")
        if args.max_minutes:
            elapsed = (time.time() - session_start) / 60
            print(f"  session: {elapsed:.1f}/{args.max_minutes} min")
        print(f"{'='*60}\n")

        # Generate training batch for this stage (fresh bugs each round)
        batch = generate_training_batch(
            args.batch_size, active_stage,
            seed=42 + stage_idx * 1000 + train_round,
            mutation_weights=adaptive.mutation_weights if adaptive else None,
        )
        if not batch:
            print(f"  No valid bugs for stage {stage.name}, advancing...")
            state.current_stage += 1
            continue

        print(f"  Generated {len(batch)} training examples")

        # Build HF dataset — tool descriptions are in the system prompt
        # already (from build_prompt), so we skip tools= to save ~650 tokens.
        from datasets import Dataset
        ds_rows = []
        bug_map: dict[str, tuple] = {}  # prompt_key → (bug, stage)
        for bug, messages in batch:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False, add_generation_prompt=True,
            )
            key = prompt_text[:200]
            ds_rows.append({"prompt": prompt_text})
            bug_map[key] = (bug, stage)

        dataset = Dataset.from_list(ds_rows)

        # Log prompt token counts + sample prompts
        for idx, row in enumerate(ds_rows[:3]):
            toks = tokenizer(row["prompt"], return_tensors="pt")["input_ids"].shape[1]
            print(f"  prompt[{idx}] tokens: {toks}")
        print()

        # Reward function that uses functional verification
        _reward_call_count = [0]
        _round_traces: list[dict] = []

        def reward_fn(completions: list[str], prompts: list[str] | None = None, **kwargs) -> list[float]:
            rewards = []
            _reward_call_count[0] += 1
            call_n = _reward_call_count[0]
            n_gen = args.num_generations

            for gi in range(0, len(completions), n_gen):
                group_completions = completions[gi : gi + n_gen]
                group_prompts = (prompts[gi : gi + n_gen] if prompts else [""] * len(group_completions))
                group_rewards: list[float] = []
                group_components: list[dict] = []
                prompt_key = group_prompts[0][:200] if group_prompts else ""
                bug_info = bug_map.get(prompt_key)
                mutation_type = bug_info[0].mutation.mutation_type if bug_info else ""
                bug_step = bug_info[0].mutation.step_name if bug_info else ""

                for i, text in enumerate(group_completions):
                    if bug_info:
                        bug, stg = bug_info
                        breakdown = compute_reward(text, bug, stg)
                        rewards.append(breakdown.total)
                        group_rewards.append(breakdown.total)
                        group_components.append(breakdown.components)
                        trace = {
                            "reward": breakdown.total,
                            "components": breakdown.components,
                            "mutation_type": bug.mutation.mutation_type,
                        }
                        _round_traces.append(trace)
                        if session and (call_n <= 5 or i == 0):
                            session.record_trace(
                                round_idx=train_round,
                                prompt_key=prompt_key,
                                completion=text,
                                reward=breakdown.total,
                                components=breakdown.components,
                                mutation_type=bug.mutation.mutation_type,
                                bug_step=bug.mutation.step_name,
                            )
                        if call_n <= 3 and i < 2:
                            print(f"  [reward call {call_n}, completion {gi + i}]")
                            print(f"    reward={breakdown.total}  components={breakdown.components}")
                            print(f"    tools_found={breakdown.tool_calls_found}")
                            preview = text[:200].replace('\n', ' ')
                            print(f"    completion: {preview}...")
                            print()
                    else:
                        rewards.append(-1.0)
                        group_rewards.append(-1.0)

                if session and group_rewards:
                    session.record_reward_group(
                        group_rewards, group_components,
                        round_idx=train_round,
                        prompt_key=prompt_key,
                        mutation_type=mutation_type,
                        step_name=bug_step,
                    )
            return rewards

        # GRPO training for this stage
        from trl import GRPOConfig, GRPOTrainer

        stage_output = Path(args.output) / f"stage_{stage_idx}_{stage.name}"
        from training.tool_utils import is_8b_adapter
        is_8b = is_8b_adapter(args.base_adapter)
        grpo_kw: dict = {}
        if is_8b:
            # Smaller logprob chunks + mini-batches reduce peak VRAM during ref forward.
            grpo_kw["unsloth_logit_chunk_multiplier"] = 8
            grpo_kw["unsloth_grpo_mini_batch"] = 1
        config = GRPOConfig(
            output_dir=str(stage_output),
            num_train_epochs=args.epochs_per_stage,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=5e-6,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            max_prompt_length=args.max_prompt_length,
            temperature=1.0,
            bf16=True,
            logging_steps=1,
            save_steps=20,
            save_total_limit=2,
            seed=42 + stage_idx + train_round,
            report_to="none",
            generation_kwargs={"min_new_tokens": 64, "top_p": 0.95},
            **grpo_kw,
        )

        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=reward_fn,
            train_dataset=dataset,
            args=config,
        )

        if session:
            from transformers import TrainerCallback

            class _MetricsCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs and session:
                        session.record_trainer_log(logs, round_idx=train_round)

            trainer.add_callback(_MetricsCallback())

        print(f"  Starting GRPO for stage {stage.name}...", flush=True)
        trainer.train()

        if adaptive and _round_traces:
            adaptive.update(_round_traces)
            adaptive.save(session.session_dir / "adaptive_profile.json") if session else adaptive.save(
                Path(args.output) / "adaptive_profile.json"
            )
            print(f"  adaptive → difficulty={adaptive.difficulty:.2f}  "
                  f"mean_r={adaptive.recent_mean_reward:.3f}  "
                  f"std={adaptive.recent_reward_std:.3f}", flush=True)

        # Save stage checkpoint
        model.save_pretrained(str(stage_output))
        tokenizer.save_pretrained(str(stage_output))
        print(f"  Saved stage checkpoint → {stage_output}")

        # Record rewards and check advancement
        # In a real run, rewards come from the trainer — here we simulate
        # by evaluating the trained model on fresh bugs
        fresh_batch = generate_training_batch(
            stage.advance_window, stage,
            seed=999 + stage_idx,
        )
        for bug, messages in fresh_batch:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False, add_generation_prompt=True,
            )
            # Generate completion
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            import torch
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_completion_length,
                    temperature=0.7,
                    do_sample=True,
                )
            completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            breakdown = compute_reward(completion, bug, stage)
            advanced = state.record_reward(breakdown.total)
            # Log first 3 eval completions
            eval_idx = len(state.episode_rewards)
            if eval_idx <= 3:
                print(f"  [eval {eval_idx}] reward={breakdown.total}  components={breakdown.components}")
                print(f"    tools={breakdown.tool_calls_found}")
                preview = completion[:300].replace('\n', ' ')
                print(f"    completion: {preview}")
                print()
            if advanced:
                print(f"\n  >>> ADVANCED to stage {state.current_stage}! <<<\n")
                break

        state.save(CURRICULUM_STATE)

        if _time_up():
            break

    if state.is_complete:
        print("\nAll curriculum stages complete!")
        final_path = Path(args.output) / "final"
        model.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        print(f"Final model → {final_path}")
    elif _time_up():
        stage_output = Path(args.output) / f"stage_{state.current_stage}_{state.stage.name}"
        model.save_pretrained(str(stage_output))
        tokenizer.save_pretrained(str(stage_output))
        print(f"Session checkpoint → {stage_output}")

    if session:
        summary = session.finalize(
            adaptive_profile=adaptive.to_dict() if adaptive else None,
            extra={
                "stage": state.stage.name,
                "stage_idx": state.current_stage,
                "rounds": train_round,
                "episode_rewards_tail": state.episode_rewards[-20:],
            },
        )
        print(f"\n  Session summary → {summary}")
        print(f"  Traces → {session.session_dir / 'traces.jsonl'}")
        print(f"  Variance → {session.session_dir / 'group_stats.jsonl'}")
        print(f"  KL/grads → {session.session_dir / 'metrics.jsonl'}")
        print(f"  Plots → {session.session_dir / 'plots/'}")

    # Log
    _log_run(args.output, f"curricular GRPO, stages completed: {state.stage_history}")


# ── Self-distillation ──────────────────────────────────────────────────

def _self_distill(args) -> None:
    """Collect good traces from the model and save for SFT.

    VibeThinker's online self-distillation: generate many completions,
    keep the ones with reward > threshold, save as SFT training data.
    """
    from training.curriculum import STAGES
    from training.rl_env import generate_training_batch, compute_reward

    _free_gpu()
    model, tokenizer = _load_model(args.base_adapter)

    good_traces = []
    total_generated = 0

    for stage_idx, stage in enumerate(STAGES):
        if len(good_traces) >= args.distill_n:
            break

        batch = generate_training_batch(
            args.distill_n, stage,
            seed=42 + stage_idx * 100,
        )

        for bug, messages in batch:
            if len(good_traces) >= args.distill_n:
                break

            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False, add_generation_prompt=True,
            )

            # Generate multiple completions
            import torch
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            for _ in range(args.num_generations):
                total_generated += 1
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_completion_length,
                        temperature=0.9,
                        do_sample=True,
                    )
                completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                breakdown = compute_reward(completion, bug, stage)

                if breakdown.total >= args.distill_threshold:
                    # Save as SFT example
                    good_traces.append({
                        "messages": messages + [{"role": "assistant", "content": completion}],
                        "reward": breakdown.total,
                        "stage": stage.name,
                        "mutation_type": bug.mutation.mutation_type,
                        "depth": bug.depth,
                    })

        print(f"Stage {stage.name}: {len(good_traces)} good traces from {total_generated} total")

    # Save traces
    out_path = Path(DISTILL_DIR) / "distill_traces.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for trace in good_traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(good_traces)} good traces → {out_path}")
    print(f"Acceptance rate: {len(good_traces)/max(total_generated,1)*100:.1f}%")
    print(f"\nNext: SFT on these traces, then GRPO on harder cases.")


# ── Legacy mode (fixed levels) ────────────────────────────────────────

def _train_legacy(args) -> None:
    """Original GRPO on fixed toy_lgd levels (backward compat)."""
    if not PROMPTS_FILE.exists():
        print("Run: .venv/bin/python training/build_rl_dataset.py")
        sys.exit(1)

    prompts = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} legacy prompts", flush=True)

    _free_gpu()
    model, tokenizer = _load_model(args.base_adapter)

    from datasets import Dataset
    from training.rl_reward import score_completion

    ds_rows = []
    _level_map: dict[str, str] = {}
    for p in prompts:
        prompt_text = tokenizer.apply_chat_template(
            p["messages"],
            tokenize=False, add_generation_prompt=True,
        )
        ds_rows.append({"prompt": prompt_text, "level": p["level"]})
        _level_map[prompt_text[:200]] = p["level"]

    dataset = Dataset.from_list(ds_rows)

    def reward_fn(completions: list[str], prompts: list[str] | None = None, **kwargs) -> list[float]:
        rewards = []
        for i, text in enumerate(completions):
            prompt_key = (prompts[i] if prompts else "")[:200]
            level = _level_map.get(prompt_key, "01_easy_varswap")
            result = score_completion(text, level)
            rewards.append(result["reward"])
        return rewards

    from trl import GRPOConfig, GRPOTrainer

    config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs_per_stage,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=2048,
        temperature=0.9,
        bf16=True,
        logging_steps=1,
        save_steps=20,
        save_total_limit=2,
        seed=42,
        report_to="none",
        generation_kwargs={"min_new_tokens": 64},
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        args=config,
    )

    print("Starting legacy GRPO training...", flush=True)
    trainer.train()

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved → {args.output}")
    _log_run(args.output, f"legacy GRPO on {len(prompts)} fixed prompts")


# ── Eval mode ──────────────────────────────────────────────────────────

def _eval_procedural() -> None:
    """Evaluate the reward function on procedurally generated bugs."""
    from training.bug_generator import make_toy_generator
    from training.curriculum import STAGES
    from training.rl_env import compute_reward

    gen = make_toy_generator(seed=42)

    for stage in STAGES:
        print(f"\n{'='*50}")
        print(f"Stage: {stage.name}")

        bugs = gen.generate_batch(5)
        for bug in bugs:
            if stage.min_depth > 0 and bug.depth < stage.min_depth:
                continue

            # Good completion (mentions the right stuff)
            good = (
                f"Llamando trace_dependencies para {bug.changed_fields[0]}.\n"
                f"Llamando inspect_lineage para {bug.mutation.target_var}.\n"
                f"El error está en el paso {bug.mutation.step_name}: "
                f"{bug.mutation.description}.\n"
                f"La expresión original debería ser: {bug.mutation.original_expr}\n"
                f"Campos afectados: {', '.join(bug.changed_fields)}"
            )
            r_good = compute_reward(good, bug, stage)

            # Bad completion (hallucinated)
            bad = "El error está en la línea 42. Según mi análisis, el valor correcto es 0.123456789."
            r_bad = compute_reward(bad, bug, stage)

            print(f"  {bug.mutation.mutation_type} @ {bug.mutation.step_name} depth={bug.depth}")
            print(f"    good: reward={r_good.total:.3f}  components={r_good.components}")
            print(f"    bad:  reward={r_bad.total:.3f}  components={r_bad.components}")


# ── Helpers ────────────────────────────────────────────────────────────

def _autotune_grpo_memory(args) -> None:
    """Clamp GRPO batch/generations for 8B on 24GB VRAM."""
    import torch
    from training.tool_utils import is_8b_adapter

    if not torch.cuda.is_available():
        return
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    total_gb = torch.cuda.mem_get_info()[1] / 1e9
    is_8b = is_8b_adapter(args.base_adapter)

    if is_8b or total_gb < 26:
        args.num_generations = min(args.num_generations, 2)
        args.batch_size = min(args.batch_size, 2)
        args.max_completion_length = min(args.max_completion_length, 256)
        args.max_prompt_length = min(args.max_prompt_length, 1024)
        print(
            f"  GRPO autotune ({'8B' if is_8b else '24GB'}): "
            f"batch={args.batch_size} gen={args.num_generations} "
            f"prompt≤{args.max_prompt_length} compl≤{args.max_completion_length} "
            f"(free={free_gb:.1f}GB)",
            flush=True,
        )


def _load_model(adapter_path: str, max_seq_length: int = 2048):
    """Load model with QLoRA, handling PEFT adapters."""
    # Patch llm_blender compatibility with transformers 5.5.0
    import transformers.utils.hub as hub
    import os
    if not hasattr(hub, "TRANSFORMERS_CACHE"):
        hub.TRANSFORMERS_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

    from unsloth import FastLanguageModel
    from training.tool_utils import _base_hf_for_adapter
    import torch

    load_kw = dict(max_seq_length=max_seq_length, dtype=None, load_in_4bit=True)
    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        if free_gb < 8:
            load_kw["device_map"] = "auto"
            load_kw["max_memory"] = {0: "4GiB", "cpu": "48GiB"}

    ap = Path(adapter_path)
    is_peft = ap.exists() and (ap / "adapter_config.json").exists()
    base = _base_hf_for_adapter(adapter_path) if is_peft else adapter_path
    print(f"Loading model: base={base}  adapter={adapter_path}  peft={is_peft}")

    if is_peft:
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=base, **load_kw)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ap), is_trainable=True)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=base, **load_kw)
        model = FastLanguageModel.get_peft_model(
            model,
            r=32, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

    return model, tokenizer


def _log_run(output: str, note: str) -> None:
    log = PROJECT_ROOT / "training/output/iterate_log.json"
    history = json.loads(log.read_text()) if log.exists() else []
    history.append({
        "iteration": "grpo-curriculum",
        "adapter": output,
        "note": note,
    })
    log.write_text(json.dumps(history, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

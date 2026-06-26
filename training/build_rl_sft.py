#!/usr/bin/env python3
"""Build oracle SFT traces for any curriculum stage (warm-up before GRPO).

AgentWorld recipe: SFT activates the behavior pattern before RL sharpens it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def oracle_completion(bug, stage) -> str:
    from training.rl_env import _sim_inspect_lineage, _sim_trace_dependencies

    field = bug.changed_fields[0] if bug.changed_fields else bug.mutation.target_var
    step = bug.mutation.step_name
    var = bug.changed_fields[0] if bug.changed_fields else bug.mutation.target_var
    if var in ("IF_condition", "WHERE_filter") and bug.changed_fields:
        var = bug.changed_fields[0]

    parts = []
    # Stage 3+: query experience KB first for faster debugging
    if stage.name in ("multi_depth", "experience_accelerated", "experience_efficiency"):
        # Build a query rich in field names and keywords for better KB matching
        q_parts = [var, field, bug.mutation.mutation_type.replace("_", " ")]
        if bug.changed_fields:
            q_parts.extend(bug.changed_fields[:2])
        q = " ".join(q_parts)
        parts.append(
            f'<tool_call>{{"name": "search_experience", "arguments": {{"query": "{q}"}}}}</tool_call>'
        )

    parts.extend([
        f'<tool_call>{{"name": "trace_dependencies", "arguments": {{"target": "{field}"}}}}</tool_call>',
        f'<tool_call>{{"name": "inspect_lineage", "arguments": {{"target": "{field}"}}}}</tool_call>',
    ])

    answer = (
        f"Tras revisar el linaje, el bug está en el paso `{step}`.\n"
        f"PASO: {step}\n"
        f"VARIABLE: {var}\n"
        f"CORRECCIÓN: {bug.mutation.original_expr.strip()}"
    )
    if stage.name == "multi_depth" and bug.depth >= 1:
        answer = (
            f"El bug se origina en `{step}` (profundidad {bug.depth}) y afecta campos downstream.\n"
            + answer
        )
    parts.append(answer)
    return "\n".join(parts)


def build_dataset(stage_idx: int, n: int, seed: int = 7) -> list[dict]:
    from training.curriculum import STAGES
    from training.rl_env import build_prompt, generate_training_batch

    stage = STAGES[stage_idx]
    rows = []
    for bug, messages in generate_training_batch(n, stage, seed=seed):
        rows.append({
            "messages": messages + [{"role": "assistant", "content": oracle_completion(bug, stage)}],
            "stage": stage.name,
            "mutation_type": bug.mutation.mutation_type,
        })
    return rows


def run_warmup_sft(
    adapter: str,
    stage_idx: int,
    n: int = 80,
    steps: int = 80,
    seed: int = 7,
    out_jsonl: Path | None = None,
) -> str:
    """Build oracle data + short SFT on adapter. Returns adapter path."""
    rows = build_dataset(stage_idx, n, seed=seed)
    out_jsonl = out_jsonl or PROJECT_ROOT / f"training/data/rl_warmup_stage{stage_idx}.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows)} oracle examples → {out_jsonl}")

    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    from training.tool_utils import _base_hf_for_adapter

    adapter_path = PROJECT_ROOT / adapter
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"PEFT adapter not found: {adapter_path / 'adapter_config.json'}")

    base_hf = _base_hf_for_adapter(adapter_path)
    print(f"  Warmup SFT: base={base_hf}  adapter={adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_hf, max_seq_length=4096, dtype=None, load_in_4bit=True,
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)

    ds = Dataset.from_list([
        {"text": tokenizer.apply_chat_template(r["messages"], tokenize=False)} for r in rows
    ])
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=str(adapter_path / "warmup_sft"),
            max_steps=steps,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            logging_steps=20,
            save_steps=steps,
            bf16=True,
            report_to="none",
        ),
        dataset_text_field="text",
    )
    trainer.train()
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"  Warmup SFT saved → {adapter_path}")

    del trainer, model, tokenizer, ds
    from training.tool_utils import release_gpu
    release_gpu()
    return str(adapter_path)


def collect_session_traces(
    session_dir: Path,
    threshold: float = 0.35,
    max_n: int = 200,
) -> list[dict]:
    """Extract good completions from a session traces.jsonl for self-distillation."""
    traces_path = session_dir / "traces.jsonl"
    if not traces_path.exists():
        return []
    good = []
    for line in traces_path.read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        if t.get("reward", -1) >= threshold:
            good.append(t)
    good.sort(key=lambda x: -x["reward"])
    return good[:max_n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    run_warmup_sft(args.adapter, args.stage, n=args.n, steps=args.steps)


if __name__ == "__main__":
    main()

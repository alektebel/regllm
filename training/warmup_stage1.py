#!/usr/bin/env python3
"""Quick SFT warmup for Stage 1 — oracle tool+answer traces before GRPO."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "training/data/stage1_oracle_sft.jsonl"


def _oracle_completion(bug, stage) -> str:
    from training.rl_env import _sim_inspect_lineage, _sim_trace_dependencies

    field = bug.changed_fields[0]
    trace = _sim_trace_dependencies(bug.mutated_code, field)
    lineage = _sim_inspect_lineage(bug.mutated_code, field)
    step = bug.mutation.step_name
    var = bug.changed_fields[0] if bug.changed_fields else bug.mutation.target_var
    if var in ("IF_condition", "WHERE_filter"):
        var = bug.changed_fields[0]

    parts = [
        f'<tool_call>{{"name": "trace_dependencies", "arguments": {{"target": "{field}"}}}}</tool_call>',
        f'<tool_call>{{"name": "inspect_lineage", "arguments": {{"target": "{field}"}}}}</tool_call>',
        (
            f"Tras revisar el linaje, el bug está en el paso `{step}`.\n"
            f"PASO: {step}\n"
            f"VARIABLE: {var}\n"
            f"CORRECCIÓN: {bug.mutation.original_expr.strip()}"
        ),
    ]
    return "\n".join(parts)


def build_dataset(n: int = 80, seed: int = 7) -> list[dict]:
    from training.curriculum import STAGES
    from training.rl_env import build_prompt, generate_training_batch

    stage = STAGES[1]
    rows = []
    for bug, messages in generate_training_batch(n, stage, seed=seed):
        rows.append({
            "messages": messages + [{"role": "assistant", "content": _oracle_completion(bug, stage)}],
        })
    return rows


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="training/output/grpo-curriculum/stage_1_single_depth")
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()

    rows = build_dataset(args.n)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} oracle examples → {OUT}")

    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    from training.tool_utils import _base_hf_for_adapter

    adapter_path = Path(args.adapter)
    if not adapter_path.is_absolute():
        adapter_path = PROJECT_ROOT / adapter_path
    base_hf = _base_hf_for_adapter(adapter_path)
    print(f"Warmup SFT: base={base_hf}  adapter={adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_hf, max_seq_length=4096, dtype=None, load_in_4bit=True,
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)

    def fmt(row):
        return {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)}

    ds = Dataset.from_list([fmt(r) for r in rows])
    out_dir = Path(args.adapter) / "warmup_sft"
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=str(out_dir),
            max_steps=args.steps,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            logging_steps=10,
            save_steps=args.steps,
            bf16=True,
            report_to="none",
        ),
        dataset_text_field="text",
    )
    trainer.train()
    model.save_pretrained(str(Path(args.adapter)))
    print(f"Saved warmed-up adapter → {args.adapter}")


if __name__ == "__main__":
    main()

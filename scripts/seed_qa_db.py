#!/usr/bin/env python3
"""
Seed the qa_interactions table from existing QA datasets.

Loads entries from:
  - data/test_ground_truth.json  (up to --limit entries)
  - data/finetuning/*.jsonl      (up to 30 entries sampled randomly)

For each entry, optionally runs the fine-tuned model to generate model_answer
(skip with --no-model to store template/reference answers only).

Usage:
  python scripts/seed_qa_db.py --no-model --limit 20   # fast, no GPU needed
  python scripts/seed_qa_db.py --limit 50               # full run with model
  python scripts/seed_qa_db.py --checkpoint models/finetuned/run_X/final_model
"""

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


# ─── Data loading (reuse logic from test_with_judge) ──────────────────────────

def find_latest_checkpoint() -> Path | None:
    models_dir = PROJECT_ROOT / "models/finetuned"
    if not models_dir.exists():
        return None
    checkpoints = [
        ckpt
        for run in models_dir.iterdir()
        if run.is_dir() and run.name.startswith("run_")
        for ckpt in run.iterdir()
        if ckpt.is_dir()
        and (ckpt.name.startswith("checkpoint-") or ckpt.name == "final_model")
        and (ckpt / "adapter_model.safetensors").exists()
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def load_ground_truth(limit: int) -> list[dict]:
    path = PROJECT_ROOT / "data/test_ground_truth.json"
    if not path.exists():
        print(f"  Warning: {path} not found, skipping ground truth.")
        return []
    data = json.load(open(path))
    entries = data.get("entries", [])[:limit]
    return [
        {
            "question":          e["pregunta"],
            "reference_answer":  e["respuesta_esperada"],
            "category":          e.get("category", "regulatory"),
            "source":            "ground_truth",
        }
        for e in entries
    ]


def load_jsonl_sample(n: int, seed: int = 42) -> list[dict]:
    pool: list[dict] = []
    finetuning_dir = PROJECT_ROOT / "data/finetuning"
    if not finetuning_dir.exists():
        print(f"  Warning: {finetuning_dir} not found, skipping JSONL sample.")
        return []
    for path in sorted(finetuning_dir.rglob("*.jsonl")):
        with open(path) as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                msgs = item.get("messages", [])
                user = next((m["content"] for m in msgs if m["role"] == "user"), None)
                asst = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
                if user and asst:
                    pool.append({
                        "question":         user,
                        "reference_answer": asst,
                        "category":         path.stem,
                        "source":           path.name,
                    })
    random.seed(seed)
    return random.sample(pool, min(n, len(pool)))


# ─── Model inference ───────────────────────────────────────────────────────────

SYSTEM_STUDENT = (
    "Eres un asistente experto en regulación bancaria y el sector bancario español. "
    "Responde con datos precisos y cita la normativa cuando sea posible."
)


def load_model(checkpoint_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    # Patch accelerate bug (set vs list in no_split_module_classes)
    import accelerate.utils.modeling as _am
    _orig = _am.get_balanced_memory
    def _patched(model, max_memory=None, no_split_module_classes=None, **kwargs):
        if isinstance(no_split_module_classes, (set, list)):
            no_split_module_classes = list(no_split_module_classes)
        return _orig(model, max_memory=max_memory,
                     no_split_module_classes=no_split_module_classes, **kwargs)
    _am.get_balanced_memory = _patched

    base_model = "Qwen/Qwen2.5-7B-Instruct"
    print(f"  Loading base model : {base_model}")
    print(f"  Adapter            : {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question: str) -> tuple[str, int]:
    import torch
    t0 = time.monotonic()
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_STUDENT},
         {"role": "user",   "content": question}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    latency_ms = int((time.monotonic() - t0) * 1000)
    return answer, latency_ms


# ─── Main seeding logic ────────────────────────────────────────────────────────

async def seed(entries: list[dict], model=None, tokenizer=None):
    from src.db import init_db, log_qa_interaction

    print("\nInitializing database...")
    await init_db()
    print(f"Seeding {len(entries)} entries...")

    for entry in tqdm(entries, desc="Seeding", unit="row"):
        question         = entry["question"]
        reference_answer = entry["reference_answer"]
        category         = entry.get("category")
        source           = entry.get("source")

        if model is not None and tokenizer is not None:
            model_answer, latency_ms = generate_answer(model, tokenizer, question)
        else:
            # --no-model: store reference as model answer placeholder
            model_answer = reference_answer
            latency_ms   = None

        await log_qa_interaction(
            question=question,
            model_answer=model_answer,
            reference_answer=reference_answer,
            category=category,
            source=source,
            latency_ms=latency_ms,
        )

    print(f"Done — {len(entries)} rows inserted into qa_interactions.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Seed qa_interactions table")
    p.add_argument("--limit",      type=int, default=50,
                   help="Max entries to load from ground truth (default 50)")
    p.add_argument("--jsonl-limit", type=int, default=30,
                   help="Max entries to sample from JSONL files (default 30)")
    p.add_argument("--checkpoint",  default=None,
                   help="LoRA adapter path. Default: auto-detected latest.")
    p.add_argument("--no-model",    action="store_true",
                   help="Skip model inference; store reference answers as model_answer")
    p.add_argument("--seed",        type=int, default=42,
                   help="Random seed for JSONL sampling (default 42)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load data
    gt_entries   = load_ground_truth(args.limit)
    jsonl_entries = load_jsonl_sample(args.jsonl_limit, args.seed)
    all_entries  = gt_entries + jsonl_entries

    print(f"\nData loaded:")
    print(f"  Ground truth : {len(gt_entries)}")
    print(f"  JSONL sample : {len(jsonl_entries)}")
    print(f"  Total        : {len(all_entries)}")

    if not all_entries:
        print("No entries found. Exiting.")
        sys.exit(1)

    model = tokenizer = None
    if not args.no_model:
        ckpt = args.checkpoint
        if ckpt is None:
            ckpt_path = find_latest_checkpoint()
            if ckpt_path is None:
                print("No checkpoint found. Use --no-model or train first.")
                sys.exit(1)
            ckpt = str(ckpt_path)
            print(f"\nAuto-detected checkpoint: {ckpt}")
        print("\nLoading fine-tuned model (requires GPU)...")
        model, tokenizer = load_model(ckpt)
    else:
        print("\n--no-model: storing reference answers without inference.")

    asyncio.run(seed(all_entries, model, tokenizer))


if __name__ == "__main__":
    main()

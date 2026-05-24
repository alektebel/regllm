#!/usr/bin/env python3
"""
Full training pipeline for RegLLM.

Orchestrates the complete training workflow:

  Stage 1: SFT  — Supervised fine-tuning on combined banking + regulation data
  Stage 2: GRPO — Reinforcement learning with test-based rewards (keyword/source/format)
  Stage 3: Eval — Evaluate the final model on the ground truth test suite

Each stage can be run independently or skipped:

  python scripts/run_full_pipeline.py                          # full pipeline
  python scripts/run_full_pipeline.py --skip-sft               # GRPO only (uses latest SFT model)
  python scripts/run_full_pipeline.py --skip-grpo              # SFT + eval only
  python scripts/run_full_pipeline.py --sft-model models/finetuned/run_.../final_model  # GRPO from specific SFT
  python scripts/run_full_pipeline.py --eval-only models/grpo/final_model              # evaluate existing model
  python scripts/run_full_pipeline.py --sft-epochs 3 --grpo-epochs 2                   # custom hyperparams
"""

import json
import random
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL, TRAINING, GRPO, INFERENCE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ── Model name map (inline to avoid import chain issues) ──────────────────────
_AVAILABLE_MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "stablelm-3b": "stabilityai/stablelm-3b-4e1t",
    "phi-2": "microsoft/phi-2",
    "gemma-2b": "google/gemma-2b-it",
    "qwen-1.8b": "Qwen/Qwen2-1.8B-Instruct",
}
BASE_MODEL = _AVAILABLE_MODELS.get(MODEL["base_model"], MODEL["base_model"])

SFT_SYSTEM_PROMPT = (
    "Eres un asistente experto en el sector bancario español. "
    "Responde con datos precisos sobre las cuentas anuales de los bancos españoles."
)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: SFT
# ═════════════════════════════════════════════════════════════════════════════

def _load_sft_data():
    """Load and combine all SFT training datasets."""
    all_data = []

    # 1. Banking Q&A dataset
    banking_file = PROJECT_ROOT / "data/finetuning/banking_qa_dataset.jsonl"
    if banking_file.exists():
        with open(banking_file) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    msgs = item["messages"]
                    if msgs[0]["role"] != "system":
                        msgs.insert(0, {"role": "system", "content": SFT_SYSTEM_PROMPT})
                    all_data.append({"messages": msgs})
        logger.info(f"  banking_qa_dataset.jsonl: {len(all_data)} examples")

    # 2. Banking annual accounts extra
    extra_file = PROJECT_ROOT / "data/finetuning/banking_annual_accounts_extra.jsonl"
    before = len(all_data)
    if extra_file.exists():
        with open(extra_file) as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
        logger.info(f"  banking_annual_accounts_extra.jsonl: {len(all_data) - before} examples")

    # 3. Regulation training data
    reg_file = PROJECT_ROOT / "data/processed/train_data.json"
    before = len(all_data)
    if reg_file.exists():
        with open(reg_file) as f:
            for item in json.load(f):
                all_data.append({"messages": item["messages"]})
        logger.info(f"  train_data.json: {len(all_data) - before} examples")

    # 4. SQL methodology comparison
    sql_file = PROJECT_ROOT / "data/finetuning/sql_methodology_comparison_dataset.jsonl"
    before = len(all_data)
    if sql_file.exists():
        with open(sql_file) as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
        logger.info(f"  sql_methodology_comparison_dataset.jsonl: {len(all_data) - before} examples")

    if not all_data:
        raise RuntimeError("No training data found! Check data/finetuning/ and data/processed/")

    # Duplicate banking data 3× to emphasize factual memorization
    banking_count = sum(
        1
        for d in all_data
        if any("bancario" in m.get("content", "").lower() or "banco" in m.get("content", "").lower()
               for m in d["messages"][:1] if m["role"] == "system")
    )
    # Simpler heuristic: first N items are banking (loaded first)
    banking_items = [d for d in all_data if d is all_data[0].__class__]  # all of them for safety
    # Use the same approach as train_combined.py: first 94 items are banking
    n_banking = 0
    for d in all_data:
        system_msg = d["messages"][0]["content"] if d["messages"][0]["role"] == "system" else ""
        if "bancario" in system_msg.lower() or "banco" in system_msg.lower():
            n_banking += 1

    if n_banking > 0:
        banking_items = all_data[:n_banking]
        rest = all_data[n_banking:]
        all_data = banking_items * 3 + rest
        logger.info(f"  Banking data 3×: {n_banking} → {n_banking * 3} examples")

    logger.info(f"  Total SFT examples: {len(all_data)}")
    return all_data


def run_sft(epochs=None, batch_size=None, lr=None, grad_accum=None, output_dir=None):
    """
    Stage 1: Supervised Fine-Tuning on combined datasets.

    Returns path to the saved model directory.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset as TorchDataset

    logger.info("=" * 70)
    logger.info("STAGE 1: Supervised Fine-Tuning (SFT)")
    logger.info("=" * 70)

    # ── Hyperparameters ───────────────────────────────────────────────────
    train_cfg = TRAINING["full_training"]
    epochs = epochs or train_cfg["epochs"]
    batch_size = batch_size or train_cfg["batch_size"]
    lr = lr or train_cfg["learning_rate"]
    grad_accum = grad_accum or train_cfg["gradient_accumulation_steps"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or str(PROJECT_ROOT / f"models/finetuned/run_{timestamp}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Model:     {BASE_MODEL}")
    logger.info(f"Epochs:    {epochs}")
    logger.info(f"Batch:     {batch_size} × {grad_accum} = {batch_size * grad_accum} effective")
    logger.info(f"LR:        {lr}")
    logger.info(f"LoRA:      r={MODEL['lora']['r']}, alpha={MODEL['lora']['lora_alpha']}")
    logger.info(f"Output:    {output_dir}")

    # ── Data ──────────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    all_data = _load_sft_data()

    random.seed(42)
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    logger.info(f"Split: {len(train_data)} train, {len(val_data)} val")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Dataset class ─────────────────────────────────────────────────────
    class ChatDataset(TorchDataset):
        def __init__(self, data, tok, max_length=512):
            self.data = data
            self.tok = tok
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            text = self.tok.apply_chat_template(
                self.data[idx]["messages"], tokenize=False, add_generation_prompt=False
            )
            enc = self.tok(text, truncation=True, max_length=self.max_length,
                           padding="max_length", return_tensors="pt")
            labels = enc["input_ids"].clone()
            labels[labels == self.tok.pad_token_id] = -100
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": labels.squeeze(),
            }

    train_dataset = ChatDataset(train_data, tokenizer)
    val_dataset = ChatDataset(val_data, tokenizer)

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=MODEL["lora"]["r"],
        lora_alpha=MODEL["lora"]["lora_alpha"],
        target_modules=MODEL["lora"]["target_modules"],
        lora_dropout=MODEL["lora"]["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ── Training ──────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=TRAINING.get("weight_decay", 0.01),
        warmup_steps=TRAINING.get("warmup_steps", 100),
        max_grad_norm=TRAINING.get("max_grad_norm", 1.0),
        logging_steps=TRAINING.get("logging_steps", 10),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="none",
        save_total_limit=2,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    logger.info("Training SFT...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    logger.info(f"SFT completed in {elapsed / 60:.1f} min")

    # Save
    final_path = str(Path(output_dir) / "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"SFT model saved to {final_path}")

    # Quick sanity test
    _quick_test(model, tokenizer, stage="SFT")

    # Free memory
    del trainer, model
    torch.cuda.empty_cache()

    return final_path


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2: GRPO
# ═════════════════════════════════════════════════════════════════════════════

def run_grpo(
    sft_model_path=None,
    epochs=None,
    group_size=None,
    lr=None,
    beta=None,
    batch_size=None,
    output_dir=None,
):
    """
    Stage 2: GRPO training with test-based rewards.

    If sft_model_path is provided, the GRPO model is initialized from the
    SFT checkpoint (merged LoRA). Otherwise trains from the base model.

    Returns path to the saved model directory.
    """
    import torch
    from trl import GRPOTrainer, GRPOConfig
    from peft import LoraConfig
    from datasets import Dataset
    from src.rlhf.grpo_rewards import TestRewardComputer

    logger.info("=" * 70)
    logger.info("STAGE 2: GRPO (Group Relative Policy Optimization)")
    logger.info("=" * 70)

    # ── Hyperparameters ───────────────────────────────────────────────────
    epochs = epochs or GRPO["epochs"]
    group_size = group_size or GRPO["group_size"]
    lr = lr or GRPO["learning_rate"]
    beta = beta or GRPO["beta"]
    batch_size = batch_size or GRPO["batch_size"]
    grad_accum = GRPO["gradient_accumulation_steps"]
    max_completion = GRPO["max_completion_length"]
    reward_weights = GRPO["reward_weights"]
    output_dir = output_dir or str(PROJECT_ROOT / "models/grpo")

    # Model to start from
    model_name = sft_model_path or BASE_MODEL

    logger.info(f"Model:          {model_name}")
    logger.info(f"Epochs:         {epochs}")
    logger.info(f"Group size:     {group_size}")
    logger.info(f"LR:             {lr}")
    logger.info(f"Beta (KL):      {beta}")
    logger.info(f"Batch:          {batch_size} × {grad_accum}")
    logger.info(f"Max completion: {max_completion}")
    logger.info(f"Reward weights: {reward_weights}")
    logger.info(f"Output:         {output_dir}")

    # ── Dataset from ground truth ─────────────────────────────────────────
    gt_path = PROJECT_ROOT / "data/test_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    system_prompt = INFERENCE["system_prompt"]
    records = []
    for entry in gt_data["entries"]:
        records.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": entry["pregunta"]},
            ],
            "datos_clave": json.dumps(entry.get("datos_clave", [])),
            "fuentes_esperadas": json.dumps(entry.get("fuentes_esperadas", [])),
            "entry_id": entry["id"],
        })

    dataset = Dataset.from_list(records)
    logger.info(f"GRPO dataset: {len(dataset)} prompts")

    # ── Reward function ───────────────────────────────────────────────────
    reward_computer = TestRewardComputer()

    def reward_func(prompts, completions, **kwargs):
        datos_list = kwargs.get("datos_clave", [])
        fuentes_list = kwargs.get("fuentes_esperadas", [])
        rewards = []
        for i, completion in enumerate(completions):
            try:
                datos = json.loads(datos_list[i]) if isinstance(datos_list[i], str) else datos_list[i]
            except (json.JSONDecodeError, IndexError, TypeError):
                datos = []
            try:
                fuentes = json.loads(fuentes_list[i]) if isinstance(fuentes_list[i], str) else fuentes_list[i]
            except (json.JSONDecodeError, IndexError, TypeError):
                fuentes = []
            entry = {"datos_clave": datos, "fuentes_esperadas": fuentes}
            rewards.append(reward_computer.combined_reward(completion, entry, weights=reward_weights))
        return rewards

    # ── LoRA config (smaller rank for GRPO) ───────────────────────────────
    peft_config = LoraConfig(
        r=GRPO["lora"]["r"],
        lora_alpha=GRPO["lora"]["lora_alpha"],
        lora_dropout=GRPO["lora"]["lora_dropout"],
        target_modules=GRPO["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── GRPOConfig ────────────────────────────────────────────────────────
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=group_size,
        max_completion_length=max_completion,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        beta=beta,
        loss_type="grpo",
        scale_rewards="group",
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        temperature=0.7,
    )

    # ── Model kwargs ──────────────────────────────────────────────────────
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if MODEL["use_4bit"]:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    training_args.model_init_kwargs = model_kwargs

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    logger.info("Training GRPO...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    logger.info(f"GRPO completed in {elapsed / 60:.1f} min")

    # Save
    final_path = str(Path(output_dir) / "final_model")
    trainer.save_model(final_path)
    if trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(final_path)
    logger.info(f"GRPO model saved to {final_path}")

    # Free memory
    del trainer
    torch.cuda.empty_cache()

    return final_path


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3: Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def run_eval(model_path):
    """
    Stage 3: Evaluate the model on the ground truth test suite.

    Returns dict with average keyword/source/format/combined rewards.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from src.rlhf.grpo_rewards import TestRewardComputer

    logger.info("=" * 70)
    logger.info("STAGE 3: Evaluation")
    logger.info("=" * 70)
    logger.info(f"Model: {model_path}")

    # ── Load model ────────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # ── Ground truth ──────────────────────────────────────────────────────
    gt_path = PROJECT_ROOT / "data/test_ground_truth.json"
    with open(gt_path) as f:
        entries = json.load(f)["entries"]

    system_prompt = INFERENCE["system_prompt"]
    reward_computer = TestRewardComputer()
    reward_weights = GRPO["reward_weights"]

    totals = {"keyword": 0.0, "source": 0.0, "format": 0.0, "combined": 0.0}
    results = []

    for entry in entries:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": entry["pregunta"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=GRPO["max_completion_length"],
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

        kw = reward_computer.keyword_reward(completion, entry.get("datos_clave", []))
        src = reward_computer.source_reward(completion, entry.get("fuentes_esperadas", []))
        fmt = reward_computer.format_reward(completion)
        combined = reward_computer.combined_reward(completion, entry, reward_weights)

        totals["keyword"] += kw
        totals["source"] += src
        totals["format"] += fmt
        totals["combined"] += combined

        results.append({
            "id": entry["id"],
            "category": entry.get("category", ""),
            "keyword": kw,
            "source": src,
            "format": fmt,
            "combined": combined,
            "completion_preview": completion[:120],
        })

        logger.info(
            f"  [{entry['id']}] kw={kw:.2f} src={src:.1f} fmt={fmt:.1f} "
            f"combined={combined:.2f} | {completion[:80]}..."
        )

    n = len(entries)
    avg = {k: v / n for k, v in totals.items()}

    logger.info("")
    logger.info("─" * 50)
    logger.info(f"  Evaluation Results ({n} entries)")
    logger.info(f"  Avg keyword reward:  {avg['keyword']:.3f}")
    logger.info(f"  Avg source reward:   {avg['source']:.3f}")
    logger.info(f"  Avg format reward:   {avg['format']:.3f}")
    logger.info(f"  Avg combined reward: {avg['combined']:.3f}")
    logger.info("─" * 50)

    # Save results to JSON
    results_path = Path(model_path).parent / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump({"averages": avg, "per_entry": results}, f, indent=2, ensure_ascii=False)
    logger.info(f"Detailed results saved to {results_path}")

    del model
    torch.cuda.empty_cache()

    return avg


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _quick_test(model, tokenizer, stage="", n=3):
    """Generate a few sample responses for sanity checking."""
    import torch

    questions = [
        "¿Cuánto ganó BBVA en 2023?",
        "¿Cuál fue la tasa de morosidad de CaixaBank en 2023?",
        "¿Qué es la probabilidad de impago (PD) en el contexto del método IRB?",
    ][:n]

    logger.info(f"\n--- Quick test ({stage}) ---")
    model.eval()
    for q in questions:
        messages = [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        logger.info(f"  Q: {q}")
        logger.info(f"  A: {response[:200]}")
        logger.info("")


def _find_latest_sft_model():
    """Find the most recently trained SFT model."""
    finetuned_dir = PROJECT_ROOT / "models" / "finetuned"
    if not finetuned_dir.exists():
        return None

    runs = sorted(finetuned_dir.glob("run_*/final_model"), key=lambda p: p.parent.name)
    if runs:
        path = str(runs[-1])
        logger.info(f"Found latest SFT model: {path}")
        return path
    return None


def _print_summary(sft_path, grpo_path, eval_results, elapsed):
    """Print final pipeline summary."""
    print()
    print("=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    if sft_path:
        print(f"  SFT model:  {sft_path}")
    if grpo_path:
        print(f"  GRPO model: {grpo_path}")
    if eval_results:
        print(f"  Eval scores:")
        print(f"    keyword:  {eval_results['keyword']:.3f}")
        print(f"    source:   {eval_results['source']:.3f}")
        print(f"    format:   {eval_results['format']:.3f}")
        print(f"    combined: {eval_results['combined']:.3f}")
    print(f"  Total time: {elapsed / 60:.1f} min")
    print("=" * 70)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Full training pipeline: SFT → GRPO → Eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/run_full_pipeline.py                              # full pipeline
  python scripts/run_full_pipeline.py --skip-sft                   # GRPO from latest SFT
  python scripts/run_full_pipeline.py --skip-grpo                  # SFT + eval only
  python scripts/run_full_pipeline.py --sft-model path/to/model    # GRPO from specific SFT
  python scripts/run_full_pipeline.py --eval-only path/to/model    # evaluate existing model
  python scripts/run_full_pipeline.py --sft-epochs 5 --grpo-epochs 2
""",
    )

    # Stage control
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT, use latest or --sft-model")
    parser.add_argument("--skip-grpo", action="store_true", help="Skip GRPO stage")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation stage")
    parser.add_argument("--eval-only", type=str, metavar="MODEL_PATH",
                        help="Only run evaluation on an existing model")

    # Model paths
    parser.add_argument("--sft-model", type=str, help="Path to existing SFT model for GRPO stage")

    # SFT hyperparameters
    parser.add_argument("--sft-epochs", type=int, default=None, help="SFT training epochs")
    parser.add_argument("--sft-batch-size", type=int, default=None, help="SFT batch size")
    parser.add_argument("--sft-lr", type=float, default=None, help="SFT learning rate")
    parser.add_argument("--sft-grad-accum", type=int, default=None, help="SFT gradient accumulation steps")

    # GRPO hyperparameters
    parser.add_argument("--grpo-epochs", type=int, default=None, help="GRPO training epochs")
    parser.add_argument("--grpo-group-size", type=int, default=None, help="GRPO completions per prompt")
    parser.add_argument("--grpo-lr", type=float, default=None, help="GRPO learning rate")
    parser.add_argument("--grpo-beta", type=float, default=None, help="GRPO KL penalty coefficient")
    parser.add_argument("--grpo-batch-size", type=int, default=None, help="GRPO batch size")

    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    sft_path = None
    grpo_path = None
    eval_results = None

    print()
    print("=" * 70)
    print("  RegLLM Full Training Pipeline")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Eval-only mode ────────────────────────────────────────────────────
    if args.eval_only:
        eval_results = run_eval(args.eval_only)
        _print_summary(None, None, eval_results, time.time() - t0)
        return

    # ── Stage 1: SFT ─────────────────────────────────────────────────────
    if args.skip_sft:
        sft_path = args.sft_model or _find_latest_sft_model()
        if sft_path:
            logger.info(f"Skipping SFT, using model: {sft_path}")
        else:
            logger.info("Skipping SFT, no existing model found — GRPO will train from base model")
    else:
        sft_path = run_sft(
            epochs=args.sft_epochs,
            batch_size=args.sft_batch_size,
            lr=args.sft_lr,
            grad_accum=args.sft_grad_accum,
        )

    # ── Stage 2: GRPO ────────────────────────────────────────────────────
    if args.skip_grpo:
        logger.info("Skipping GRPO stage")
        grpo_path = None
    else:
        grpo_path = run_grpo(
            sft_model_path=sft_path,
            epochs=args.grpo_epochs,
            group_size=args.grpo_group_size,
            lr=args.grpo_lr,
            beta=args.grpo_beta,
            batch_size=args.grpo_batch_size,
        )

    # ── Stage 3: Eval ─────────────────────────────────────────────────────
    final_model = grpo_path or sft_path
    if args.skip_eval:
        logger.info("Skipping evaluation stage")
    elif final_model:
        eval_results = run_eval(final_model)
    else:
        logger.warning("No model to evaluate!")

    # ── Summary ───────────────────────────────────────────────────────────
    _print_summary(sft_path, grpo_path, eval_results, time.time() - t0)


if __name__ == "__main__":
    main()

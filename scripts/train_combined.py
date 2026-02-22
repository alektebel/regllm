#!/usr/bin/env python3
"""
Training script that auto-discovers all QA datasets and trains with LoRA/4-bit.
By default resumes from the latest checkpoint found in models/finetuned/.
"""

import argparse
import json
import random
import torch
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM_PROMPT = (
    "Eres un asistente experto en regulación bancaria y el sector bancario español. "
    "Responde con datos precisos y cita la normativa cuando sea posible."
)


# ─── Checkpoint discovery ────────────────────────────────────────────────────

def find_latest_checkpoint() -> Path | None:
    """Return the most recently modified checkpoint directory, or None."""
    models_dir = PROJECT_ROOT / "models/finetuned"
    if not models_dir.exists():
        return None

    checkpoints = []
    for run_dir in models_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        for ckpt in run_dir.iterdir():
            if ckpt.is_dir() and ckpt.name.startswith("checkpoint-"):
                # Verify it's a real HF checkpoint (has optimizer state)
                if (ckpt / "optimizer.pt").exists():
                    checkpoints.append(ckpt)

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


# ─── Data loading ────────────────────────────────────────────────────────────

def _normalize(item: dict, fallback_system: str | None = None) -> dict | None:
    """Convert various formats to {messages: [...]} with a system prompt."""
    if "messages" in item:
        msgs = list(item["messages"])
        if msgs and msgs[0]["role"] != "system" and fallback_system:
            msgs.insert(0, {"role": "system", "content": fallback_system})
        return {"messages": msgs}

    # {input, output} format
    if "input" in item and "output" in item:
        msgs = []
        sys_content = item.get("system", fallback_system)
        if sys_content:
            msgs.append({"role": "system", "content": sys_content})
        msgs.append({"role": "user", "content": item["input"]})
        msgs.append({"role": "assistant", "content": item["output"]})
        return {"messages": msgs}

    return None


def _load_file(path: Path) -> list[dict]:
    """Load a .jsonl or .json file; return list of normalised items."""
    items = []
    try:
        if path.suffix == ".jsonl":
            with open(path) as f:
                raw = [json.loads(l) for l in f if l.strip()]
        else:  # .json
            with open(path) as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                # Might be a dict with a top-level list key
                raw = raw.get("data", raw.get("examples", raw.get("messages", [])))
            if not isinstance(raw, list):
                return []

        for obj in raw:
            if not isinstance(obj, dict):
                continue
            norm = _normalize(obj, fallback_system=SYSTEM_PROMPT)
            if norm and len(norm["messages"]) >= 2:
                items.append(norm)
    except Exception as e:
        logger.warning(f"  Skipped {path.name}: {e}")
    return items


def load_all_data() -> list[dict]:
    """Auto-discover and load every QA dataset under data/finetuning/ and data/processed/."""
    all_data: list[dict] = []

    # 1. Scan data/finetuning/ for every .jsonl / .json
    finetuning_dir = PROJECT_ROOT / "data/finetuning"
    if finetuning_dir.exists():
        for path in sorted(finetuning_dir.rglob("*")):
            if path.suffix in (".jsonl", ".json") and path.is_file():
                before = len(all_data)
                all_data.extend(_load_file(path))
                logger.info(f"  {path.relative_to(PROJECT_ROOT)}: {len(all_data) - before} examples")

    # 2. data/processed/train_data.json (regulation Q&A)
    reg_file = PROJECT_ROOT / "data/processed/train_data.json"
    if reg_file.exists():
        before = len(all_data)
        all_data.extend(_load_file(reg_file))
        logger.info(f"  {reg_file.relative_to(PROJECT_ROOT)}: {len(all_data) - before} examples")

    logger.info(f"Total examples loaded: {len(all_data)}")
    return all_data


# ─── Dataset ─────────────────────────────────────────────────────────────────

class ChatDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer, max_length: int = 768):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train combined banking/regulation QA model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume from. "
             "Defaults to the latest checkpoint found automatically. "
             "Pass 'none' to start fresh.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=768)
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Resolve checkpoint ──
    resume_from: str | None = None
    if args.checkpoint and args.checkpoint.lower() != "none":
        resume_from = args.checkpoint
        logger.info(f"Resuming from explicit checkpoint: {resume_from}")
    else:
        latest = find_latest_checkpoint()
        if latest and (not args.checkpoint or args.checkpoint.lower() != "none"):
            resume_from = str(latest)
            logger.info(f"Auto-detected latest checkpoint: {resume_from}")
        else:
            logger.info("Starting fresh (no checkpoint found or --checkpoint none).")

    # ── Load data ──
    logger.info("Discovering QA datasets...")
    all_data = load_all_data()
    if not all_data:
        raise RuntimeError("No training data found! Check data/finetuning/")

    random.seed(42)
    random.shuffle(all_data)
    split = int(len(all_data) * 0.9)
    train_data, val_data = all_data[:split], all_data[split:]
    logger.info(f"Train: {len(train_data)}  Val: {len(val_data)}")

    # ── Tokenizer ──
    logger.info(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Model ──
    logger.info(f"Loading model: {BASE_MODEL}")
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
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Datasets ──
    train_dataset = ChatDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = ChatDataset(val_data, tokenizer, max_length=args.max_length)

    # ── Output dir ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / f"models/finetuned/run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training args ──
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=30,
        max_grad_norm=1.0,
        logging_steps=10,
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

    # ── Train ──
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save ──
    final_path = output_dir / "final_model"
    logger.info(f"Saving model to {final_path}")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"Training complete. Model saved to {final_path}")

    # ── Quick test ──
    logger.info("\n=== Quick test ===")
    model.eval()
    test_questions = [
        "¿Qué es el capital de nivel 1 (Tier 1) según Basilea III?",
        "¿Cuánto ganó BBVA en 2023?",
        "¿Qué ratio de morosidad exige la EBA para los bancos IRB?",
        "¿Cuál es el objetivo del CRR (Capital Requirements Regulation)?",
    ]
    for q in test_questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        logger.info(f"Q: {q}")
        logger.info(f"A: {response}\n")


if __name__ == "__main__":
    main()

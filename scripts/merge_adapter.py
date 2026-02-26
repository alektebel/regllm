#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and save as full HuggingFace weights.

Output: models/merged/  (float16 safetensors shards, ~14 GB)

This is required before GGUF conversion — llama.cpp's convert script needs
the full unquantized weights, not a LoRA delta.

Usage:
    python scripts/merge_adapter.py
    python scripts/merge_adapter.py --adapter models/finetuned/run_.../final_model
    python scripts/merge_adapter.py --device cuda   # faster, needs ~14 GB VRAM
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_latest_adapter() -> Path:
    root = PROJECT_ROOT / "models" / "finetuned"
    candidates = sorted(
        [p for p in root.glob("run_*/final_model") if (p / "adapter_model.safetensors").exists()]
        + [p for p in root.glob("run_*/checkpoint-*") if (p / "adapter_model.safetensors").exists()],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No adapter found under {root}/\n"
            "Train first: python scripts/train_combined.py"
        )
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--adapter", default=None,
        help="Path to LoRA adapter dir (auto-detects latest if omitted)",
    )
    parser.add_argument(
        "--output", default="models/merged",
        help="Output directory for merged model (default: models/merged)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for loading weights: cpu (safe, default) or cuda (faster)",
    )
    args = parser.parse_args()

    adapter_path = (
        Path(args.adapter).resolve() if args.adapter else find_latest_adapter()
    )
    output_path = (PROJECT_ROOT / args.output).resolve()

    logger.info(f"Adapter : {adapter_path}")
    logger.info(f"Output  : {output_path}")
    logger.info(f"Device  : {args.device}")

    if (output_path / "config.json").exists():
        logger.info("Merged model already exists — skipping.")
        logger.info("Delete models/merged/ to force a redo.")
        sys.exit(0)

    output_path.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer from adapter dir...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)

    # ── Base model (float16, full precision) ───────────────────────────────────
    logger.info(f"Loading base model {BASE_MODEL} in float16 on {args.device}...")
    logger.info("  Requires ~14 GB RAM (or VRAM if --device cuda).")
    logger.info("  Expect 3-10 minutes on CPU — grab a coffee.")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map=args.device,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # ── Apply LoRA adapter ─────────────────────────────────────────────────────
    logger.info("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base, str(adapter_path))

    # ── Merge and unload ───────────────────────────────────────────────────────
    logger.info("Merging LoRA weights into base (merge_and_unload)...")
    merged = model.merge_and_unload()
    merged.eval()

    # ── Save ───────────────────────────────────────────────────────────────────
    logger.info(f"Saving merged model to {output_path} ...")
    merged.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(str(output_path))

    logger.info("")
    logger.info("✓  Merged model saved.")
    logger.info(f"   Path: {output_path}")


if __name__ == "__main__":
    main()

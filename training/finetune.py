"""Two-phase fine-tuning for RegLLM Qwen3-3B.

Phase 1: Continued pretraining (full parameters) — domain knowledge
Phase 2: SFT with QLoRA — task-specific behavior (tool calling, etc.)

Requirements:
    pip install unsloth transformers datasets peft trl pyyaml

Usage:
    # Full pipeline:
    python training/build_corpus.py                  # build Phase 1 corpus
    python training/generate_training_data.py        # build Phase 2 dataset
    python training/finetune.py --phase 1            # domain pretraining
    python training/finetune.py --phase 2            # SFT on tools/tasks
    python training/finetune.py --phase export       # GGUF export for Ollama

    # Or run both phases sequentially:
    python training/finetune.py --phase all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def phase1_continued_pretraining(cfg: dict) -> str:
    """Full-parameter continued pretraining on domain corpus.

    Returns the output directory path for use as Phase 2 base model.
    """
    from unsloth import FastLanguageModel

    p1 = cfg["phase1"]
    base = cfg["base_model"]
    output_dir = str(PROJECT_ROOT / p1["output_dir"])

    corpus_path = PROJECT_ROOT / p1["dataset"]
    if not corpus_path.exists():
        print(
            f"ERROR: Corpus not found at {corpus_path}\n"
            "Run: python training/build_corpus.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"=== Phase 1: Continued Pretraining (full params) ===")
    print(f"Base model: {base}")
    print(f"Corpus: {corpus_path}")

    load_4bit = p1.get("load_in_4bit", False)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base,
        max_seq_length=p1["max_seq_length"],
        dtype=None,
        load_in_4bit=load_4bit,
    )

    if load_4bit:
        lora_r = p1.get("lora_r", 64)
        lora_alpha = p1.get("lora_alpha", lora_r)
        print(f"Phase 1 QLoRA domain CPT (r={lora_r}) — fits 3090 24GB")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
        )
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "embed_tokens", "lm_head",
            ],
            use_gradient_checkpointing="unsloth",
        )

    # Load corpus as a text dataset
    from datasets import Dataset

    corpus_text = corpus_path.read_text(encoding="utf-8")
    # Split on document separator
    documents = [d.strip() for d in corpus_text.split("<|endoftext|>") if d.strip()]
    print(f"Documents: {len(documents)}")

    # Tokenize into chunks of max_seq_length
    chunks: list[str] = []
    for doc in documents:
        tokens = tokenizer.encode(doc, add_special_tokens=False)
        max_len = p1["max_seq_length"] - 2  # room for special tokens
        for i in range(0, len(tokens), max_len):
            chunk_tokens = tokens[i:i + max_len]
            chunk_text = tokenizer.decode(chunk_tokens)
            if len(chunk_tokens) > 64:  # skip tiny fragments
                chunks.append(chunk_text)

    print(f"Training chunks: {len(chunks)}")
    dataset = Dataset.from_dict({"text": chunks})

    # Train
    from trl import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=p1["num_train_epochs"],
            per_device_train_batch_size=p1["per_device_train_batch_size"],
            gradient_accumulation_steps=p1["gradient_accumulation_steps"],
            learning_rate=p1["learning_rate"],
            weight_decay=p1["weight_decay"],
            warmup_ratio=p1["warmup_ratio"],
            lr_scheduler_type=p1["lr_scheduler_type"],
            max_seq_length=p1["max_seq_length"],
            bf16=p1["bf16"],
            logging_steps=p1["logging_steps"],
            save_steps=p1["save_steps"],
            save_total_limit=p1["save_total_limit"],
            seed=p1["seed"],
            dataset_text_field="text",
            packing=True,
            optim=p1.get("optim", "adamw_8bit"),
            report_to="none",
        ),
    )

    print("Starting Phase 1 training...")
    trainer.train()

    # Save the full model
    print(f"Saving Phase 1 model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Also save merged (non-LoRA) version for Phase 2
    merged_dir = output_dir + "-merged"
    print(f"Saving merged model to {merged_dir}")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    return merged_dir


def phase2_sft(cfg: dict, base_override: str | None = None) -> str:
    """SFT with QLoRA on task-specific data (tool calling, investigation flow).

    Returns the output directory path.
    """
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    p2 = cfg["phase2"]
    base = base_override or p2.get("base_model_override") or cfg["base_model"]
    output_dir = str(PROJECT_ROOT / p2["output_dir"])
    lora_cfg = p2["lora"]

    ds_cfg = p2["dataset"]
    train_file = PROJECT_ROOT / ds_cfg["train_file"]
    eval_file = PROJECT_ROOT / ds_cfg["eval_file"]

    if not train_file.exists():
        print(
            f"ERROR: Training data not found at {train_file}\n"
            "Run: python training/generate_training_data.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"=== Phase 2: SFT with QLoRA ===")
    print(f"Base model: {base}")

    # Load with 4-bit quantization for QLoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base,
        max_seq_length=p2["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing="unsloth",
    )

    # Load chat dataset
    dataset = load_dataset("json", data_files={
        "train": str(train_file),
        "eval": str(eval_file) if eval_file.exists() else str(train_file),
    })

    # Load tool schemas so the training data includes the same tool
    # definition block that Ollama injects at inference time.
    import sys as _sys
    if str(PROJECT_ROOT) not in _sys.path:
        _sys.path.insert(0, str(PROJECT_ROOT))
    from src.agent.tools import tool_schemas
    _tool_defs = tool_schemas([
        "trace_dependencies", "inspect_lineage",
        "search_docs", "search_regulation", "get_field_definition",
    ])

    def format_chat(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tools=_tool_defs,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_chat)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=p2["num_train_epochs"],
            per_device_train_batch_size=p2["per_device_train_batch_size"],
            gradient_accumulation_steps=p2["gradient_accumulation_steps"],
            learning_rate=p2["learning_rate"],
            weight_decay=p2["weight_decay"],
            warmup_ratio=p2["warmup_ratio"],
            lr_scheduler_type=p2["lr_scheduler_type"],
            max_seq_length=p2["max_seq_length"],
            bf16=p2["bf16"],
            logging_steps=p2["logging_steps"],
            save_steps=p2["save_steps"],
            eval_steps=p2["eval_steps"],
            eval_strategy="steps" if eval_file.exists() else "no",
            save_total_limit=p2["save_total_limit"],
            seed=p2["seed"],
            dataset_text_field="text",
            packing=False,
            report_to="none",
        ),
    )

    print("Starting Phase 2 training...")
    trainer.train()

    print(f"Saving Phase 2 model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir


def export_gguf(cfg: dict, model_dir: str | None = None) -> None:
    """Export the trained model to GGUF for Ollama serving.

    Loads the base model + LoRA adapter, merges weights, then converts.
    """
    from unsloth import FastLanguageModel
    from peft import PeftModel

    p2 = cfg["phase2"]
    export_cfg = cfg.get("export", {})
    quant = export_cfg.get("quantization", "q4_K_M")
    adapter_path = model_dir or str(PROJECT_ROOT / p2["output_dir"])

    # Determine base model: check adapter_config, then config overrides
    adapter_cfg_file = Path(adapter_path) / "adapter_config.json"
    if adapter_cfg_file.exists():
        adapter_cfg = json.loads(adapter_cfg_file.read_text())
        base = adapter_cfg.get("base_model_name_or_path", cfg["base_model"])
    else:
        base = p2.get("base_model_override") or cfg["base_model"]

    print(f"=== Export to GGUF ({quant}) ===")
    print(f"Base model: {base}")
    print(f"Adapter: {adapter_path}")

    # Step 1: Merge LoRA into base model and save as fp16
    merged_dir = Path(adapter_path) / "merged_fp16"
    if not (merged_dir / "model.safetensors.index.json").exists():
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base,
            max_seq_length=p2["max_seq_length"],
            dtype=None,
            load_in_4bit=True,
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        print("Merging LoRA weights and saving fp16...")
        model.save_pretrained_merged(
            str(merged_dir), tokenizer, save_method="merged_16bit",
        )
        del model
        import torch
        torch.cuda.empty_cache()
    else:
        print(f"Using existing merged model at {merged_dir}")

    # Step 2: Load merged model and export to GGUF
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(merged_dir),
        max_seq_length=p2["max_seq_length"],
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    gguf_dir = Path(adapter_path) / "gguf"
    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method=quant,
    )

    # Generate Ollama Modelfile
    ollama_name = export_cfg.get("ollama_model_name", "regllm-3b")
    gguf_files = list(gguf_dir.glob("*.gguf"))
    if gguf_files:
        modelfile = gguf_dir / "Modelfile"
        modelfile.write_text(
            f'FROM ./{gguf_files[0].name}\n'
            f'PARAMETER temperature 0.1\n'
            f'PARAMETER num_ctx {p2["max_seq_length"]}\n'
            f'SYSTEM "Eres un investigador de cumplimiento regulatorio para bases de datos '
            f'bancarias (COREP/FINREP IRB). Respondes en español profesional."\n',
            encoding="utf-8",
        )
        print(f"\nTo serve with Ollama:")
        print(f"  cd {gguf_dir}")
        print(f"  ollama create {ollama_name} -f Modelfile")
        print(f"  ollama run {ollama_name}")

    print("\nExport done!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 for RegLLM")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--phase", choices=["1", "2", "export", "all"], required=True,
                        help="Which phase to run: 1 (pretrain), 2 (SFT), export (GGUF), all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.dry_run:
        print(json.dumps(cfg, indent=2, default=str))
        return

    if args.phase == "1":
        phase1_continued_pretraining(cfg)
    elif args.phase == "2":
        phase2_sft(cfg)
    elif args.phase == "export":
        export_gguf(cfg)
    elif args.phase == "all":
        merged_dir = phase1_continued_pretraining(cfg)
        output_dir = phase2_sft(cfg, base_override=merged_dir)
        export_gguf(cfg, model_dir=output_dir)


if __name__ == "__main__":
    main()

"""QLoRA SFT for RegLLM tool-calling agents (any Unsloth-supported base)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LITE_TOOL_NAMES = [
    "trace_causal_chain", "query_cycle_data", "get_sas_formula",
    "trace_dependencies", "inspect_lineage",
    "search_docs", "search_regulation", "get_field_definition",
]


def _load_model_preset(name: str) -> dict:
    import yaml
    cfg = yaml.safe_load((PROJECT_ROOT / "training/sft_models.yaml").read_text())
    models = cfg.get("models", {})
    if name not in models:
        raise KeyError(f"Unknown model {name!r}. Choose from: {list(models)}")
    return models[name]


def _resolve_base_hf(model_key: str | None, adapter_in: str | None) -> tuple[str, dict]:
    """Return (base_hf_id, preset_meta)."""
    if model_key:
        preset = _load_model_preset(model_key)
        return preset["hf_id"], preset
    base = "Qwen/Qwen3-4B"
    preset: dict = {"model_key": "qwen3-4b", "base_hf": base, "lora_r": 32, "lora_alpha": 32}
    if adapter_in:
        meta_path = Path(adapter_in) / "regllm_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            base = meta.get("base_hf", base)
            preset.update(meta)
        else:
            cfg_path = Path(adapter_in) / "adapter_config.json"
            if cfg_path.exists():
                base = json.loads(cfg_path.read_text()).get("base_model_name_or_path", base)
                preset["base_hf"] = base
    return base, preset


def run_sft(
    model_key: str | None = None,
    output_dir: str | None = None,
    *,
    adapter_in: str | None = None,
    base_model_override: str | None = None,
    resume_from_checkpoint: str | None = None,
    train_file: str = "training/data/tool_sft_train.jsonl",
    eval_file: str = "training/data/tool_sft_eval.jsonl",
    epochs: int = 3,
    max_seq_length: int = 2048,
    eval_strategy: str = "steps",
) -> str:
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    import torch

    base_hf, preset = _resolve_base_hf(model_key, adapter_in)
    if base_model_override:
        base_hf = base_model_override
        preset = dict(preset, base_hf=base_hf, base_model_override=base_model_override)
    key = model_key or preset.get("model_key", "qwen3-4b")
    out = str(PROJECT_ROOT / (output_dir or f"training/output/sft-{key}"))
    train_path = PROJECT_ROOT / train_file
    eval_path = PROJECT_ROOT / eval_file

    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} missing — run: python training/build_agent_sft.py"
        )

    print(f"SFT [{key}] base={base_hf} → {out}  epochs={epochs}")

    load_kw: dict = dict(
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {torch.cuda.get_device_name(0)}  free={free_gb:.1f}GB / {total_gb:.1f}GB")
        if free_gb < 10:
            load_kw["device_map"] = "auto"
            load_kw["max_memory"] = {0: "12GiB", "cpu": "24GiB"}

    r = preset.get("lora_r", 32)
    alpha = preset.get("lora_alpha", 32)

    model, tokenizer = FastLanguageModel.from_pretrained(model_name=base_hf, **load_kw)
    if adapter_in and Path(adapter_in).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_in), is_trainable=True)
        print(f"  continued from adapter {adapter_in}")
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=r, lora_alpha=alpha, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

    dataset = load_dataset("json", data_files={
        "train": str(train_path),
        "eval": str(eval_path),
    })

    from src.agent.tools import tool_schemas
    tool_defs = tool_schemas(LITE_TOOL_NAMES)

    def format_chat(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"], tools=tool_defs,
            tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_chat)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"] if eval_path.exists() else None,
        args=SFTConfig(
            output_dir=out,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=preset.get("learning_rate", 1.0e-4),
            weight_decay=0.01,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            max_seq_length=max_seq_length,
            bf16=True,
            logging_steps=5,
            save_steps=25,
            eval_steps=50 if eval_strategy == "steps" and eval_path.exists() else None,
            eval_strategy=eval_strategy if eval_path.exists() else "no",
            save_total_limit=2,
            seed=42,
            dataset_text_field="text",
            packing=False,
            report_to="none",
        ),
    )
    ckpt = resume_from_checkpoint
    if ckpt and Path(ckpt).exists():
        print(f"  resuming from checkpoint {ckpt}")
    trainer.train(resume_from_checkpoint=ckpt if ckpt and Path(ckpt).exists() else None)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    meta = {
        "model_key": key,
        "base_hf": base_hf,
        "ollama_name": preset.get("ollama_name"),
        "label": preset.get("label"),
        "train_file": train_file,
        "epochs": epochs,
        "continued_from": adapter_in,
    }
    (Path(out) / "regllm_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved → {out}")
    return out


def _free_ollama_vram() -> None:
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
                print(f"  unloaded ollama: {name}")
    except Exception as e:
        print(f"  (ollama cleanup: {e})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="RegLLM mid-size SFT")
    p.add_argument("--model", default=None, help="Key from training/sft_models.yaml (olmo3-7b, qwen3-8b)")
    p.add_argument("--adapter-in", default=None, help="Continue SFT from existing LoRA adapter")
    p.add_argument("--resume-from-checkpoint", default=None, help="HF trainer checkpoint dir to resume")
    p.add_argument("--output", default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-file", default="training/data/agent_sft_train.jsonl")
    p.add_argument("--eval-file", default="training/data/agent_sft_eval.jsonl")
    p.add_argument("--free-gpu", action="store_true", help="Unload Ollama models before training")
    args = p.parse_args()
    if args.free_gpu:
        _free_ollama_vram()
    if not args.model and not args.adapter_in:
        p.error("Provide --model or --adapter-in")
    run_sft(
        args.model,
        args.output,
        adapter_in=args.adapter_in,
        resume_from_checkpoint=args.resume_from_checkpoint,
        train_file=args.train_file,
        eval_file=args.eval_file,
        epochs=args.epochs,
    )

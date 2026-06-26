#!/usr/bin/env python3
"""Export a RegLLM PEFT adapter to GGUF and register in Ollama.

Usage:
    .venv/bin/python training/export_sft_ollama.py training/output/sft-olmo3-7b
    .venv/bin/python training/export_sft_ollama.py training/output/sft-olmo3-7b --create
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _free_gpu() -> None:
    try:
        import httpx
        for m in httpx.get("http://localhost:11434/api/ps", timeout=5).json().get("models", []):
            name = m.get("name", "")
            if name:
                httpx.post("http://localhost:11434/api/generate",
                           json={"model": name, "keep_alive": 0}, timeout=10)
    except Exception:
        pass
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def export(adapter_path: str | Path, *, quant: str = "q4_K_M", create: bool = False) -> Path:
    from unsloth import FastLanguageModel
    from peft import PeftModel
    from training.tool_utils import _base_hf_for_adapter

    ap = Path(adapter_path)
    if not (ap / "adapter_config.json").exists():
        raise FileNotFoundError(f"No adapter at {ap}")

    meta_path = ap / "regllm_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    base = meta.get("base_hf") or _base_hf_for_adapter(ap)
    ollama_name = meta.get("ollama_name", f"regllm-{ap.name}")

    print(f"Export {ap.name}  base={base}  quant={quant}  ollama={ollama_name}")

    _free_gpu()

    merged_dir = ap / "merged_fp16"
    if not (merged_dir / "config.json").exists():
        print("Merging LoRA → fp16...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base, max_seq_length=4096, dtype=None, load_in_4bit=True,
        )
        model = PeftModel.from_pretrained(model, str(ap))
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        del model
        _free_gpu()
    else:
        print(f"Using cached merge: {merged_dir}")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))

    print("Loading merged model for GGUF export...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(merged_dir), max_seq_length=4096, dtype=None, load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    gguf_dir = ap / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing GGUF to {gguf_dir}...")
    model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=quant)
    del model
    _free_gpu()

    gguf_files = sorted(gguf_dir.glob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
    if not gguf_files:
        raise RuntimeError(f"No GGUF produced in {gguf_dir}")

    modelfile = gguf_dir / "Modelfile"
    modelfile.write_text(
        f"FROM ./{gguf_files[0].name}\n"
        f"PARAMETER temperature 0.1\n"
        f"PARAMETER num_ctx 4096\n"
        f'SYSTEM "Eres RegLLM: investigador de pipelines SAS LGD/ECL (IRB). '
        f'Respondes en español. Usa herramientas cuando proceda."\n',
        encoding="utf-8",
    )
    print(f"Modelfile → {modelfile}")

    if create:
        print(f"Creating Ollama model `{ollama_name}`...")
        subprocess.run(
            ["ollama", "create", ollama_name, "-f", "Modelfile"],
            cwd=gguf_dir, check=True,
        )
        print(f"Done. Try: ollama run {ollama_name}")

    return gguf_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("adapter", help="Path to PEFT adapter dir")
    p.add_argument("--quant", default="q4_K_M")
    p.add_argument("--create", action="store_true", help="Run ollama create after export")
    args = p.parse_args()
    export(args.adapter, quant=args.quant, create=args.create)


if __name__ == "__main__":
    main()

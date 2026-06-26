"""Shared helpers for tool-selection training and evaluation."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TOOLS = [
    "trace_causal_chain", "query_cycle_data", "get_sas_formula",
    "trace_dependencies", "inspect_lineage",
    "search_docs", "search_regulation", "get_field_definition",
]

LINEAGE_TOOLS = {"trace_causal_chain", "get_sas_formula", "trace_dependencies", "inspect_lineage"}

from src.agent.prompts import LITE_SYSTEM_PROMPT


def parse_tools_from_text(text: str) -> list[str]:
    called: list[str] = []
    for tool in TOOLS:
        if f'"name": "{tool}"' in text or f'"name":"{tool}"' in text:
            called.append(tool)
    if not called:
        for tool in TOOLS:
            if re.search(rf"\b{tool}\b", text) and ("tool_call" in text or "function" in text):
                called.append(tool)
    return called


def score_item(item: dict, called: list[str], *, lineage_flexible: bool = False) -> int:
    called_set = set(called)
    if "expected_tools" in item:
        expected = set(item["expected_tools"])
        if expected <= called_set:
            return 1
        if lineage_flexible:
            exp_lin = expected & LINEAGE_TOOLS
            got_lin = called_set & LINEAGE_TOOLS
            exp_rag = expected - LINEAGE_TOOLS
            if exp_lin and got_lin and exp_rag <= called_set:
                return 1
        return -1
    expected = item["expected_tool"]
    if expected in called_set:
        return 1
    if lineage_flexible and expected in LINEAGE_TOOLS and called_set & LINEAGE_TOOLS:
        return 0  # partial — not full credit
    return -1


def _base_hf_for_adapter(adapter_path: Path | str) -> str:
    """Resolve HuggingFace base model id from adapter metadata."""
    ap = Path(adapter_path)
    meta = ap / "regllm_meta.json"
    if meta.exists():
        base = json.loads(meta.read_text(encoding="utf-8")).get("base_hf")
        if base:
            return base
    cfg = ap / "adapter_config.json"
    if cfg.exists():
        return json.loads(cfg.read_text(encoding="utf-8")).get(
            "base_model_name_or_path", "Qwen/Qwen3-4B"
        )
    return "Qwen/Qwen3-4B"


def is_8b_adapter(adapter_path: Path | str) -> bool:
    base = _base_hf_for_adapter(adapter_path).lower()
    return "8b" in base or "8-b" in base


def grpo_memory_profile(adapter_path: Path | str) -> dict[str, str]:
    """GRPO CLI args tuned for adapter size and typical 24GB VRAM."""
    if is_8b_adapter(adapter_path):
        return {
            "batch_size": "2",
            "num_generations": "2",
            "max_completion_length": "256",
            "max_prompt_length": "1024",
        }
    return {
        "batch_size": "12",
        "num_generations": "8",
        "max_completion_length": "512",
        "max_prompt_length": "2048",
    }


def release_gpu() -> None:
    """Best-effort free CUDA memory in the current process."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def load_adapter(adapter_path: Path | str, max_seq_length: int = 2048):
    from unsloth import FastLanguageModel
    from peft import PeftModel
    ap = str(adapter_path)
    base = _base_hf_for_adapter(adapter_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, ap)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def predict_tools(model, tokenizer, question: str, max_new_tokens: int = 256) -> list[str]:
    from src.agent.tools import tool_schemas

    messages = [
        {"role": "system", "content": LITE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    schemas = tool_schemas(TOOLS)
    text = tokenizer.apply_chat_template(
        messages, tools=schemas, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False,
        use_cache=True,
    )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    return parse_tools_from_text(decoded)


def eval_adapter(
    adapter_path: Path | str,
    manifest_path: Path | str | None = None,
    *,
    verbose: bool = False,
) -> dict:
    manifest_path = Path(manifest_path or PROJECT_ROOT / "training/data/tool_selection_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    model, tokenizer = load_adapter(adapter_path)

    total = 0
    by_kind: dict[str, list[int]] = {}
    failures: list[dict] = []

    for i, item in enumerate(manifest):
        called = predict_tools(model, tokenizer, item["question"])
        reward = score_item(item, called)
        total += reward
        by_kind.setdefault(item.get("kind", "?"), []).append(reward)
        if reward < 0:
            failures.append({**item, "tools_called": called})
        if verbose:
            icon = "+" if reward > 0 else "-"
            exp = item.get("expected_tool") or item.get("expected_tools")
            print(f"[{i+1}] [{icon}] got={called} exp={exp}")

    n = len(manifest)
    acc = (total + n) / (2 * n) * 100
    result = {
        "accuracy_pct": round(acc, 1),
        "total_reward": total,
        "n": n,
        "by_kind": {
            k: round(sum(1 for r in v if r > 0) / len(v) * 100, 1)
            for k, v in by_kind.items()
        },
        "failures": failures,
    }
    del model
    import torch
    torch.cuda.empty_cache()
    return result

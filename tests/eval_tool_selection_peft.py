"""Evaluate tool-selection on the fine-tuned PEFT adapter (no Ollama export needed).

Usage:
    .venv/bin/python tests/eval_tool_selection_peft.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST = PROJECT_ROOT / "training" / "data" / "tool_selection_manifest.json"
ADAPTER = PROJECT_ROOT / "training" / "output" / "phase2-sft"
BASE = "Qwen/Qwen3-4B"
TOOLS = ["trace_causal_chain", "get_sas_formula", "trace_dependencies", "inspect_lineage", "search_docs",
         "search_regulation", "get_field_definition"]

_LITE_SYSTEM_PROMPT = """\
You are a banking regulation assistant for COREP/FINREP IRB reporting.
You MUST call tools before answering. Never answer from memory alone.

## When to use each tool

- **trace_dependencies** or **inspect_lineage**: ALWAYS call one of these when the question asks about field origins, lineage, dependencies, calculations, or "where does X come from". Pass the target field name.
- **search_docs**: Call this to find field definitions, table schemas, or changelog entries. Good for "what is X?" questions.
- **search_regulation**: Call this when the question mentions regulation, CRR, EBA, circular, normativa, suelo regulatorio, or compliance rules.
- **get_field_definition**: Call this for quick field lookup by exact name.

## Rules

1. Call at least one tool per question. If unsure which, call search_docs first.
2. For lineage questions, ALWAYS call trace_dependencies or inspect_lineage.
3. Base your answer ONLY on tool results. Do not invent data.
4. If a field is not found, say so explicitly.
5. Answer in the same language as the question.
"""

from src.agent.tools import tool_schemas


def load_model():
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def first_tool(model, tokenizer, question: str) -> list[str]:
    from unsloth import FastLanguageModel  # noqa: F401
    messages = [
        {"role": "system", "content": _LITE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tools = tool_schemas(TOOLS)
    text = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=256, temperature=0.1, use_cache=True)
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

    called: list[str] = []
    for tool in TOOLS:
        if f'"name": "{tool}"' in decoded or f'"name":"{tool}"' in decoded:
            called.append(tool)
    if not called:
        # Qwen tool call format fallback
        for line in decoded.split("\n"):
            for tool in TOOLS:
                if tool in line and "function" in decoded:
                    called.append(tool)
                    break
    return called[:3]


def score_item(item: dict, called: list[str]) -> int:
    called_set = set(called)
    if "expected_tools" in item:
        return 1 if set(item["expected_tools"]) <= called_set else -1
    return 1 if item["expected_tool"] in called_set else -1


def main() -> None:
    if not ADAPTER.exists():
        print(f"Missing adapter: {ADAPTER}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(MANIFEST.read_text())
    print(f"PEFT tool-selection eval | adapter={ADAPTER.name} | n={len(manifest)}")

    model, tokenizer = load_model()
    total = 0
    by_kind: dict[str, list[int]] = {}

    for i, item in enumerate(manifest):
        t0 = time.time()
        called = first_tool(model, tokenizer, item["question"])
        reward = score_item(item, called)
        total += reward
        by_kind.setdefault(item.get("kind", "?"), []).append(reward)
        icon = "+" if reward > 0 else "-"
        print(f"[{i+1}/{len(manifest)}] [{icon}] got={called} expected={item.get('expected_tool') or item.get('expected_tools')} ({time.time()-t0:.1f}s)")

    n = len(manifest)
    acc = (total + n) / (2 * n) * 100
    print(f"\nAccuracy: {acc:.1f}%  reward={total}/{n}")
    for k, v in sorted(by_kind.items()):
        print(f"  {k}: {sum(1 for r in v if r>0)}/{len(v)}")

    out = PROJECT_ROOT / "data" / "tool_selection_sft.json"
    out.write_text(json.dumps({"accuracy_pct": acc, "total_reward": total, "n": n}, indent=2))


if __name__ == "__main__":
    main()

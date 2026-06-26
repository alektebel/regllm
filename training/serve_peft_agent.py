#!/usr/bin/env python3
"""OpenAI-compatible tool-calling server for a RegLLM PEFT adapter (no Ollama export).

Usage:
    .venv/bin/python training/serve_peft_agent.py --adapter training/output/sft-olmo3-7b

Then in the UI pick model `peft:olmo3-7b` (API must reach this host — run API locally or
set REGLLM_PEFT_URL=http://host.docker.internal:8767).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_PAYLOAD = re.compile(
    r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{(?:[^{}]|\[[^\]]*\])*\})\s*\}',
    re.DOTALL,
)

_model = None
_tokenizer = None
_adapter_path: Path | None = None


def _free_gpu() -> None:
    try:
        import httpx
        for m in httpx.get("http://localhost:11434/api/ps", timeout=5).json().get("models", []):
            if m.get("name"):
                httpx.post("http://localhost:11434/api/generate",
                           json={"model": m["name"], "keep_alive": 0}, timeout=10)
    except Exception:
        pass


def load(adapter: Path) -> None:
    global _model, _tokenizer, _adapter_path
    if _model is not None:
        return
    _free_gpu()
    from training.tool_utils import load_adapter
    _model, _tokenizer = load_adapter(adapter, max_seq_length=4096)
    _adapter_path = adapter


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    idx = 0

    def add(name: str, args: dict) -> None:
        nonlocal idx
        key = f"{name}:{json.dumps(args, sort_keys=True)}"
        if not name or key in seen:
            return
        seen.add(key)
        calls.append({
            "id": f"peft_call_{idx}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
        idx += 1

    for m in _TOOL_CALL_TAG.finditer(text or ""):
        try:
            p = json.loads(m.group(1))
            add(p.get("name", ""), p.get("arguments") or {})
        except json.JSONDecodeError:
            pass
    for m in _TOOL_PAYLOAD.finditer(text or ""):
        try:
            add(m.group(1), json.loads(m.group(2)))
        except json.JSONDecodeError:
            pass
    return calls


def _strip_tool_text(text: str) -> str:
    text = _TOOL_CALL_TAG.sub("", text or "")
    text = _TOOL_PAYLOAD.sub("", text)
    return text.strip()


def run_chat(body: dict[str, Any]) -> dict[str, Any]:
    assert _model is not None and _tokenizer is not None
    messages = body.get("messages") or []
    tools = body.get("tools") or []
    temperature = float(body.get("temperature", 0.1))
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024)

    text = _tokenizer.apply_chat_template(
        messages, tools=tools or None,
        tokenize=False, add_generation_prompt=True,
    )
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
    out = _model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=max(temperature, 0.01),
        do_sample=temperature > 0,
        use_cache=True,
    )
    decoded = _tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    tool_calls = _parse_tool_calls(decoded)
    content = _strip_tool_text(decoded)

    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls

    return {
        "id": "peft-chat",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "model": body.get("model", "peft"),
    }


def main() -> None:
    from fastapi import FastAPI
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="training/output/sft-olmo3-7b")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()

    adapter = PROJECT_ROOT / args.adapter
    app = FastAPI(title="RegLLM PEFT Agent Server")

    @app.get("/health")
    def health():
        return {"status": "ok", "adapter": str(adapter)}

    @app.get("/v1/models")
    @app.get("/models")
    def models():
        return {"data": [{"id": "peft", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def v1_chat(body: dict):
        return run_chat(body)

    @app.post("/chat/completions")
    async def legacy_chat(body: dict):
        return run_chat(body)

    print(f"Loading {adapter}...", flush=True)
    load(adapter)
    print(f"PEFT agent server → http://0.0.0.0:{args.port}", flush=True)
    print("  Set REGLLM_LLM=litert REGLLM_LITERT_URL=http://127.0.0.1:8767", flush=True)
    print("  UI model: peft:olmo3-7b", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

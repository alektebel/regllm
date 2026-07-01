"""Local-LLM client for the GraphRAG justifier.

Auto-detects an available OpenAI-compatible local backend:

1. **LiteRT-LM** OpenAI-compatible server on ``http://localhost:9379``
   (Google's official local serving stack — used when a Gemma 4 ``.litertlm``
   weight bundle is in place).
2. **Ollama** on ``http://localhost:11434`` — works with any chat-tuned model
   (Qwen, Gemma, Llama, Phi, …). The default model
   ``qwen2.5:14b-instruct-q4_K_M`` is a great drop-in for structured JSON RAG
   verdicts and is small enough to run on a single 24 GB GPU.
3. **Amazon Bedrock** — managed LLM backend via the Converse API. Set
   ``REGLLM_LLM=bedrock`` and let the IAM task role handle auth.
4. **Stub** mode: when ``REGLLM_LLM=stub`` or no backend reachable, returns a
   deterministic JSON-shaped placeholder so unit tests and the frontend keep
   working without any model installed.

Configuration via environment variables (with sensible defaults):

- ``REGLLM_LLM``             ``auto`` | ``litert`` | ``ollama`` | ``bedrock`` | ``stub``
- ``OLLAMA_URL``             default ``http://localhost:11434``
- ``OLLAMA_MODEL``           default ``qwen2.5:14b-instruct-q4_K_M``
- ``LITERT_URL``             default ``http://localhost:9379/v1``
- ``LITERT_MODEL``           default ``gemma4-12b,gpu``
- ``BEDROCK_MODEL_ID``       default ``anthropic.claude-3-haiku-20240307-v1:0``
- ``BEDROCK_REGION``         default ``eu-west-1``
- ``REGLLM_LLM_TIMEOUT``     request timeout in seconds, default ``120``

Backwards-compatible aliases (still read for migration):
``GEMMA_LITERT_URL``, ``GEMMA_LITERT_MODEL``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

try:
    import yaml as _yaml
except ImportError:  # pyyaml not installed — fall back to env-only config
    _yaml = None

try:
    import boto3 as _boto3
except ImportError:  # boto3 not installed — bedrock backend unavailable
    _boto3 = None


def _load_yaml_config() -> dict[str, Any]:
    """Load config.yaml from the project root (best-effort)."""
    for candidate in (
        Path(__file__).resolve().parents[2] / "config.yaml",
        Path("config.yaml"),
    ):
        if candidate.is_file():
            if _yaml is None:
                return {}
            with open(candidate) as f:
                return _yaml.safe_load(f) or {}
    return {}


_CFG = _load_yaml_config()
_LLM_CFG: dict[str, Any] = _CFG.get("llm", {})

logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT = float(os.getenv("REGLLM_LLM_TIMEOUT", "120"))


@dataclass
class ChatResponse:
    text: str
    backend: str                       # "litert" | "ollama" | "bedrock" | "stub"
    model: str
    raw: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None  # populated by ``chat_tools``


class LocalLLMClient:
    """Talks to whichever local OpenAI-compatible LLM backend is reachable."""

    def __init__(
        self,
        litert_url: str | None = None,
        litert_model: str | None = None,
        ollama_url: str | None = None,
        ollama_model: str | None = None,
        prefer: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.litert_url = (
            litert_url
            or os.getenv("LITERT_URL")
            or os.getenv("GEMMA_LITERT_URL")
            or _LLM_CFG.get("litert_url")
            or "http://localhost:9379/v1"
        )
        self.litert_model = (
            litert_model
            or os.getenv("LITERT_MODEL")
            or os.getenv("GEMMA_LITERT_MODEL")
            or _LLM_CFG.get("litert_model")
            or "gemma4-12b,gpu"
        )
        self.ollama_url = (
            ollama_url
            or os.getenv("OLLAMA_URL")
            or _LLM_CFG.get("ollama_url")
            or "http://localhost:11434"
        )
        self.ollama_model = (
            ollama_model
            or os.getenv("OLLAMA_MODEL")
            or _LLM_CFG.get("model")
            or "qwen3:32b"
        )
        # OLLAMA_MODEL=none → force stub mode (no model download needed)
        if self.ollama_model == "none":
            prefer = "stub"
        self.bedrock_model_id = (
            os.getenv("BEDROCK_MODEL_ID")
            or _LLM_CFG.get("bedrock_model_id")
            or "anthropic.claude-3-haiku-20240307-v1:0"
        )
        self.bedrock_region = (
            os.getenv("BEDROCK_REGION")
            or _LLM_CFG.get("bedrock_region")
            or "eu-west-1"
        )
        self._bedrock_client = None
        self.prefer = prefer or os.getenv("REGLLM_LLM") or _LLM_CFG.get("backend") or "auto"
        self.timeout = timeout if timeout is not None else _DEFAULT_TIMEOUT
        self._backend: str | None = None
        self._probed = False

    # ── Probing ───────────────────────────────────────────────────────────

    def detect_backend(self) -> str:
        if self._probed:
            return self._backend or "stub"
        self._probed = True
        if self.prefer == "stub":
            self._backend = "stub"
            return "stub"
        if self.prefer == "bedrock":
            if _boto3 is None:
                logger.error("REGLLM_LLM=bedrock but boto3 is not installed")
                self._backend = "stub"
                return "stub"
            self._backend = "bedrock"
            return "bedrock"
        if self.prefer in ("auto", "litert") and self._probe_litert():
            self._backend = "litert"
            return "litert"
        if self.prefer in ("auto", "ollama") and self._probe_ollama():
            # Verify the configured model is actually pulled before committing
            if self._ollama_has_model(self.ollama_model):
                self._backend = "ollama"
                return "ollama"
            logger.warning(
                "Ollama is reachable but model %r is not pulled. "
                "Run `ollama pull %s` or set OLLAMA_MODEL to one of the "
                "installed models. Falling back to stub.",
                self.ollama_model, self.ollama_model,
            )
        if self.prefer == "ollama":
            # User explicitly asked for ollama but it's unreachable
            logger.warning("REGLLM_LLM=ollama but Ollama unreachable; falling back to stub")
        elif self.prefer == "auto":
            logger.info("No local LLM backend reachable; falling back to stub mode")
        self._backend = "stub"
        return "stub"

    def _probe_litert(self) -> bool:
        try:
            r = httpx.get(f"{self.litert_url}/models", timeout=2.0)
            return r.status_code < 500
        except Exception:
            return False

    def _probe_ollama(self) -> bool:
        try:
            r = httpx.get(f"{self.ollama_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def _ollama_has_model(self, name: str) -> bool:
        try:
            r = httpx.get(f"{self.ollama_url}/api/tags", timeout=2.0)
            if r.status_code != 200:
                return False
            tags = {m.get("name") for m in r.json().get("models", [])}
            # Match exact tag, or any tag that starts with the bare name (without ":tag")
            if name in tags:
                return True
            bare = name.split(":", 1)[0]
            return any(t.split(":", 1)[0] == bare for t in tags)
        except Exception:
            return False

    # ── Bedrock ───────────────────────────────────────────────────────────

    def _get_bedrock_client(self):
        if self._bedrock_client is None:
            self._bedrock_client = _boto3.client(
                "bedrock-runtime",
                region_name=self.bedrock_region,
            )
        return self._bedrock_client

    def _chat_bedrock(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> ChatResponse:
        client = self._get_bedrock_client()

        system_parts: list[dict[str, str]] = []
        converse_msgs: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append({"text": m["content"]})
            else:
                converse_msgs.append({
                    "role": m["role"],
                    "content": [{"text": m["content"]}],
                })

        if json_mode and system_parts:
            system_parts[0]["text"] += "\n\nRespond ONLY with valid JSON."

        kwargs: dict[str, Any] = {
            "modelId": self.bedrock_model_id,
            "messages": converse_msgs,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }
        if system_parts:
            kwargs["system"] = system_parts

        response = client.converse(**kwargs)
        text = response["output"]["message"]["content"][0]["text"]
        return ChatResponse(
            text=text,
            backend="bedrock",
            model=self.bedrock_model_id,
            raw=response,
        )

    # ── Chat ──────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> ChatResponse:
        """Send a list of OpenAI-format messages and return the response text."""
        backend = self.detect_backend()
        if backend == "litert":
            return self._chat_litert(messages, temperature, max_tokens, json_mode)
        if backend == "ollama":
            return self._chat_ollama(messages, temperature, max_tokens, json_mode)
        if backend == "bedrock":
            return self._chat_bedrock(messages, temperature, max_tokens, json_mode)
        return self._chat_stub(messages, json_mode)

    def _chat_litert(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.litert_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = httpx.post(
            f"{self.litert_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return ChatResponse(text=text, backend="litert", model=self.litert_model, raw=data)

    def _chat_ollama(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> ChatResponse:
        # For thinking-capable models (qwen3*) in non-JSON mode, disable
        # thinking to prevent <think> tokens consuming the output budget.
        # In JSON mode, skip /no_think — it causes empty output with format:json.
        # Instead we let the model think and strip tags from the result.
        msgs = list(messages)
        if self._is_thinking_model() and not json_mode:
            msgs = _inject_no_think(msgs)
        msgs = _inject_spanish(msgs)

        # Thinking models need extra token budget for <think> tokens in JSON mode;
        # the think content is stripped from the final output.
        predict = max_tokens
        if json_mode and self._is_thinking_model():
            predict = max_tokens + 4096

        payload: dict[str, Any] = {
            "model": self.ollama_model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": predict},
        }
        if json_mode:
            payload["format"] = "json"
        r = httpx.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        text = _strip_think_tags(data.get("message", {}).get("content", ""))
        return ChatResponse(text=text, backend="ollama", model=self.ollama_model, raw=data)

    def _is_thinking_model(self) -> bool:
        """Return True if the current model has a thinking mode (qwen3, deepseek-r1, etc.)."""
        name = self.ollama_model.lower()
        return any(prefix in name for prefix in ("qwen3", "deepseek-r1", "qwq"))

    def _chat_stub(self, messages: list[dict[str, str]], json_mode: bool) -> ChatResponse:
        last = messages[-1]["content"] if messages else ""
        if json_mode:
            stub = {
                "justified": False,
                "evidence": [],
                "confidence": 0.0,
                "rationale": (
                    "Stub mode: no local LLM backend reachable. "
                    "Start Ollama and pull a model "
                    "(`ollama pull qwen3:32b`), or set "
                    "REGLLM_LLM=litert with LiteRT-LM serving Gemma."
                ),
            }
            return ChatResponse(text=json.dumps(stub), backend="stub", model="stub")
        return ChatResponse(
            text=f"[stub LLM]: received {len(last)} chars of prompt",
            backend="stub", model="stub",
        )

    # ── JSON helper ───────────────────────────────────────────────────────

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Strict-JSON helper: instructs the model to reply with valid JSON only."""
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        resp = self.chat(msgs, temperature=temperature, max_tokens=max_tokens, json_mode=True)
        return _safe_json(resp.text)

    # ── Tool-calling chat (single round) ─────────────────────────────────

    def chat_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """One LLM round with tool-calling support.

        Returns a :class:`ChatResponse` whose ``tool_calls`` is a list of
        ``{"id": str, "name": str, "arguments": dict}`` if the model wants
        to invoke tools, otherwise empty/None.
        """
        backend = self.detect_backend()
        if backend == "litert":
            return self._chat_tools_openai(messages, tools, temperature, max_tokens)
        if backend == "ollama":
            return self._chat_tools_ollama(messages, tools, temperature, max_tokens)
        if backend == "bedrock":
            raise NotImplementedError("Bedrock tool-calling not yet implemented")
        return self._chat_tools_stub(messages, tools)

    def _chat_tools_ollama(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> ChatResponse:
        msgs = _inject_spanish(list(messages))
        if self._is_thinking_model():
            msgs = _inject_no_think(msgs)
        payload: dict[str, Any] = {
            "model": self.ollama_model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "tools": tools,
        }
        r = httpx.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        if r.status_code >= 400:
            # Surface the server-side detail; Ollama returns helpful error JSON.
            detail = r.text[:500]
            raise httpx.HTTPStatusError(
                f"Ollama {r.status_code}: {detail}",
                request=r.request, response=r,
            )
        data = r.json()
        msg = data.get("message", {}) or {}
        text = _strip_think_tags(msg.get("content", "") or "")
        raw_calls = msg.get("tool_calls") or []
        calls = []
        for c in raw_calls:
            fn = (c.get("function") or {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append({
                "id": c.get("id", f"call_{len(calls)}"),
                "name": fn.get("name", ""),
                "arguments": args or {},
            })
        return ChatResponse(
            text=text, backend="ollama", model=self.ollama_model,
            raw=data, tool_calls=calls or None,
        )

    def _chat_tools_openai(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> ChatResponse:
        # LiteRT-LM speaks the OpenAI shape for tools.
        payload = {
            "model": self.litert_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = httpx.post(
            f"{self.litert_url}/chat/completions",
            json=payload, timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        text = msg.get("content") or ""
        raw_calls = msg.get("tool_calls") or []
        calls = []
        for c in raw_calls:
            fn = (c.get("function") or {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append({
                "id": c.get("id", f"call_{len(calls)}"),
                "name": fn.get("name", ""),
                "arguments": args or {},
            })
        return ChatResponse(
            text=text, backend="litert", model=self.litert_model,
            raw=data, tool_calls=calls or None,
        )

    def _chat_tools_stub(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ChatResponse:
        """Deterministic stub: emit one final-answer with the prompt summary.

        We don't fabricate a tool call here because the agent loop would
        then dispatch real tools against fake arguments. Returning a
        plain text answer terminates the loop cleanly in offline tests.
        """
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        return ChatResponse(
            text=(
                "[stub LLM] No local LLM backend reachable, so I cannot run "
                "the agentic Q&A. Pull a model with "
                "`ollama pull qwen3:32b` or set "
                f"REGLLM_LLM=litert. Question was: {last_user[:200]}"
            ),
            backend="stub", model="stub", tool_calls=None,
        )


# ── Module-level singleton ──────────────────────────────────────────────────


_default_client: LocalLLMClient | None = None


def get_client() -> LocalLLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LocalLLMClient()
    return _default_client


def reset_client() -> None:
    """Reset the singleton (useful in tests when env vars change)."""
    global _default_client
    _default_client = None


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output (qwen3, deepseek-r1)."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _inject_spanish(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure the model stays in Spanish for the whole conversation."""
    lock = "Responde SIEMPRE en español. No cambies de idioma a mitad de conversación."
    out = list(messages)
    for i, m in enumerate(out):
        if m.get("role") == "system":
            content = m.get("content", "")
            if "SIEMPRE en español" not in content:
                out[i] = {**m, "content": f"{lock}\n\n{content}"}
            return out
    out.insert(0, {"role": "system", "content": lock})
    return out


def _inject_no_think(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Prepend /no_think to the last user message to disable thinking mode."""
    if not messages:
        return messages
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i] = {**out[i], "content": "/no_think\n" + out[i]["content"]}
            break
    return out


def _safe_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from a model response."""
    # Strip any residual thinking tags (belt-and-suspenders with _strip_think_tags)
    text = _strip_think_tags(text)
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}") + 1
        if s >= 0 and e > s:
            try:
                return json.loads(text[s:e])
            except json.JSONDecodeError:
                pass
    return {"error": "invalid_json", "raw": text[:500]}


# ── Backwards-compatible aliases ────────────────────────────────────────────

GemmaClient = LocalLLMClient   # legacy name

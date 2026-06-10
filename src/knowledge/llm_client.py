"""Local-LLM client for the GraphRAG justifier.

Auto-detects an available OpenAI-compatible local backend:

1. **LiteRT-LM** OpenAI-compatible server on ``http://localhost:9379``
   (Google's official local serving stack — used when a Gemma 4 ``.litertlm``
   weight bundle is in place).
2. **Ollama** on ``http://localhost:11434`` — works with any chat-tuned model
   (Qwen, Gemma, Llama, Phi, …). The default model
   ``qwen2.5:14b-instruct-q4_K_M`` is a great drop-in for structured JSON RAG
   verdicts and is small enough to run on a single 24 GB GPU.
3. **Stub** mode: when ``REGLLM_LLM=stub`` or no backend reachable, returns a
   deterministic JSON-shaped placeholder so unit tests and the frontend keep
   working without any model installed.

Configuration via environment variables (with sensible defaults):

- ``REGLLM_LLM``             ``auto`` | ``litert`` | ``ollama`` | ``stub``
- ``OLLAMA_URL``             default ``http://localhost:11434``
- ``OLLAMA_MODEL``           default ``qwen2.5:14b-instruct-q4_K_M``
- ``LITERT_URL``             default ``http://localhost:9379/v1``
- ``LITERT_MODEL``           default ``gemma4-12b,gpu``
- ``REGLLM_LLM_TIMEOUT``     request timeout in seconds, default ``120``

Backwards-compatible aliases (still read for migration):
``GEMMA_LITERT_URL``, ``GEMMA_LITERT_MODEL``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT = float(os.getenv("REGLLM_LLM_TIMEOUT", "120"))


@dataclass
class ChatResponse:
    text: str
    backend: str                       # "litert" | "ollama" | "stub"
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
            or "http://localhost:9379/v1"
        )
        self.litert_model = (
            litert_model
            or os.getenv("LITERT_MODEL")
            or os.getenv("GEMMA_LITERT_MODEL")
            or "gemma4-12b,gpu"
        )
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.ollama_model = (
            ollama_model
            or os.getenv("OLLAMA_MODEL")
            or "qwen2.5:14b-instruct-q4_K_M"
        )
        # OLLAMA_MODEL=none → force stub mode (no model download needed)
        if self.ollama_model == "none":
            prefer = "stub"
        self.prefer = prefer or os.getenv("REGLLM_LLM", "auto")
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
        payload: dict[str, Any] = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
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
        text = data.get("message", {}).get("content", "")
        return ChatResponse(text=text, backend="ollama", model=self.ollama_model, raw=data)

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
                    "(`ollama pull qwen2.5:14b-instruct-q4_K_M`), or set "
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
        return self._chat_tools_stub(messages, tools)

    def _chat_tools_ollama(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.ollama_model,
            "messages": messages,
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
        text = msg.get("content", "") or ""
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
                "`scripts/setup_llm.sh` (default Qwen 2.5 14B) or set "
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


def _safe_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from a model response."""
    text = text.strip()
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

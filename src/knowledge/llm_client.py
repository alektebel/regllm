"""Local-LLM client for the GraphRAG justifier.

Auto-detects an available OpenAI-compatible local backend:

1. **LiteRT-LM** OpenAI-compatible server on ``http://localhost:9379``
   (Google's official local serving stack — used when a Gemma 4 ``.litertlm``
   weight bundle is in place).
2. **Ollama** on ``http://localhost:11434`` — works with any chat-tuned model
   (Qwen, Gemma, Llama, Phi, …). The default model
   ``qwen2.5:14b-instruct-q4_K_M`` is a great drop-in for structured JSON RAG
   verdicts and is small enough to run on a single 24 GB GPU.
3. **GGUF (standalone, via `llama-cpp-python`)** — loads a ``.gguf`` weight
   file directly **in-process**, no Ollama server required. Set
   ``GGUF_MODEL_PATH=/path/to/model.gguf`` (and optionally
   ``REGLLM_LLM=gguf`` to force it). This is distinct from pointing
   ``OLLAMA_MODEL`` at a ``.gguf`` path (still supported — that path
   auto-registers the file *into a running Ollama server* via
   ``ollama create``). The standalone backend needs nothing but the weight
   file and the optional ``llama-cpp-python`` dependency, which makes it the
   right choice for a fully self-contained, single-process deployment (e.g.
   an air-gapped bank environment with no sidecar model server).
4. **Amazon Bedrock** — managed LLM backend via the Converse API. Set
   ``REGLLM_LLM=bedrock`` and let the IAM task role handle auth.
5. **Stub** mode: when ``REGLLM_LLM=stub`` or no backend reachable, returns a
   deterministic JSON-shaped placeholder so unit tests and the frontend keep
   working without any model installed.

Configuration via environment variables (with sensible defaults):

- ``REGLLM_LLM``             ``auto`` | ``litert`` | ``ollama`` | ``gguf`` | ``bedrock`` | ``stub``
- ``OLLAMA_URL``             default ``http://localhost:11434``
- ``OLLAMA_MODEL``           Ollama tag or path to ``.gguf`` file
- ``LITERT_URL``             default ``http://localhost:9379/v1``
- ``LITERT_MODEL``           default ``gemma4-12b,gpu``
- ``GGUF_MODEL_PATH``        path to a ``.gguf`` chat model — enables the standalone backend
- ``GGUF_N_CTX``             context window, default ``8192``
- ``GGUF_N_GPU_LAYERS``      layers to offload to GPU, default ``0`` (CPU-only)
- ``GGUF_N_THREADS``         CPU threads, default: let ``llama.cpp`` choose
- ``GGUF_CHAT_FORMAT``       override the chat template ``llama.cpp`` infers
  from the GGUF's metadata (rarely needed — set only if a model's built-in
  template isn't recognised, e.g. ``"chatml"``, ``"qwen"``, ``"gemma"``)
- ``BEDROCK_MODEL_ID``       default ``eu.amazon.nova-micro-v1:0``
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

try:
    from llama_cpp import Llama as _Llama
except ImportError:  # llama-cpp-python not installed — gguf backend unavailable
    _Llama = None


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
    backend: str                       # "litert" | "ollama" | "gguf" | "bedrock" | "stub"
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
        # If model is a .gguf path, register it in Ollama automatically
        elif self.ollama_model.endswith(".gguf"):
            self.ollama_model = self._register_gguf(self.ollama_model, self.ollama_url)
        self.bedrock_model_id = (
            os.getenv("BEDROCK_MODEL_ID")
            or _LLM_CFG.get("bedrock_model_id")
            or "eu.amazon.nova-micro-v1:0"
        )
        self.bedrock_region = (
            os.getenv("BEDROCK_REGION")
            or _LLM_CFG.get("bedrock_region")
            or "eu-west-1"
        )
        self._bedrock_client = None
        # Azure OpenAI (managed, for Azure deployments — see DQC/azure/)
        self.azure_endpoint = (
            os.getenv("AZURE_OPENAI_ENDPOINT")
            or _LLM_CFG.get("azure_endpoint") or ""
        )
        self.azure_api_key = (
            os.getenv("AZURE_OPENAI_API_KEY")
            or _LLM_CFG.get("azure_api_key") or ""
        )
        self.azure_deployment = (
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or _LLM_CFG.get("azure_deployment") or "gpt-4o-mini"
        )
        self.azure_api_version = (
            os.getenv("AZURE_OPENAI_API_VERSION")
            or _LLM_CFG.get("azure_api_version") or "2024-06-01"
        )
        # Standalone GGUF (llama-cpp-python) — see module docstring. Distinct
        # from OLLAMA_MODEL=*.gguf, which registers the file into a running
        # Ollama server instead of loading it in-process.
        self.gguf_model_path = (
            os.getenv("GGUF_MODEL_PATH") or _LLM_CFG.get("gguf_model_path") or ""
        )
        self.gguf_n_ctx = int(
            os.getenv("GGUF_N_CTX") or _LLM_CFG.get("gguf_n_ctx") or 8192
        )
        self.gguf_n_gpu_layers = int(
            os.getenv("GGUF_N_GPU_LAYERS") or _LLM_CFG.get("gguf_n_gpu_layers") or 0
        )
        _threads = os.getenv("GGUF_N_THREADS") or _LLM_CFG.get("gguf_n_threads")
        self.gguf_n_threads = int(_threads) if _threads else None
        self.gguf_chat_format = (
            os.getenv("GGUF_CHAT_FORMAT") or _LLM_CFG.get("gguf_chat_format") or None
        )
        self._gguf_llama: Any = None
        self.prefer = prefer or os.getenv("REGLLM_LLM") or _LLM_CFG.get("backend") or "auto"
        self.timeout = timeout if timeout is not None else _DEFAULT_TIMEOUT
        self._backend: str | None = None
        self._probed = False

    # ── GGUF auto-registration ─────────────────────────────────────────────

    @staticmethod
    def _register_gguf(gguf_path: str, ollama_url: str) -> str:
        """Register a local .gguf file in Ollama and return the model tag.

        Creates an Ollama model from the GGUF via ``ollama create`` so it can
        be used like any other Ollama model.  If already registered, skips.
        """
        import subprocess
        from pathlib import Path

        path = Path(gguf_path).expanduser().resolve()
        if not path.is_file():
            logger.error("GGUF file not found: %s", path)
            return gguf_path  # fall through — probe will fail gracefully

        tag = f"local-{path.stem}"
        # Check if already registered
        try:
            r = httpx.get(f"{ollama_url}/api/tags", timeout=3.0)
            if r.status_code == 200:
                names = {m.get("name", "") for m in r.json().get("models", [])}
                if tag in names or any(n.split(":")[0] == tag for n in names):
                    logger.info("GGUF model already registered as %r", tag)
                    return tag
        except Exception:
            pass

        # Create via temporary Modelfile
        import tempfile
        logger.info("Registering GGUF %s as Ollama model %r …", path.name, tag)
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as f:
                f.write(f"FROM {path}\n")
                f.flush()
                subprocess.run(
                    ["ollama", "create", tag, "-f", f.name],
                    capture_output=True, text=True,
                    timeout=600, check=True,
                )
            Path(f.name).unlink(missing_ok=True)
            logger.info("Registered GGUF as %r", tag)
        except Exception as exc:
            logger.error("Failed to register GGUF in Ollama: %s", exc)
        return tag

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
        if self.prefer == "azure":
            if not (self.azure_endpoint and self.azure_api_key):
                logger.error("REGLLM_LLM=azure but AZURE_OPENAI_ENDPOINT / "
                             "AZURE_OPENAI_API_KEY are not set")
                self._backend = "stub"
                return "stub"
            self._backend = "azure"
            return "azure"
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
        if self.prefer in ("auto", "gguf") and self._probe_gguf():
            self._backend = "gguf"
            return "gguf"
        if self.prefer == "gguf":
            # User explicitly asked for gguf but the weight file or the
            # llama-cpp-python dependency is missing.
            if _Llama is None:
                logger.warning(
                    "REGLLM_LLM=gguf but llama-cpp-python is not installed "
                    "(pip install llama-cpp-python); falling back to stub"
                )
            else:
                logger.warning(
                    "REGLLM_LLM=gguf but GGUF_MODEL_PATH=%r does not point to "
                    "a file; falling back to stub", self.gguf_model_path,
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

    def _probe_gguf(self) -> bool:
        """Cheap availability check — does NOT load the model.

        Loading a multi-GB GGUF file is expensive (seconds, significant
        memory), so ``detect_backend()`` only verifies the dependency is
        importable and the weight file exists; the actual ``Llama`` instance
        is constructed lazily on first real chat call, in
        :meth:`_get_gguf_llama`.
        """
        if _Llama is None or not self.gguf_model_path:
            return False
        return Path(self.gguf_model_path).expanduser().is_file()

    # ── Bedrock ───────────────────────────────────────────────────────────

    def _get_bedrock_client(self):
        if self._bedrock_client is None:
            self._bedrock_client = _boto3.client(
                "bedrock-runtime",
                region_name=self.bedrock_region,
            )
        return self._bedrock_client

    @staticmethod
    def _bedrock_convert_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        """Convert OpenAI/Ollama-shaped chat history into Bedrock Converse
        ``system`` + ``messages`` blocks, including the ``tool_calls``
        (assistant) and ``tool`` (tool-result) turns the agent loop appends
        to conversation history."""
        system_parts: list[dict[str, str]] = []
        converse_msgs: list[dict[str, Any]] = []
        for m in messages:
            role = m["role"]
            if role == "system":
                system_parts.append({"text": m["content"]})
            elif role == "tool":
                converse_msgs.append({
                    "role": "user",
                    "content": [{
                        "toolResult": {
                            "toolUseId": m.get("tool_call_id", ""),
                            "content": [{"text": m.get("content", "")}],
                        }
                    }],
                })
            elif role == "assistant" and m.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if m.get("content"):
                    blocks.append({"text": m["content"]})
                for c in m["tool_calls"]:
                    fn = c.get("function", c)
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    blocks.append({
                        "toolUse": {
                            "toolUseId": c.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": args or {},
                        }
                    })
                converse_msgs.append({"role": "assistant", "content": blocks})
            else:
                converse_msgs.append({
                    "role": role,
                    "content": [{"text": m.get("content", "")}],
                })
        return system_parts, converse_msgs

    def _chat_bedrock(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> ChatResponse:
        client = self._get_bedrock_client()
        system_parts, converse_msgs = self._bedrock_convert_messages(messages)

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

    def _chat_json_stream_bedrock(
        self, system: str, user: str, temperature: float, max_tokens: int,
    ):
        client = self._get_bedrock_client()
        kwargs: dict[str, Any] = {
            "modelId": self.bedrock_model_id,
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {"temperature": temperature, "maxTokens": max_tokens},
            "system": [{"text": system + "\n\nRespond ONLY with valid JSON."}],
        }
        # TODO(streaming-reasoning): to stream model reasoning from Bedrock,
        # switch to a reasoning-capable model (e.g. a Claude model id — the
        # default nova-micro has no reasoning mode) and enable it via
        #   kwargs["additionalModelRequestFields"] = {
        #       "thinking": {"type": "enabled", "budget_tokens": 2048}}
        # converse_stream then interleaves `contentBlockDelta` events whose
        # delta carries `reasoningContent` (thinking) alongside `text` —
        # yield them as separate ("thinking", token) / ("text", token) events
        # for chat_stream(). Docs: AWS Bedrock ConverseStream + "inference
        # reasoning".
        response = client.converse_stream(**kwargs)
        full_text = ""
        for event in response["stream"]:
            delta = (event.get("contentBlockDelta") or {}).get("delta") or {}
            token = delta.get("text")
            if not token:
                continue
            full_text += token
            yield token
        yield _safe_json(full_text)

    def _chat_tools_bedrock(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> ChatResponse:
        client = self._get_bedrock_client()
        system_parts, converse_msgs = self._bedrock_convert_messages(messages)

        kwargs: dict[str, Any] = {
            "modelId": self.bedrock_model_id,
            "messages": converse_msgs,
            "inferenceConfig": {"temperature": temperature, "maxTokens": max_tokens},
            "toolConfig": {"tools": [_bedrock_tool_spec(t) for t in tools]},
        }
        if system_parts:
            kwargs["system"] = system_parts

        response = client.converse(**kwargs)
        blocks = response["output"]["message"]["content"]
        text_parts: list[str] = []
        calls: list[dict[str, Any]] = []
        for block in blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                calls.append({
                    "id": tu.get("toolUseId", f"call_{len(calls)}"),
                    "name": tu.get("name", ""),
                    "arguments": tu.get("input") or {},
                })
        return ChatResponse(
            text="".join(text_parts),
            backend="bedrock",
            model=self.bedrock_model_id,
            raw=response,
            tool_calls=calls or None,
        )

    # ── GGUF (standalone, llama-cpp-python) ─────────────────────────────────

    def _get_gguf_llama(self):
        """Lazily construct (and cache) the in-process ``Llama`` instance.

        Loading is deferred to first use so ``detect_backend()`` stays cheap
        and a process that never actually calls the GGUF backend never pays
        the multi-GB load cost.
        """
        if self._gguf_llama is None:
            logger.info(
                "Loading GGUF model %s (n_ctx=%d, n_gpu_layers=%d) …",
                self.gguf_model_path, self.gguf_n_ctx, self.gguf_n_gpu_layers,
            )
            kwargs: dict[str, Any] = {
                "model_path": str(Path(self.gguf_model_path).expanduser()),
                "n_ctx": self.gguf_n_ctx,
                "n_gpu_layers": self.gguf_n_gpu_layers,
                "verbose": False,
            }
            if self.gguf_n_threads:
                kwargs["n_threads"] = self.gguf_n_threads
            if self.gguf_chat_format:
                kwargs["chat_format"] = self.gguf_chat_format
            self._gguf_llama = _Llama(**kwargs)
        return self._gguf_llama

    def _chat_gguf(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> ChatResponse:
        llama = self._get_gguf_llama()
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            # Grammar-constrained JSON generation — llama.cpp guarantees
            # syntactically valid JSON output when this is honoured by the
            # loaded chat handler (all built-in handlers support it).
            kwargs["response_format"] = {"type": "json_object"}
        result = llama.create_chat_completion(**kwargs)
        text = _strip_think_tags(
            (result.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        )
        return ChatResponse(
            text=text, backend="gguf",
            model=Path(self.gguf_model_path).name, raw=result,
        )

    def _chat_json_stream_gguf(
        self, system: str, user: str, temperature: float, max_tokens: int,
    ):
        llama = self._get_gguf_llama()
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        stream = llama.create_chat_completion(
            messages=msgs, temperature=temperature, max_tokens=max_tokens,
            response_format={"type": "json_object"}, stream=True,
        )
        # TODO(streaming-reasoning): for thinking GGUF models the reasoning
        # arrives INLINE as <think>…</think> tokens in this same stream.
        # Instead of stripping at the end, track whether the cursor is inside
        # a <think> block and route tokens to ("thinking", token) vs
        # ("text", token) events for chat_stream(); only the text channel is
        # accumulated for the final _safe_json parse.
        full_text = ""
        for chunk in stream:
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            token = delta.get("content") or ""
            if not token:
                continue
            full_text += token
            yield token
        yield _safe_json(_strip_think_tags(full_text))

    def _chat_tools_gguf(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> ChatResponse:
        """Best-effort tool-calling — quality depends on the GGUF model's
        chat template and whether ``llama-cpp-python`` recognises it as a
        function-calling-capable handler. Falls back to a plain text answer
        (no ``tool_calls``) if the response doesn't carry any, which the
        agent loop already treats as a normal final answer."""
        llama = self._get_gguf_llama()
        result = llama.create_chat_completion(
            messages=messages, tools=tools, temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = (result.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        text = _strip_think_tags(msg.get("content") or "")
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
            text=text, backend="gguf",
            model=Path(self.gguf_model_path).name,
            raw=result, tool_calls=calls or None,
        )

    # ── Chat ──────────────────────────────────────────────────────────────

    # TODO(streaming-reasoning): add a `chat_stream()` sibling of `chat()` that
    # returns a generator of typed events instead of a final ChatResponse:
    #     {"type": "thinking", "delta": str}   # model reasoning tokens
    #     {"type": "text",     "delta": str}   # answer tokens
    #     {"type": "done",     "text": str}    # full final text (JSON-parseable)
    # Dispatch per backend like chat() does; the per-backend TODOs below say
    # where each backend's reasoning tokens come from. Keep thinking and text
    # in separate event types — callers in JSON mode must only parse the
    # "done"/"text" channel, never the thinking channel.

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
        if backend == "gguf":
            return self._chat_gguf(messages, temperature, max_tokens, json_mode)
        if backend == "bedrock":
            return self._chat_bedrock(messages, temperature, max_tokens, json_mode)
        if backend == "azure":
            return self._chat_azure(messages, temperature, max_tokens, json_mode)
        return self._chat_stub(messages, json_mode)

    def _chat_azure(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> ChatResponse:
        """Azure OpenAI chat completions (deployment-scoped endpoint,
        api-key header). Same OpenAI message shape the rest of the client
        uses; json_mode maps to response_format json_object."""
        url = (f"{self.azure_endpoint.rstrip('/')}/openai/deployments/"
               f"{self.azure_deployment}/chat/completions")
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = httpx.post(
            url,
            params={"api-version": self.azure_api_version},
            headers={"api-key": self.azure_api_key},
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"] or ""
        return ChatResponse(text=text, backend="azure",
                            model=self.azure_deployment, raw=data)

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
        # For thinking-capable models (qwen3*, deepseek-r1) disable reasoning
        # with Ollama's native `think: false` request field — unlike the old
        # /no_think prompt hack it also works alongside format:json, where
        # letting the model think burned minutes of hidden <think> tokens per
        # call (and could exhaust the whole budget, yielding empty output).
        #
        # TODO(streaming-reasoning): for a streaming variant, POST /api/chat
        # with `"stream": true, "think": true`. Ollama then delivers each
        # chunk with the reasoning in `message.thinking` SEPARATE from
        # `message.content`, so JSON mode stays clean — no <think> tags to
        # strip. Iterate the ndjson lines with `httpx.stream(...)` and yield
        # ("thinking", delta) / ("text", delta) events for chat_stream().
        # Docs: ollama/docs/api.md, "thinking".
        msgs = _inject_spanish(list(messages))
        thinking = self._is_thinking_model()

        payload: dict[str, Any] = {
            "model": self.ollama_model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        if thinking:
            payload["think"] = False

        try:
            r = httpx.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if not (thinking and exc.response.status_code == 400):
                raise
            # Ollama < 0.9 rejects the `think` field — fall back to the old
            # behaviour: /no_think prompt in plain mode; in JSON mode let the
            # model think with extra budget and strip the tags afterwards.
            payload.pop("think", None)
            if json_mode:
                payload["options"]["num_predict"] = max_tokens + 4096
            else:
                payload["messages"] = _inject_no_think(msgs)
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

    # ── Streaming ─────────────────────────────────────────────────────────

    def chat_json_stream(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        """Yield (token, full_text_so_far) tuples, then final parsed JSON.

        Yields ``str`` tokens as they arrive. The last yield is a
        ``dict`` with the parsed JSON result (or an error dict).
        Implemented for Ollama, GGUF and Bedrock; other backends (LiteRT,
        stub) fall back to non-streaming and yield the full result at once.
        """
        backend = self.detect_backend()
        if backend == "gguf":
            yield from self._chat_json_stream_gguf(system, user, temperature, max_tokens)
            return
        if backend == "bedrock":
            yield from self._chat_json_stream_bedrock(system, user, temperature, max_tokens)
            return
        if backend != "ollama":
            result = self.chat_json(system, user, temperature=temperature, max_tokens=max_tokens)
            yield result
            return

        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        msgs = _inject_spanish(msgs)

        predict = max_tokens
        if self._is_thinking_model():
            predict = max_tokens + 4096

        payload: dict[str, Any] = {
            "model": self.ollama_model,
            "messages": msgs,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": predict},
            "format": "json",
        }

        full_text = ""
        in_think = False
        with httpx.stream(
            "POST",
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = (chunk.get("message") or {}).get("content", "")
                if not token:
                    continue
                full_text += token
                # Suppress <think> content from streaming output
                if "<think>" in token:
                    in_think = True
                if in_think:
                    if "</think>" in token:
                        in_think = False
                    continue
                yield token

        # Final parsed result
        cleaned = _strip_think_tags(full_text)
        yield _safe_json(cleaned)

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
        if backend == "gguf":
            return self._chat_tools_gguf(messages, tools, temperature, max_tokens)
        if backend == "bedrock":
            return self._chat_tools_bedrock(messages, tools, temperature, max_tokens)
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


_inspect_client: LocalLLMClient | None = None


def get_inspect_client() -> LocalLLMClient:
    """Lightweight client for the Excel inspection / sheet-mapping step.

    That step runs on every upload but only has to pick which sheet holds the
    dictionary and map its columns — a small, fast model does this well and
    keeps inspection snappy, instead of paying for the full chat model. It is
    configured independently of the main model, in ``config.yaml`` under
    ``llm`` (env vars in parentheses override):

        inspect_model            (INSPECT_MODEL)
            Ollama tag (e.g. ``"qwen3:1.7b"``) or a local ``.gguf`` path,
            which is auto-registered into Ollama like the main ``model``.
        inspect_gguf_model_path  (INSPECT_GGUF_MODEL_PATH)
            A ``.gguf`` weight file loaded in-process via llama-cpp-python
            (standalone ``gguf`` backend, no Ollama server). Tunable with
            ``inspect_gguf_n_ctx`` (INSPECT_GGUF_N_CTX) and
            ``inspect_gguf_n_gpu_layers`` (INSPECT_GGUF_N_GPU_LAYERS).

        inspect_bedrock_model_id (INSPECT_BEDROCK_MODEL_ID)
            A cheaper Bedrock model id for the inspection step on an
            all-Bedrock deployment, so a strong (e.g. Claude) generation
            model isn't wasted on trivial sheet mapping. Falls back to the
            main ``bedrock_model_id`` when unset.

    When none is configured this returns the main :func:`get_client`, so
    behaviour is unchanged until a lighter model is set up.
    """
    global _inspect_client
    if _inspect_client is not None:
        return _inspect_client

    inspect_gguf = (
        os.getenv("INSPECT_GGUF_MODEL_PATH")
        or _LLM_CFG.get("inspect_gguf_model_path")
    )
    inspect_bedrock = (
        os.getenv("INSPECT_BEDROCK_MODEL_ID")
        or _LLM_CFG.get("inspect_bedrock_model_id")
    )
    inspect_model = os.getenv("INSPECT_MODEL") or _LLM_CFG.get("inspect_model")

    if inspect_gguf:
        # Standalone GGUF weights loaded in-process — override the main
        # client's gguf settings on a fresh instance forced onto that backend.
        client = LocalLLMClient(prefer="gguf")
        client.gguf_model_path = str(inspect_gguf)
        n_ctx = os.getenv("INSPECT_GGUF_N_CTX") or _LLM_CFG.get("inspect_gguf_n_ctx")
        if n_ctx:
            client.gguf_n_ctx = int(n_ctx)
        n_gpu = os.getenv("INSPECT_GGUF_N_GPU_LAYERS")
        if n_gpu is None:
            n_gpu = _LLM_CFG.get("inspect_gguf_n_gpu_layers")
        if n_gpu is not None and str(n_gpu) != "":
            client.gguf_n_gpu_layers = int(n_gpu)
    elif inspect_bedrock:
        # A cheaper Bedrock model just for inspection (all-Bedrock deploys).
        client = LocalLLMClient(prefer="bedrock")
        client.bedrock_model_id = str(inspect_bedrock)
    elif inspect_model:
        # A dedicated (smaller) Ollama tag or auto-registered .gguf path.
        # On a Bedrock deployment REGLLM_LLM forces this client to Bedrock too;
        # the tag is then ignored and the main bedrock_model_id is used.
        client = LocalLLMClient(ollama_model=str(inspect_model))
    else:
        client = get_client()

    _inspect_client = client
    return _inspect_client


def reset_client() -> None:
    """Reset the singletons (useful in tests when env vars change)."""
    global _default_client, _inspect_client
    _default_client = None
    _inspect_client = None


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output (qwen3, deepseek-r1)."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _bedrock_tool_spec(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert one OpenAI/Ollama-format tool schema (as produced by
    ``ToolSpec.schema()``) into a Bedrock Converse ``toolConfig`` entry."""
    fn = tool.get("function", tool)
    return {
        "toolSpec": {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "inputSchema": {
                "json": fn.get("parameters") or {"type": "object", "properties": {}},
            },
        }
    }


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
    """Best-effort JSON extraction from a model response.

    Handles truncated JSON from models that hit token limits by attempting
    to repair incomplete output (close open arrays/objects).
    """
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
        # Attempt to repair truncated JSON by closing open brackets
        repaired = _repair_truncated_json(text)
        if repaired:
            return repaired
    return {"error": "invalid_json", "raw": text[:500]}


def _repair_truncated_json(text: str) -> dict[str, Any] | None:
    """Try to repair truncated JSON by closing open brackets/braces."""
    # Find the outermost opening brace
    start = text.find("{")
    if start < 0:
        return None
    fragment = text[start:]
    # Try progressively truncating from the last comma and closing
    # E.g. '{"dqcs": [{...}, {... incomplete' → '{"dqcs": [{...}]}'
    for _ in range(5):
        # Count open vs closed brackets
        opens = []
        for ch in fragment:
            if ch in "{[":
                opens.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if opens:
                    opens.pop()
        # Close remaining open brackets
        if opens:
            candidate = fragment + "".join(reversed(opens))
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        # Try truncating at last comma (remove incomplete last element)
        last_comma = fragment.rfind(",")
        if last_comma > 0:
            fragment = fragment[:last_comma]
        else:
            break
    return None


# ── Backwards-compatible aliases ────────────────────────────────────────────

GemmaClient = LocalLLMClient   # legacy name

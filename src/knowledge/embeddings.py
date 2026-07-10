"""Embedding client for semantic search.

Multi-backend, mirroring ``src.knowledge.llm_client.LocalLLMClient``'s
backend selection:

1. **Ollama** ``/api/embed`` endpoint — the original/default backend, e.g.
   ``nomic-embed-text`` (274 MB, runs on CPU).
2. **GGUF (standalone, `llama-cpp-python`)** — loads a local embedding
   ``.gguf`` file in-process, no Ollama server needed. Set
   ``GGUF_EMBED_MODEL_PATH``; falls back to the chat model at
   ``GGUF_MODEL_PATH`` if unset (works, but a dedicated embedding model
   gives materially better retrieval quality).
3. **Amazon Bedrock** — Titan Text Embeddings (``amazon.titan-embed-text-v2:0``
   by default) via the same IAM credentials the chat backend uses.
4. **Stub**: zero vectors when nothing is reachable/configured, so the rest
   of the system degrades gracefully to BM25-only / unranked search.

Configuration via environment variables:

- ``REGLLM_EMBED_BACKEND``    ``auto`` | ``ollama`` | ``gguf`` | ``bedrock`` | ``stub``
- ``OLLAMA_URL``              default ``http://localhost:11434``
- ``REGLLM_EMBED_MODEL``      default ``nomic-embed-text`` (Ollama tag)
- ``REGLLM_EMBED_DIM``        default ``768`` (nomic-embed-text dimension)
- ``GGUF_EMBED_MODEL_PATH``   path to a ``.gguf`` embedding model
- ``BEDROCK_REGION``          default ``eu-west-1``
- ``BEDROCK_EMBED_MODEL_ID``  default ``amazon.titan-embed-text-v2:0``

``auto`` tries Ollama first (the original, zero-config-change default), then
GGUF, then Bedrock, then falls back to zero vectors — a strict superset of
the previous Ollama-only behaviour: if none of the new backends are
configured, the outcome is identical to before.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import httpx

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

try:
    from llama_cpp import Llama as _Llama
except ImportError:  # llama-cpp-python not installed — gguf backend unavailable
    _Llama = None

try:
    import boto3 as _boto3
except ImportError:  # boto3 not installed — bedrock backend unavailable
    _boto3 = None

logger = logging.getLogger(__name__)


def _load_embed_cfg() -> dict[str, Any]:
    for candidate in (
        Path(__file__).resolve().parents[2] / "config.yaml",
        Path("config.yaml"),
    ):
        if candidate.is_file() and _yaml is not None:
            with open(candidate) as f:
                return (_yaml.safe_load(f) or {}).get("embedding", {})
    return {}


_EMBED_CFG = _load_embed_cfg()

_DEFAULT_MODEL = os.getenv("REGLLM_EMBED_MODEL") or _EMBED_CFG.get("model") or "nomic-embed-text"
_DEFAULT_DIM = int(os.getenv("REGLLM_EMBED_DIM") or _EMBED_CFG.get("dim") or 768)


class EmbeddingService:
    """Generates embeddings via Ollama, standalone GGUF, or Bedrock; falls
    back to zero vectors when nothing is reachable."""

    def __init__(
        self,
        ollama_url: str | None = None,
        model: str | None = None,
        dim: int | None = None,
        backend: str | None = None,
    ) -> None:
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL") or _EMBED_CFG.get("ollama_url") or "http://localhost:11434"
        self.model = model or _DEFAULT_MODEL
        self.dim = dim or _DEFAULT_DIM
        self.backend = (
            backend or os.getenv("REGLLM_EMBED_BACKEND")
            or _EMBED_CFG.get("backend") or "auto"
        )

        self.gguf_embed_model_path = (
            os.getenv("GGUF_EMBED_MODEL_PATH")
            or _EMBED_CFG.get("gguf_model_path")
            or os.getenv("GGUF_MODEL_PATH")  # fall back to the chat GGUF
            or ""
        )
        self._gguf_llama: Any = None

        self.bedrock_region = os.getenv("BEDROCK_REGION") or _EMBED_CFG.get("bedrock_region") or "eu-west-1"
        self.bedrock_embed_model_id = (
            os.getenv("BEDROCK_EMBED_MODEL_ID")
            or _EMBED_CFG.get("bedrock_embed_model_id")
            or "amazon.titan-embed-text-v2:0"
        )
        self._bedrock_client = None

        self._resolved: str | None = None
        self._client = httpx.Client(timeout=30.0)

    # ── backend resolution ───────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """Whether *some* real (non-stub) backend is usable."""
        return self._resolve() != "stub"

    @property
    def resolved_backend(self) -> str:
        """Which concrete backend ``embed()`` will actually use — as opposed
        to ``self.backend``, which is the requested preference (e.g.
        ``"auto"``). Resolving probes lazily on first access, same as
        ``available``."""
        return self._resolve()

    def _resolve(self) -> str:
        if self._resolved is not None:
            return self._resolved
        if self.backend == "stub":
            self._resolved = "stub"
        elif self.backend == "ollama":
            self._resolved = "ollama" if self._probe_ollama() else "stub"
        elif self.backend == "gguf":
            self._resolved = "gguf" if self._probe_gguf() else "stub"
        elif self.backend == "bedrock":
            self._resolved = "bedrock" if self._probe_bedrock() else "stub"
        else:  # auto
            if self._probe_ollama():
                self._resolved = "ollama"
            elif self._probe_gguf():
                self._resolved = "gguf"
            elif self._probe_bedrock():
                self._resolved = "bedrock"
            else:
                self._resolved = "stub"
        if self._resolved == "stub":
            logger.info("No embedding backend reachable/configured — using zero-vector stubs")
        return self._resolved

    def _probe_ollama(self) -> bool:
        try:
            resp = self._client.get(f"{self.ollama_url}/api/tags")
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False

    def _probe_gguf(self) -> bool:
        if _Llama is None or not self.gguf_embed_model_path:
            return False
        return Path(self.gguf_embed_model_path).expanduser().is_file()

    def _probe_bedrock(self) -> bool:
        # Checking "boto3 is importable" alone is not enough: boto3 is a
        # hard dependency of the DQC production image (used by the chat
        # backend), so it's *always* importable there — without a real
        # credential check, "auto" would silently prefer a misconfigured
        # Bedrock over falling back to stub. ``get_credentials()`` resolves
        # the standard chain (env vars, ~/.aws/credentials, or the ECS task
        # role via IMDS) without making a billed API call.
        if _boto3 is None:
            return False
        try:
            return _boto3.Session().get_credentials() is not None
        except Exception:
            return False

    # ── embedding ─────────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns zero vectors if backend unavailable."""
        if not texts:
            return []
        backend = self._resolve()
        try:
            if backend == "ollama":
                return self._embed_ollama(texts)
            if backend == "gguf":
                return self._embed_gguf(texts)
            if backend == "bedrock":
                return self._embed_bedrock(texts)
        except Exception as exc:
            logger.warning("Embedding request failed (%s): %s — returning zero vectors", backend, exc)
        return [self._zero_vector() for _ in texts]

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed([text])[0]

    def _embed_ollama(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def _get_gguf_llama(self):
        if self._gguf_llama is None:
            logger.info("Loading GGUF embedding model %s …", self.gguf_embed_model_path)
            self._gguf_llama = _Llama(
                model_path=str(Path(self.gguf_embed_model_path).expanduser()),
                embedding=True, verbose=False,
            )
        return self._gguf_llama

    def _embed_gguf(self, texts: list[str]) -> list[list[float]]:
        llama = self._get_gguf_llama()
        result = llama.create_embedding(texts)
        # llama.cpp returns one embedding vector per input, in order
        return [item["embedding"] for item in result["data"]]

    def _get_bedrock_client(self):
        if self._bedrock_client is None:
            self._bedrock_client = _boto3.client(
                "bedrock-runtime", region_name=self.bedrock_region,
            )
        return self._bedrock_client

    def _embed_bedrock(self, texts: list[str]) -> list[list[float]]:
        import json as _json
        client = self._get_bedrock_client()
        out: list[list[float]] = []
        for text in texts:
            resp = client.invoke_model(
                modelId=self.bedrock_embed_model_id,
                body=_json.dumps({"inputText": text}),
            )
            body = _json.loads(resp["body"].read())
            out.append(body["embedding"])
        return out

    def _zero_vector(self) -> list[float]:
        return [0.0] * self.dim


# ── Singleton ────────────────────────────────────────────────────────────────

_default: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _default
    if _default is None:
        _default = EmbeddingService()
    return _default


def reset_embedding_service() -> None:
    """Reset the singleton (useful in tests when env vars change)."""
    global _default
    _default = None

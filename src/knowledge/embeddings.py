"""Embedding client for semantic search.

Uses Ollama's ``/api/embed`` endpoint with a dedicated embedding model
(``nomic-embed-text`` by default, 274 MB, runs on CPU).

Falls back to a zero-vector stub when Ollama is unreachable, so the
rest of the system degrades gracefully to BM25-only search.

Configuration via environment variables:

- ``OLLAMA_URL``              default ``http://localhost:11434``
- ``REGLLM_EMBED_MODEL``      default ``nomic-embed-text``
- ``REGLLM_EMBED_DIM``        default ``768`` (nomic-embed-text dimension)
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
    """Generates embeddings via Ollama or returns zero vectors as fallback."""

    def __init__(
        self,
        ollama_url: str | None = None,
        model: str | None = None,
        dim: int | None = None,
    ) -> None:
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL") or _EMBED_CFG.get("ollama_url") or "http://localhost:11434"
        self.model = model or _DEFAULT_MODEL
        self.dim = dim or _DEFAULT_DIM
        self._available: bool | None = None
        self._client = httpx.Client(timeout=30.0)

    @property
    def available(self) -> bool:
        """Check whether the embedding backend is reachable."""
        if self._available is None:
            self._available = self._probe()
        return self._available

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns zero vectors if backend unavailable."""
        if not texts:
            return []
        if not self.available:
            return [self._zero_vector() for _ in texts]
        try:
            resp = self._client.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"]
        except Exception as exc:
            logger.warning("Embedding request failed: %s — returning zero vectors", exc)
            return [self._zero_vector() for _ in texts]

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed([text])[0]

    def _probe(self) -> bool:
        try:
            resp = self._client.get(f"{self.ollama_url}/api/tags")
            if resp.status_code != 200:
                logger.info("Ollama not reachable at %s — embeddings will use stubs", self.ollama_url)
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(self.model in m for m in models):
                logger.info(
                    "Embedding model %s not found in Ollama (available: %s) — will use stubs",
                    self.model, models,
                )
                return False
            return True
        except Exception:
            logger.info("Ollama not reachable — embeddings will use stubs")
            return False

    def _zero_vector(self) -> list[float]:
        return [0.0] * self.dim


# ── Singleton ────────────────────────────────────────────────────────────────

_default: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _default
    if _default is None:
        _default = EmbeddingService()
    return _default

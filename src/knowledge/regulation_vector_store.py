"""Lightweight, dependency-free vector store for the PD/LGD regulation RAG.

Deliberately **not** ChromaDB-backed (unlike ``src.knowledge.vector_store``,
which serves the SAS-diff explainer's knowledge-graph RAG): cosine similarity
over a few hundred short regulatory chunks is trivial in pure Python, and
staying dependency-free means the DQC production image
(``requirements-dqc.txt`` — no chromadb, no numpy) can serve embedding-based
regulation search without pulling in a compiled vector database. See
``docs/DEPLOYMENT.md``.

Persistence is a single JSON file: a ``manifest`` (embedding backend/model/
dim/build time, for cache-invalidation and debugging) plus the chunk records
themselves, each carrying its embedding vector inline.

Typical use::

    from src.knowledge.regulation_chunker import chunk_eba_guidelines
    from src.knowledge.embeddings import get_embedding_service
    from src.knowledge.regulation_vector_store import RegulationVectorStore

    store = RegulationVectorStore.build(chunk_eba_guidelines(), get_embedding_service())
    store.save()

    # later, e.g. inside a request handler:
    store = RegulationVectorStore.load()
    hits = store.search_text("suelo de LGD para hipotecas", get_embedding_service(), k=5)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .regulation_chunker import RegChunk, SOURCE_DOC

DEFAULT_INDEX_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "regulation" / "embeddings" / "pd_lgd_chunks.json"
)


@dataclass
class RegSearchHit:
    chunk_id: str
    paragraph: int
    section: str
    subsection: str
    text: str
    source_doc: str
    score: float
    citation: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "paragraph": self.paragraph,
            "section": self.section, "subsection": self.subsection,
            "text": self.text, "source_doc": self.source_doc,
            "score": self.score, "citation": self.citation,
        }


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity; 0.0 for a zero vector (undefined
    otherwise) rather than raising, so an unresolved/stub embedding backend
    degrades to "no ranking signal" instead of crashing retrieval."""
    if not a or not b or len(a) != len(b):
        return 0.0
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


class RegulationVectorStore:
    """In-memory chunk store with pure-Python cosine-similarity search."""

    def __init__(self, chunks: list[dict[str, Any]], manifest: dict[str, Any]):
        self._chunks = chunks
        self.manifest = manifest

    @property
    def size(self) -> int:
        return len(self._chunks)

    # ── search ────────────────────────────────────────────────────────────

    def search(self, query_embedding: list[float], k: int = 5) -> list[RegSearchHit]:
        if not self._chunks or not query_embedding:
            return []
        scored = sorted(
            (
                (cosine_similarity(query_embedding, c["embedding"]), c)
                for c in self._chunks
            ),
            key=lambda t: t[0], reverse=True,
        )
        return [
            RegSearchHit(
                chunk_id=c["chunk_id"], paragraph=c["paragraph"],
                section=c["section"], subsection=c["subsection"],
                text=c["text"], source_doc=c["source_doc"],
                score=round(score, 4), citation=f"{c['source_doc']} §{c['paragraph']}",
            )
            for score, c in scored[:k]
        ]

    def search_text(self, query: str, embedder, k: int = 5) -> list[RegSearchHit]:
        """Convenience: embed ``query`` with ``embedder`` (an
        ``EmbeddingService``-shaped object) and search."""
        return self.search(embedder.embed_one(query), k=k)

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: Path | str = DEFAULT_INDEX_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"manifest": self.manifest, "chunks": self._chunks}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_INDEX_PATH) -> "RegulationVectorStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(chunks=data["chunks"], manifest=data.get("manifest", {}))

    # ── build ─────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls, chunks: list[RegChunk], embedder, *, batch_size: int = 32,
    ) -> "RegulationVectorStore":
        """Embed ``chunks`` via ``embedder.embed(list[str])`` in batches and
        return a store ready to :meth:`save`."""
        texts = [c.text for c in chunks]
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            vectors.extend(embedder.embed(texts[i:i + batch_size]))
        dim = len(vectors[0]) if vectors and vectors[0] else 0
        stored = [{**c.to_dict(), "embedding": v} for c, v in zip(chunks, vectors)]
        manifest = {
            "source_doc": SOURCE_DOC,
            "n_chunks": len(chunks),
            "embedding_backend": getattr(embedder, "resolved_backend", "unknown"),
            "embedding_model": getattr(embedder, "model", ""),
            "dim": dim,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return cls(chunks=stored, manifest=manifest)


# ── module-level convenience: cached default-index loader ───────────────────

_default_store: RegulationVectorStore | None = None
_load_attempted = False


def get_default_store(path: Path | str = DEFAULT_INDEX_PATH) -> RegulationVectorStore | None:
    """Load (and cache) the index at ``path``; ``None`` if it doesn't exist
    or fails to parse, so callers can degrade gracefully instead of raising
    (the tool layer turns this into an informative, not fatal, result)."""
    global _default_store, _load_attempted
    if _default_store is not None:
        return _default_store
    if _load_attempted:
        return None
    _load_attempted = True
    try:
        _default_store = RegulationVectorStore.load(path)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None
    return _default_store


def reset_default_store() -> None:
    """Reset the module-level cache (tests only)."""
    global _default_store, _load_attempted
    _default_store = None
    _load_attempted = False

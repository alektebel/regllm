"""Hybrid search: BM25 + vector similarity with reciprocal rank fusion.

Wraps the existing ``DocsIndex`` (BM25) and ``VectorStore`` (ChromaDB)
to provide a single search interface.  When the vector store is empty
or the embedding service is unavailable, falls back to BM25 only.

Configuration:

- ``REGLLM_HYBRID_ALPHA``  weight for vector results (default ``0.5``).
  0.0 = pure BM25, 1.0 = pure vector.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from ..agent.docs_index import DocsIndex, SearchHit as BM25Hit, get_default_index
from .embeddings import EmbeddingService, get_embedding_service
from .vector_store import VectorStore, SearchHit as VectorHit

logger = logging.getLogger(__name__)

_DEFAULT_ALPHA = float(os.getenv("REGLLM_HYBRID_ALPHA", "0.5"))
_RRF_K = 60  # standard RRF constant


@dataclass
class HybridHit:
    """Merged search result from BM25 + vector."""
    id: str
    text: str
    score: float
    source: str  # "bm25", "vector", or "both"
    # Original BM25 hit fields (when available)
    path: str = ""
    heading: str = ""
    line_start: int = 0
    line_end: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": round(self.score, 4),
            "source": self.source,
            "path": self.path,
            "heading": self.heading,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }


class HybridIndex:
    """Combines BM25 keyword search with vector semantic search."""

    def __init__(
        self,
        bm25_index: DocsIndex | None = None,
        vector_store: VectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        alpha: float | None = None,
    ) -> None:
        self._bm25 = bm25_index
        self._vector = vector_store
        self._embed = embedding_service
        self._alpha = alpha if alpha is not None else _DEFAULT_ALPHA

    @property
    def bm25(self) -> DocsIndex:
        if self._bm25 is None:
            self._bm25 = get_default_index()
        return self._bm25

    @property
    def vector(self) -> VectorStore | None:
        return self._vector

    @property
    def embed(self) -> EmbeddingService:
        if self._embed is None:
            self._embed = get_embedding_service()
        return self._embed

    def search(self, query: str, k: int = 5) -> list[HybridHit]:
        """Search using BM25 + vector with reciprocal rank fusion.

        Falls back to BM25-only when vector store is empty or unavailable.
        """
        # BM25 results
        bm25_hits = self.bm25.search(query, k=k * 2)

        # Vector results (if available)
        vector_hits: list[VectorHit] = []
        if self._vector and self._vector.count() > 0 and self.embed.available:
            q_emb = self.embed.embed_one(query)
            vector_hits = self._vector.search(q_emb, k=k * 2)

        if not vector_hits:
            # Pure BM25 fallback
            return [self._from_bm25(h, i) for i, h in enumerate(bm25_hits[:k])]

        # Reciprocal Rank Fusion
        scores: dict[str, float] = {}
        meta: dict[str, HybridHit] = {}

        # Score BM25 hits
        for rank, hit in enumerate(bm25_hits):
            hit_id = f"bm25:{hit.section.path}:{hit.section.heading}"
            scores[hit_id] = scores.get(hit_id, 0.0) + (1 - self._alpha) / (rank + _RRF_K)
            if hit_id not in meta:
                meta[hit_id] = self._from_bm25(hit, rank)

        # Score vector hits
        for rank, hit in enumerate(vector_hits):
            hit_id = hit.id
            # Check if this ID matches a BM25 hit (merge)
            bm25_key = None
            for bk in meta:
                if bk.startswith("bm25:") and hit.id in bk:
                    bm25_key = bk
                    break

            if bm25_key:
                scores[bm25_key] = scores.get(bm25_key, 0.0) + self._alpha / (rank + _RRF_K)
                meta[bm25_key] = HybridHit(
                    id=bm25_key,
                    text=meta[bm25_key].text,
                    score=0,
                    source="both",
                    path=meta[bm25_key].path,
                    heading=meta[bm25_key].heading,
                    line_start=meta[bm25_key].line_start,
                    line_end=meta[bm25_key].line_end,
                )
            else:
                scores[hit_id] = scores.get(hit_id, 0.0) + self._alpha / (rank + _RRF_K)
                if hit_id not in meta:
                    meta[hit_id] = HybridHit(
                        id=hit_id,
                        text=hit.text,
                        score=0,
                        source="vector",
                        path=hit.metadata.get("path", ""),
                        heading=hit.metadata.get("heading", ""),
                    )

        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        results: list[HybridHit] = []
        for hit_id, score in ranked:
            h = meta[hit_id]
            h.score = score
            results.append(h)
        return results

    @staticmethod
    def _from_bm25(hit: BM25Hit, rank: int) -> HybridHit:
        return HybridHit(
            id=f"bm25:{hit.section.path}:{hit.section.heading}",
            text=hit.snippet,
            score=hit.score,
            source="bm25",
            path=hit.section.path,
            heading=hit.section.heading,
            line_start=hit.section.line_start,
            line_end=hit.section.line_end,
        )


# ── Singleton ────────────────────────────────────────────────────────────────

_default: HybridIndex | None = None


def get_hybrid_index(
    vector_store: VectorStore | None = None,
) -> HybridIndex:
    global _default
    if _default is None:
        _default = HybridIndex(vector_store=vector_store)
    return _default

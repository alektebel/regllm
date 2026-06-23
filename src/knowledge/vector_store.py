"""ChromaDB-backed vector store for semantic search.

Provides a simple interface for upserting and searching document
embeddings.  Uses ChromaDB in embedded (persistent) mode — no server
needed.

Usage::

    from src.knowledge.vector_store import VectorStore
    from src.knowledge.embeddings import get_embedding_service

    vs = VectorStore("data/knowledge/vectors")
    emb = get_embedding_service()

    vs.upsert("doc_1", "Article 15 text...", emb.embed_one("Article 15 text..."),
              metadata={"node_type": "Regulation", "article_id": "art15"})

    results = vs.search(emb.embed_one("LGD floor"), k=5)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb

logger = logging.getLogger(__name__)

_DEFAULT_COLLECTION = "regllm"


class VectorStore:
    """Persistent ChromaDB store for document embeddings."""

    def __init__(
        self,
        path: str | Path = "data/knowledge/vectors",
        collection_name: str = _DEFAULT_COLLECTION,
    ) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        doc_id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a document with its embedding."""
        # ChromaDB requires metadata values to be str, int, float, or bool
        # and rejects empty dicts — pass None when no metadata
        clean_meta = None
        if metadata:
            filtered = {
                k: v for k, v in metadata.items()
                if isinstance(v, (str, int, float, bool))
            }
            if filtered:
                clean_meta = filtered
        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[clean_meta] if clean_meta else None,
        )

    def upsert_batch(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Batch upsert."""
        clean_metas = None
        if metadatas:
            clean_metas = []
            for m in metadatas:
                filtered = {k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))}
                clean_metas.append(filtered if filtered else None)
            if all(x is None for x in clean_metas):
                clean_metas = None
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=clean_metas,
        )

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Search by embedding similarity. Returns ranked hits."""
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, self._collection.count() or 1),
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        hits: list[SearchHit] = []
        if not results["ids"] or not results["ids"][0]:
            return hits

        for i, doc_id in enumerate(results["ids"][0]):
            hits.append(SearchHit(
                id=doc_id,
                text=results["documents"][0][i] if results["documents"] else "",
                score=1.0 - (results["distances"][0][i] if results["distances"] else 0.0),
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
            ))
        return hits

    def delete(self, doc_id: str) -> None:
        """Delete a document by id."""
        self._collection.delete(ids=[doc_id])

    def count(self) -> int:
        return self._collection.count()


class SearchHit:
    """A single search result."""

    __slots__ = ("id", "text", "score", "metadata")

    def __init__(
        self,
        id: str,
        text: str,
        score: float,
        metadata: dict[str, Any],
    ) -> None:
        self.id = id
        self.text = text
        self.score = score
        self.metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"SearchHit(id={self.id!r}, score={self.score:.3f})"

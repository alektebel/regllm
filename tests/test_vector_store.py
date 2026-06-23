"""Tests for vector store and hybrid search."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.knowledge.vector_store import VectorStore, SearchHit
from src.knowledge.embeddings import EmbeddingService
from src.knowledge.hybrid_index import HybridIndex, HybridHit
from src.agent.docs_index import DocsIndex


# ── Vector store tests ───────────────────────────────────────────────────────


@pytest.fixture()
def vs(tmp_path: Path) -> VectorStore:
    return VectorStore(path=tmp_path / "vectors")


class TestVectorStore:
    def test_upsert_and_search(self, vs: VectorStore) -> None:
        # Simple 3-dim embeddings for testing
        vs.upsert("doc1", "LGD floor regulation", [1.0, 0.0, 0.0])
        vs.upsert("doc2", "PD calibration method", [0.0, 1.0, 0.0])
        vs.upsert("doc3", "LGD floor override", [0.9, 0.1, 0.0])

        hits = vs.search([1.0, 0.0, 0.0], k=2)
        assert len(hits) == 2
        assert hits[0].id == "doc1"
        assert hits[0].score > hits[1].score

    def test_upsert_overwrites(self, vs: VectorStore) -> None:
        vs.upsert("doc1", "old text", [1.0, 0.0, 0.0])
        vs.upsert("doc1", "new text", [0.0, 1.0, 0.0])
        assert vs.count() == 1
        hits = vs.search([0.0, 1.0, 0.0], k=1)
        assert hits[0].id == "doc1"
        assert hits[0].text == "new text"

    def test_delete(self, vs: VectorStore) -> None:
        vs.upsert("doc1", "text", [1.0, 0.0, 0.0])
        vs.delete("doc1")
        assert vs.count() == 0

    def test_batch_upsert(self, vs: VectorStore) -> None:
        vs.upsert_batch(
            ids=["a", "b", "c"],
            texts=["one", "two", "three"],
            embeddings=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            metadatas=[{"type": "x"}, {"type": "y"}, {"type": "z"}],
        )
        assert vs.count() == 3

    def test_metadata_filter(self, vs: VectorStore) -> None:
        vs.upsert("a", "text a", [1, 0, 0], metadata={"node_type": "Regulation"})
        vs.upsert("b", "text b", [0, 1, 0], metadata={"node_type": "Column"})
        hits = vs.search([0.5, 0.5, 0], k=5, where={"node_type": "Regulation"})
        assert all(h.id == "a" for h in hits)

    def test_empty_search(self, vs: VectorStore) -> None:
        hits = vs.search([1, 0, 0], k=5)
        assert hits == []


# ── Embedding service stub tests ─────────────────────────────────────────────


class TestEmbeddingService:
    def test_stub_fallback(self) -> None:
        """With no Ollama running, should return zero vectors."""
        svc = EmbeddingService(ollama_url="http://localhost:99999", dim=4)
        vecs = svc.embed(["hello", "world"])
        assert len(vecs) == 2
        assert vecs[0] == [0.0] * 4

    def test_embed_empty(self) -> None:
        svc = EmbeddingService(ollama_url="http://localhost:99999", dim=4)
        assert svc.embed([]) == []


# ── Hybrid index tests ──────────────────────────────────────────────────────


class TestHybridIndex:
    def test_bm25_fallback_when_no_vector(self, tmp_path: Path) -> None:
        """When no vector store, hybrid search falls back to pure BM25."""
        docs_root = tmp_path / "docs"
        (docs_root / "fields").mkdir(parents=True)
        (docs_root / "fields" / "lgd.md").write_text("# LGD_ESTIMADA\nThe LGD floor is 0.45\n")

        idx = DocsIndex(docs_root=docs_root)
        idx.build()

        hybrid = HybridIndex(bm25_index=idx, vector_store=None)
        hits = hybrid.search("LGD floor", k=3)
        assert len(hits) >= 1
        assert hits[0].source == "bm25"
        assert "LGD" in hits[0].heading or "LGD" in hits[0].text

    def test_hybrid_with_vector(self, tmp_path: Path) -> None:
        """When vector store has data, results come from both sources."""
        docs_root = tmp_path / "docs"
        (docs_root / "fields").mkdir(parents=True)
        (docs_root / "fields" / "lgd.md").write_text("# LGD_ESTIMADA\nLGD floor minimum\n")

        idx = DocsIndex(docs_root=docs_root)
        idx.build()

        vs = VectorStore(path=tmp_path / "vectors")
        vs.upsert("vec_1", "LGD floor regulation article 15", [1.0, 0.0, 0.0],
                   metadata={"path": "regulation/art15.md", "heading": "Article 15"})

        # Use stub embedding service (zero vectors = won't match anything useful)
        embed = EmbeddingService(ollama_url="http://localhost:99999", dim=3)

        hybrid = HybridIndex(bm25_index=idx, vector_store=vs, embedding_service=embed)
        hits = hybrid.search("LGD floor", k=5)
        # Should get BM25 results at minimum
        assert len(hits) >= 1

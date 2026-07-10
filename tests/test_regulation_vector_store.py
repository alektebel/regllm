"""Tests for src/knowledge/regulation_vector_store.py.

Uses a deterministic fake embedder (no real backend / network) so the
build → save → load → search round trip is fully reproducible in CI.
"""

from __future__ import annotations

import json

import pytest

from src.knowledge.regulation_chunker import RegChunk
from src.knowledge.regulation_vector_store import (
    RegulationVectorStore, cosine_similarity, get_default_store,
    reset_default_store,
)


class _FakeEmbedder:
    """Deterministic bag-of-words-ish embedder: dimension = len(vocab),
    each coordinate is the count of that vocab word in the text. Similar
    texts (shared words) get high cosine similarity — good enough to prove
    real ranking behaviour without a real model."""

    VOCAB = ["lgd", "suelo", "hipoteca", "pd", "default", "moc"]
    model = "fake-bow"
    resolved_backend = "stub"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        lower = text.lower()
        return [float(lower.count(w)) for w in self.VOCAB]


def _chunks() -> list[RegChunk]:
    return [
        RegChunk("par_1#0", 1, "LGD", "", "El suelo de LGD para hipoteca es 30%.", 0, 1),
        RegChunk("par_2#0", 2, "PD", "", "La PD de default se calibra anualmente.", 0, 1),
        RegChunk("par_3#0", 3, "MoC", "", "El margen de cautela (MoC) cubre deficiencias.", 0, 1),
    ]


# ── cosine_similarity ────────────────────────────────────────────────────────

def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_zero_not_nan():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([], []) == 0.0


# ── build / search ───────────────────────────────────────────────────────────

def test_build_embeds_every_chunk():
    store = RegulationVectorStore.build(_chunks(), _FakeEmbedder())
    assert store.size == 3
    assert store.manifest["n_chunks"] == 3
    assert store.manifest["dim"] == len(_FakeEmbedder.VOCAB)
    assert store.manifest["embedding_model"] == "fake-bow"


def test_search_ranks_relevant_chunk_first():
    store = RegulationVectorStore.build(_chunks(), _FakeEmbedder())
    hits = store.search_text("¿Cuál es el suelo de LGD hipotecaria?", _FakeEmbedder(), k=2)
    assert hits[0].paragraph == 1
    assert hits[0].citation == "EBA GL 2017/16 §1"
    assert hits[0].score > hits[1].score


def test_search_respects_k():
    store = RegulationVectorStore.build(_chunks(), _FakeEmbedder())
    assert len(store.search_text("lgd", _FakeEmbedder(), k=1)) == 1
    assert len(store.search_text("lgd", _FakeEmbedder(), k=10)) == 3


def test_search_empty_store_returns_no_hits():
    store = RegulationVectorStore(chunks=[], manifest={})
    assert store.search([1.0, 2.0], k=5) == []


def test_search_empty_query_embedding_returns_no_hits():
    store = RegulationVectorStore.build(_chunks(), _FakeEmbedder())
    assert store.search([], k=5) == []


# ── persistence round trip ───────────────────────────────────────────────────

def test_save_and_load_round_trip(tmp_path):
    store = RegulationVectorStore.build(_chunks(), _FakeEmbedder())
    path = store.save(tmp_path / "idx.json")
    assert path.exists()

    loaded = RegulationVectorStore.load(path)
    assert loaded.size == store.size
    assert loaded.manifest == store.manifest
    hits = loaded.search_text("suelo hipoteca", _FakeEmbedder(), k=1)
    assert hits[0].paragraph == 1


def test_saved_json_is_human_readable_and_self_contained(tmp_path):
    store = RegulationVectorStore.build(_chunks(), _FakeEmbedder())
    path = store.save(tmp_path / "idx.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "manifest" in data and "chunks" in data
    assert data["chunks"][0]["chunk_id"] == "par_1#0"
    assert "embedding" in data["chunks"][0]


# ── default-store caching / graceful-missing behaviour ──────────────────────

def test_get_default_store_returns_none_when_missing(tmp_path, monkeypatch):
    reset_default_store()
    missing = tmp_path / "nope.json"
    assert get_default_store(missing) is None
    # second call must not re-attempt parsing a nonexistent file and crash
    assert get_default_store(missing) is None
    reset_default_store()


def test_get_default_store_caches_after_first_load(tmp_path):
    reset_default_store()
    store = RegulationVectorStore.build(_chunks(), _FakeEmbedder())
    path = tmp_path / "idx.json"
    store.save(path)
    first = get_default_store(path)
    assert first is not None and first.size == 3
    # calling again returns the SAME cached object even if we point at a
    # different (nonexistent) path — proves it's cached, not re-read
    second = get_default_store(tmp_path / "does_not_matter.json")
    assert second is first
    reset_default_store()

"""Tests that /dqc/generate's context gathering wires together BOTH RAG
channels: context RAG (SAS lineage/formula + docs BM25) and regulation RAG
(graph traversal + the new embedding-based semantic search over the chunked
EBA GL/2017/16 PD & LGD guidelines)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("REGLLM_LLM", "stub")

from api.routers.dqc import _extract_sources, _gather_context  # noqa: E402
from src.knowledge.regulation_chunker import RegChunk  # noqa: E402
from src.knowledge.regulation_vector_store import RegulationVectorStore  # noqa: E402


# ── _gather_context wires both regulation channels ──────────────────────────

def test_gather_context_includes_both_regulation_channels():
    ctx = _gather_context("PD_ESTIMADA", "default")
    assert "regulation" in ctx           # graph-based (existing)
    assert "regulation_semantic" in ctx  # embedding-based (new)
    assert "docs" in ctx                 # context RAG (existing)


def test_gather_context_survives_missing_semantic_index():
    """No index built yet: must degrade gracefully, not raise / poison ctx."""
    ctx = _gather_context("SOME_UNKNOWN_FIELD_XYZ", "default")
    reg_sem = ctx["regulation_semantic"]
    assert reg_sem.get("available") is False
    assert "hint" in reg_sem


# ── _extract_sources turns semantic hits into RAGSource ─────────────────────

def test_extract_sources_maps_semantic_hits():
    ctx = {
        "regulation_semantic": {
            "available": True,
            "hits": [{
                "chunk_id": "par_73#0", "paragraph": 73, "section": "5.3 Calibración de la PD",
                "subsection": "5.3.2 Cálculo de las tasas de default",
                "text": "Para calcular la tasa de default a un año...",
                "source_doc": "EBA GL 2017/16", "score": 0.87,
                "citation": "EBA GL 2017/16 §73",
            }],
        },
    }
    sources = _extract_sources(ctx)
    assert len(sources) == 1
    s = sources[0]
    assert s.source_type == "regulation_semantic"
    assert s.document == "EBA GL 2017/16 §73"
    assert "5.3.2" in s.heading
    assert s.snippet.startswith("Para calcular")


def test_extract_sources_deduplicates_semantic_hits_by_chunk_id():
    ctx = {
        "regulation_semantic": {"hits": [
            {"chunk_id": "par_1#0", "paragraph": 1, "section": "", "subsection": "",
             "text": "x", "source_doc": "EBA GL 2017/16", "citation": "EBA GL 2017/16 §1"},
            {"chunk_id": "par_1#0", "paragraph": 1, "section": "", "subsection": "",
             "text": "x", "source_doc": "EBA GL 2017/16", "citation": "EBA GL 2017/16 §1"},
        ]},
    }
    sources = _extract_sources(ctx)
    assert len(sources) == 1


def test_extract_sources_combines_graph_and_semantic_regulation():
    """Both regulation channels can surface sources simultaneously, and the
    dedup key namespaces them separately (a graph hit and a semantic hit
    sharing a coincidental id must not collide)."""
    ctx = {
        "regulation": {"evidence": [
            {"section_id": "sec_1", "document": "Circular 6/2016", "heading": "Art. 15", "snippet": "y"},
        ]},
        "regulation_semantic": {"hits": [
            {"chunk_id": "sec_1", "paragraph": 15, "section": "", "subsection": "",
             "text": "z", "source_doc": "EBA GL 2017/16", "citation": "EBA GL 2017/16 §15"},
        ]},
    }
    sources = _extract_sources(ctx)
    assert len(sources) == 2
    types = {s.source_type for s in sources}
    assert types == {"regulation", "regulation_semantic"}


# ── end-to-end: /dqc/generate surfaces a semantic-regulation source ────────

class _FakeEmbedder:
    model = "fake"
    resolved_backend = "stub"

    def embed(self, texts):
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text):
        low = text.lower()
        return [float(low.count(w)) for w in ("pd", "estimada", "suelo")]


@pytest.fixture()
def fake_regulation_index(monkeypatch):
    chunks = [
        RegChunk("par_1#0", 1, "PD", "", "La PD_ESTIMADA respeta un suelo regulatorio.", 0, 1),
        RegChunk("par_2#0", 2, "LGD", "", "La LGD tiene un suelo distinto.", 0, 1),
    ]
    store = RegulationVectorStore.build(chunks, _FakeEmbedder())
    monkeypatch.setattr(
        "src.knowledge.regulation_vector_store.get_default_store",
        lambda *a, **k: store,
    )
    return store


def test_dqc_generate_includes_semantic_regulation_source(fake_regulation_index):
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    resp = client.post(
        "/dqc/generate",
        json={"message": "Genera DQCs para PD_ESTIMADA", "session_id": "rag_test"},
    )
    assert resp.status_code == 200
    body = resp.json()
    semantic_sources = [s for s in body["sources"] if s["source_type"] == "regulation_semantic"]
    assert semantic_sources, f"no semantic regulation source in {body['sources']}"
    assert semantic_sources[0]["document"] == "EBA GL 2017/16 §1"

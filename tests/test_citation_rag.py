"""
Unit tests for src/citation_rag.py — CitationRAG and module-level helpers.

Uses real CitationRAG instances with an isolated tmp_path ChromaDB.
The sentence-transformer model is loaded from cache (CPU, no GPU required).

Note: test_index_tree_* tests load data/citation_trees/eu_regulations.json
and may take ~10–30 s on first run while embeddings are computed.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.citation_rag import CitationRAG, _parse_reference, _segment_by_article

TREE_PATH = str(PROJECT_ROOT / "data" / "citation_trees" / "eu_regulations.json")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def fresh_crag(tmp_path) -> CitationRAG:
    """Return a CitationRAG backed by a temporary, empty ChromaDB."""
    return CitationRAG(chroma_path=str(tmp_path / "chroma"))


# ─── count / empty ────────────────────────────────────────────────────────────

def test_empty_count(tmp_path):
    crag = fresh_crag(tmp_path)
    assert crag.count() == 0


# ─── index_citation_tree ──────────────────────────────────────────────────────

@pytest.mark.slow
def test_index_tree_adds_items(tmp_path):
    crag = fresh_crag(tmp_path)
    added = crag.index_citation_tree(TREE_PATH)
    assert added > 0
    assert crag.count() == added


@pytest.mark.slow
def test_index_tree_idempotent(tmp_path):
    crag = fresh_crag(tmp_path)
    first = crag.index_citation_tree(TREE_PATH)
    second = crag.index_citation_tree(TREE_PATH)
    # Second call should add 0 new items (all already exist)
    assert second == 0
    assert crag.count() == first


# ─── search ───────────────────────────────────────────────────────────────────

def test_search_empty_collection(tmp_path):
    crag = fresh_crag(tmp_path)
    results = crag.search("CET1 ratio requirements", top_k=5)
    assert results == []


@pytest.mark.slow
def test_search_returns_results(tmp_path):
    crag = fresh_crag(tmp_path)
    crag.index_citation_tree(TREE_PATH)
    results = crag.search("CET1 capital ratio", top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0


@pytest.mark.slow
def test_search_result_schema(tmp_path):
    crag = fresh_crag(tmp_path)
    crag.index_citation_tree(TREE_PATH)
    results = crag.search("leverage ratio", top_k=2)
    for r in results:
        assert "reference" in r
        assert "documento" in r
        assert "articulo" in r
        assert "text" in r
        assert "score" in r


# ─── index_regulation_doc ─────────────────────────────────────────────────────

def test_index_regulation_doc(tmp_path):
    # Build a minimal regulation JSON file
    doc_data = [
        {"text": "Artículo 1. Ámbito de aplicación. Este reglamento se aplica a las entidades de crédito."},
        {"text": "Artículo 2. Definiciones. A efectos del presente reglamento se entenderá por entidad de crédito."},
    ]
    doc_path = tmp_path / "test_reg.json"
    doc_path.write_text(json.dumps(doc_data), encoding="utf-8")

    crag = fresh_crag(tmp_path)
    added = crag.index_regulation_doc(str(doc_path), "TestReg")
    assert added > 0
    assert crag.count() == added


# ─── _parse_reference ─────────────────────────────────────────────────────────

def test_parse_reference_crr():
    doc, art, par = _parse_reference("CRR Art. 92 §1(a)", "article")
    assert doc == "CRR"
    assert "92" in art
    assert par == "§1(a)"


def test_parse_reference_no_article():
    doc, art, par = _parse_reference("SomeName", "section")
    assert doc == "SomeName"
    # No article pattern found → articulo is empty or equals reference
    # (non-article node_type → empty)
    assert par == ""


# ─── _segment_by_article ──────────────────────────────────────────────────────

def test_segment_with_articles():
    text = (
        "Artículo 1. Ámbito. Este reglamento establece los requisitos. "
        "Artículo 2. Definiciones. Se entiende por banco toda entidad."
    )
    chunks = _segment_by_article(text)
    assert len(chunks) >= 2
    # Each chunk is a (articulo_header, chunk_text) tuple
    headers = [c[0] for c in chunks]
    assert any("1" in h for h in headers)
    assert any("2" in h for h in headers)


def test_segment_no_structure():
    text = "Este es un texto sin estructura de artículos y tiene más de cincuenta caracteres aquí."
    chunks = _segment_by_article(text)
    assert len(chunks) == 1
    assert chunks[0][0] == ""  # no article header
    assert text.strip() in chunks[0][1]

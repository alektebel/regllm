"""
Unit tests for src/chat_engine.py — ChatEngine pipeline and helpers.

No GPU, no DB, no network required.
RegulatoryRAGSystem and CitationRAG are replaced with MagicMock instances.
"""

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chat_engine import ChatEngine, _content_to_text, SYSTEM_PROMPT, REJECTION_CARD


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_engine(citation_rag=None):
    """Build a ChatEngine with a MagicMock RAG system."""
    rag = MagicMock()
    rag.buscar_hibrida.return_value = []
    rag.buscar_contexto.return_value = []
    return ChatEngine(rag, citation_rag=citation_rag, db_source="test")


# ─── parse_response ───────────────────────────────────────────────────────────

def test_parse_valid_json():
    engine = make_engine()
    raw = json.dumps({
        "respuesta": "El ratio CET1 mínimo es 4.5%.",
        "referencias": [{"documento": "CRR", "articulo": "Art. 92", "paragrafo": "§1(a)", "descripcion": "Capital"}],
        "confianza": 90,
        "justificacion_confianza": "Directamente citado.",
    })
    result = engine.parse_response(raw)
    assert result["respuesta"] == "El ratio CET1 mínimo es 4.5%."
    assert result["confianza"] == 90
    assert len(result["referencias"]) == 1
    assert result["referencias"][0]["documento"] == "CRR"


def test_parse_json_in_code_fences():
    engine = make_engine()
    raw = '```json\n{"respuesta": "Respuesta.", "referencias": [], "confianza": 75, "justificacion_confianza": ""}\n```'
    result = engine.parse_response(raw)
    assert result["respuesta"] == "Respuesta."
    assert result["confianza"] == 75


def test_parse_fallback_plain_text():
    engine = make_engine()
    raw = "Esta es una respuesta en texto plano sin JSON."
    result = engine.parse_response(raw)
    assert result["respuesta"] == raw
    assert result["referencias"] == []
    assert result["confianza"] is None
    assert result["justificacion_confianza"] == ""


def test_parse_malformed_json():
    engine = make_engine()
    raw = '{"respuesta": "Texto incompleto", "referencias": [{'  # malformed
    result = engine.parse_response(raw)
    # Must not raise; falls back to plain text
    assert result["respuesta"] == raw
    assert result["referencias"] == []


# ─── render_message ───────────────────────────────────────────────────────────

def test_render_with_references():
    engine = make_engine()
    parsed = {
        "respuesta": "El capital mínimo es del 8%.",
        "referencias": [
            {"documento": "CRR", "articulo": "Art. 92", "paragrafo": "§1", "descripcion": "Capital total"},
        ],
        "confianza": None,
        "justificacion_confianza": "",
    }
    rendered = engine.render_message(parsed)
    assert "El capital mínimo es del 8%." in rendered
    assert 'resp-card' in rendered
    assert 'resp-section--refs' in rendered
    assert "CRR" in rendered
    assert "Art. 92" in rendered


def test_render_with_confidence():
    engine = make_engine()
    parsed = {
        "respuesta": "Basilea III requiere LCR ≥ 100%.",
        "referencias": [],
        "confianza": 82,
        "justificacion_confianza": "Norma bien establecida.",
    }
    rendered = engine.render_message(parsed)
    assert "82%" in rendered
    assert "Confianza" in rendered
    assert "Norma bien establecida." in rendered


def test_render_minimal():
    engine = make_engine()
    parsed = {
        "respuesta": "Solo texto.",
        "referencias": [],
        "confianza": None,
        "justificacion_confianza": "",
    }
    rendered = engine.render_message(parsed)
    assert "Solo texto." in rendered
    assert 'resp-card' in rendered
    assert "<details>" not in rendered


# ─── _content_to_text ─────────────────────────────────────────────────────────

def test_content_to_text_string():
    assert _content_to_text("hola") == "hola"


def test_content_to_text_list():
    # Gradio 6 content-dict list format
    content = [{"type": "text", "text": "Hola"}, {"type": "text", "text": " mundo"}]
    result = _content_to_text(content)
    assert "Hola" in result
    assert "mundo" in result


# ─── build_messages ───────────────────────────────────────────────────────────

def test_build_messages_no_history():
    engine = make_engine()
    messages = engine.build_messages("¿Qué es CET1?", "Contexto aquí.", [])
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[-1]["role"] == "user"
    assert "¿Qué es CET1?" in messages[-1]["content"]
    assert "Contexto aquí." in messages[-1]["content"]
    assert len(messages) == 2


def test_build_messages_with_history():
    engine = make_engine()
    history = [
        {"role": "user", "content": "Primera pregunta"},
        {"role": "assistant", "content": "Primera respuesta"},
    ]
    messages = engine.build_messages("Segunda pregunta", "Contexto.", history)
    # system + 2 history turns + user = 4
    assert len(messages) == 4
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Primera pregunta"
    assert messages[2]["role"] == "assistant"
    assert messages[-1]["content"].endswith("Segunda pregunta")


# ─── enrich_references ────────────────────────────────────────────────────────

def test_enrich_references_no_citation_rag():
    engine = make_engine(citation_rag=None)
    parsed = {
        "respuesta": "Texto.",
        "referencias": [{"documento": "CRR", "articulo": "Art. 1", "paragrafo": "", "descripcion": "x"}],
        "confianza": None,
        "justificacion_confianza": "",
    }
    result = engine.enrich_references(parsed, "cualquier pregunta")
    # No citation_rag → referencias cleared, confianza None
    assert result["referencias"] == []
    assert result["confianza"] is None
    assert result["justificacion_confianza"] == ""


def test_enrich_references_overwrites_from_db():
    """References come exclusively from the DB; model-generated ones are discarded."""
    citation_rag = MagicMock()
    db_hits = [
        {"documento": "CRR", "articulo": "Art. 92", "paragrafo": "§1", "text": "Capital requirements", "score": 0.85},
        {"documento": "CRD", "articulo": "Art. 10", "paragrafo": "", "text": "Governance rules", "score": 0.72},
    ]
    citation_rag.search.return_value = db_hits
    engine = make_engine(citation_rag=citation_rag)
    parsed = {
        "respuesta": "El capital mínimo es 8%.",
        "referencias": [
            # model-hallucinated ref that should be discarded
            {"documento": "INVENTED", "articulo": "Art. 999", "paragrafo": "", "descripcion": "fake"},
        ],
        "confianza": None,
        "justificacion_confianza": "",
    }
    result = engine.enrich_references(parsed, "capital ratios")
    # Model ref discarded; only DB hits remain
    docs = [r["documento"] for r in result["referencias"]]
    assert "INVENTED" not in docs
    assert "CRR" in docs
    assert "CRD" in docs
    assert len(result["referencias"]) == 2


def test_enrich_references_computes_confidence():
    """confianza is the mean cosine similarity of the answer against citation nodes."""
    citation_rag = MagicMock()
    citation_rag.search.return_value = [
        {"documento": "CRR", "articulo": "Art. 92", "paragrafo": "§1", "text": "Capital", "score": 0.80},
        {"documento": "CRD", "articulo": "Art. 10", "paragrafo": "", "text": "Gov", "score": 0.60},
    ]
    engine = make_engine(citation_rag=citation_rag)
    parsed = {
        "respuesta": "El capital mínimo es 8%.",
        "referencias": [],
        "confianza": None,
        "justificacion_confianza": "",
    }
    result = engine.enrich_references(parsed, "requisitos de capital")
    # mean of (0.80, 0.60) = 0.70 → 70%
    assert result["confianza"] == 70
    assert "0.70" in result["justificacion_confianza"]


# ─── check_topic ──────────────────────────────────────────────────────────────

def test_check_topic_allows_regulatory_question():
    engine = make_engine()
    with patch("src.chat_engine._topic_embedding_score", return_value=0.72):
        assert engine.check_topic("¿Qué establece el artículo 92 del CRR sobre capital?") is True


def test_check_topic_blocks_poem_request():
    """Caught by regex — embedding guard is never called."""
    engine = make_engine()
    assert engine.check_topic("Hazme una poesía de riesgo de crédito") is False


def test_check_topic_blocks_joke():
    """Caught by regex — embedding guard is never called."""
    engine = make_engine()
    assert engine.check_topic("Cuéntame un chiste sobre Basilea III") is False


def test_check_topic_embedding_blocks_low_score():
    """Low embedding score rejects the question."""
    engine = make_engine()
    with patch("src.chat_engine._topic_embedding_score", return_value=0.12):
        assert engine.check_topic("¿Cuál es el precio del aceite de oliva?") is False


def test_check_topic_embedding_allows_high_score():
    """High embedding score passes the question."""
    engine = make_engine()
    with patch("src.chat_engine._topic_embedding_score", return_value=0.75):
        assert engine.check_topic("¿Cómo se calcula el ratio CET1?") is True


def test_check_topic_embedding_boundary_just_below():
    """Score exactly at threshold - ε must be rejected."""
    engine = make_engine()
    with patch("src.chat_engine._topic_embedding_score", return_value=0.299):
        assert engine.check_topic("alguna pregunta ambigua") is False


def test_check_topic_embedding_boundary_at_threshold():
    """Score exactly at threshold must be accepted."""
    engine = make_engine()
    with patch("src.chat_engine._topic_embedding_score", return_value=0.30):
        assert engine.check_topic("alguna pregunta borderline") is True


def test_check_topic_failopen_when_guard_unavailable():
    """If embedding guard raises, fail-open (return 1.0 → accept)."""
    engine = make_engine()
    with patch("src.chat_engine._topic_embedding_score", return_value=1.0):
        assert engine.check_topic("¿Qué es el NSFR?") is True


def test_rejection_card_content():
    assert "resp-card" in REJECTION_CARD
    assert "regulación bancaria" in REJECTION_CARD


# ─── build_context ────────────────────────────────────────────────────────────

def test_build_context_empty_results():
    engine = make_engine()
    # Both search methods return []
    engine.rag.buscar_hibrida.return_value = []
    engine.rag.buscar_contexto.return_value = []
    context, sources = engine.build_context("¿Cuánto es el LCR?", n_sources=5, hybrid=True)
    assert sources == []
    assert "No se encontraron" in context

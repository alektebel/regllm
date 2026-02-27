"""
Unit tests for app.py — ask() and ask_stream() query pipeline.

No GPU, no DB, no network required. ChatEngine and generation
functions are replaced with MagicMock / fake generators.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import app
from src.chat_engine import REJECTION_CARD


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_engine(on_topic: bool = True, confidence: int = 80, n_refs: int = 2):
    """Return a fully-mocked ChatEngine."""
    engine = MagicMock()
    engine.check_topic.return_value = on_topic
    engine.build_context.return_value = ("Mock regulatory context.", [])
    engine.build_messages.return_value = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]
    parsed = {
        "respuesta": "El ratio CET1 mínimo es 4.5%.",
        "referencias": [
            {"documento": "CRR", "articulo": "Art. 92", "paragrafo": "§1", "descripcion": "Capital"}
        ] * n_refs,
        "confianza": confidence,
        "justificacion_confianza": "Alta similitud regulatoria.",
    }
    engine.parse_response.return_value = parsed
    engine.enrich_references.return_value = parsed
    engine.render_message.return_value = "<div class='resp-card'>Mocked response</div>"
    return engine


def _passthrough_engine(on_topic: bool = True):
    """Engine whose parse/render return the raw string as content — useful for error tests."""
    engine = _make_engine(on_topic=on_topic)
    engine.parse_response.side_effect = lambda raw: {
        "respuesta": raw,
        "referencias": [],
        "confianza": None,
        "justificacion_confianza": "",
    }
    engine.enrich_references.side_effect = lambda parsed, q, **kw: parsed
    engine.render_message.side_effect = lambda parsed: parsed["respuesta"]
    return engine


def _assistant_messages(history: list) -> list[str]:
    return [m["content"] for m in history if isinstance(m, dict) and m["role"] == "assistant"]


# ─── ask() — edge cases ───────────────────────────────────────────────────────

def test_ask_empty_question_returns_early():
    txt, history = app.ask("", [])
    assert txt == ""
    assert history == []


def test_ask_whitespace_only_returns_early():
    txt, history = app.ask("   ", [])
    assert txt == ""
    assert history == []


def test_ask_engine_not_initialised():
    with patch.object(app, "engine", None):
        txt, history = app.ask("¿Qué es CET1?", [])
    assert txt != "" or history != []  # some error response
    assert "inicializado" in txt.lower()


# ─── ask() — topic guard ─────────────────────────────────────────────────────

def test_ask_off_topic_returns_rejection_card():
    with patch.object(app, "engine", _make_engine(on_topic=False)), \
         patch.object(app, "_backend", "groq"):
        txt, history = app.ask("Cuéntame un chiste", [])
    assert txt == ""
    assert REJECTION_CARD in _assistant_messages(history)


def test_ask_off_topic_does_not_call_generate():
    with patch.object(app, "engine", _make_engine(on_topic=False)), \
         patch.object(app, "_backend", "groq"), \
         patch.object(app, "_generate_groq") as mock_gen:
        app.ask("Hazme una poesía", [])
    mock_gen.assert_not_called()


# ─── ask() — successful generation ───────────────────────────────────────────

def test_ask_groq_returns_rendered_response():
    with patch.object(app, "engine", _make_engine()), \
         patch.object(app, "_backend", "groq"), \
         patch.object(app, "_generate_groq", return_value='{"respuesta": "Test."}'):
        txt, history = app.ask("¿Qué es CET1?", [])
    assert txt == ""
    assert any("resp-card" in m for m in _assistant_messages(history))


def test_ask_local_returns_rendered_response():
    with patch.object(app, "engine", _make_engine()), \
         patch.object(app, "_backend", "local"), \
         patch.object(app, "_generate_local", return_value='{"respuesta": "Test."}'):
        txt, history = app.ask("¿Qué es el LCR?", [])
    assert txt == ""
    assert any("resp-card" in m for m in _assistant_messages(history))


def test_ask_logs_interaction():
    mock_engine = _make_engine()
    with patch.object(app, "engine", mock_engine), \
         patch.object(app, "_backend", "groq"), \
         patch.object(app, "_generate_groq", return_value='{"respuesta": "Test."}'):
        app.ask("¿Cuál es el LCR mínimo?", [])
    mock_engine.log.assert_called_once()


def test_ask_passes_latency_to_log():
    mock_engine = _make_engine()
    with patch.object(app, "engine", mock_engine), \
         patch.object(app, "_backend", "groq"), \
         patch.object(app, "_generate_groq", return_value='{"respuesta": "Test."}'):
        app.ask("¿Qué es CET1?", [])
    _, kwargs = mock_engine.log.call_args
    assert kwargs.get("latency_ms") is not None
    assert kwargs["latency_ms"] >= 0


def test_ask_preserves_and_extends_history():
    history = [
        {"role": "user", "content": "Primera pregunta"},
        {"role": "assistant", "content": "Primera respuesta"},
    ]
    with patch.object(app, "engine", _make_engine()), \
         patch.object(app, "_backend", "groq"), \
         patch.object(app, "_generate_groq", return_value='{"respuesta": "Resp."}'):
        _, new_history = app.ask("Segunda pregunta", history)
    assert len(new_history) == len(history) + 2
    assert new_history[-2]["role"] == "user"
    assert new_history[-1]["role"] == "assistant"


# ─── ask() — generation error ─────────────────────────────────────────────────

def test_ask_generation_error_returns_error_message():
    with patch.object(app, "engine", _passthrough_engine()), \
         patch.object(app, "_backend", "groq"), \
         patch.object(app, "_generate_groq", side_effect=RuntimeError("API down")):
        txt, history = app.ask("¿Qué es CET1?", [])
    assert any("Error" in m for m in _assistant_messages(history))


def test_ask_generation_error_does_not_raise():
    with patch.object(app, "engine", _make_engine()), \
         patch.object(app, "_backend", "groq"), \
         patch.object(app, "_generate_groq", side_effect=RuntimeError("API down")):
        try:
            app.ask("¿Qué es CET1?", [])
        except Exception as e:
            pytest.fail(f"ask() raised unexpectedly: {e}")


# ─── ask_stream() — edge cases ────────────────────────────────────────────────

def test_ask_stream_empty_question_returns_early():
    results = list(app.ask_stream("", []))
    assert len(results) == 1
    txt, history = results[0]
    assert txt == ""
    assert history == []


def test_ask_stream_engine_not_initialised():
    with patch.object(app, "engine", None):
        results = list(app.ask_stream("¿Qué es CET1?", []))
    assert len(results) == 1


# ─── ask_stream() — topic guard ───────────────────────────────────────────────

def test_ask_stream_off_topic_returns_rejection_card():
    with patch.object(app, "engine", _make_engine(on_topic=False)), \
         patch.object(app, "_backend", "ollama"):
        results = list(app.ask_stream("Cuéntame un chiste", []))
    assert len(results) == 1
    _, history = results[0]
    assert REJECTION_CARD in _assistant_messages(history)


# ─── ask_stream() — streaming generation ─────────────────────────────────────

def test_ask_stream_yields_chunks_then_final():
    def _fake_stream(_msgs):
        yield "El ratio "
        yield "CET1 es "
        yield "4.5%."

    mock_engine = _make_engine()
    with patch.object(app, "engine", mock_engine), \
         patch.object(app, "_backend", "ollama"), \
         patch.object(app, "_generate_ollama_stream", new=_fake_stream):
        results = list(app.ask_stream("¿Qué es CET1?", []))

    # 3 streaming yields + 1 final rendered response
    assert len(results) >= 4
    mock_engine.log.assert_called_once()


def test_ask_stream_final_response_is_rendered():
    def _fake_stream(_msgs):
        yield "chunk"

    mock_engine = _make_engine()
    with patch.object(app, "engine", mock_engine), \
         patch.object(app, "_backend", "ollama"), \
         patch.object(app, "_generate_ollama_stream", new=_fake_stream):
        results = list(app.ask_stream("¿Qué es CET1?", []))

    # Last result must contain the rendered (not raw streamed) response
    _, final_history = results[-1]
    assert any("resp-card" in m for m in _assistant_messages(final_history))


def test_ask_stream_error_yields_error_message():
    def _fail(_msgs):
        raise RuntimeError("Ollama down")
        yield  # makes it a generator

    with patch.object(app, "engine", _passthrough_engine()), \
         patch.object(app, "_backend", "ollama"), \
         patch.object(app, "_generate_ollama_stream", new=_fail):
        results = list(app.ask_stream("¿Qué es CET1?", []))

    _, history = results[-1]
    assert any("Error" in m for m in _assistant_messages(history))


def test_ask_stream_error_does_not_raise():
    def _fail(_msgs):
        raise RuntimeError("Ollama down")
        yield

    with patch.object(app, "engine", _make_engine()), \
         patch.object(app, "_backend", "ollama"), \
         patch.object(app, "_generate_ollama_stream", new=_fail):
        try:
            list(app.ask_stream("¿Qué es CET1?", []))
        except Exception as e:
            pytest.fail(f"ask_stream() raised unexpectedly: {e}")

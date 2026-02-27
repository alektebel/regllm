"""
Unit tests for src/db.py — no live PostgreSQL required.

All DB I/O is intercepted via unittest.mock (AsyncMock / MagicMock).
Run: pytest tests/test_db_unit.py -v
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.db as db


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _reset_engine():
    """Reset singletons between tests that touch them."""
    db._engine = None
    db._session_factory = None


# ─── _build_url ───────────────────────────────────────────────────────────────

def test_build_url_defaults():
    with patch.dict("os.environ", {}, clear=True):
        url = db._build_url()
    assert url.startswith("postgresql+asyncpg://")
    assert "regllm" in url
    assert "localhost" in url


def test_build_url_uses_env_vars():
    env = {
        "POSTGRES_HOST": "myhost",
        "POSTGRES_PORT": "5433",
        "POSTGRES_DB": "mydb",
        "POSTGRES_USER": "myuser",
        "POSTGRES_PASSWORD": "s3cr3t",
    }
    with patch.dict("os.environ", env):
        url = db._build_url()
    assert "myhost:5433" in url
    assert "mydb" in url
    assert "myuser" in url
    assert "s3cr3t" in url


# ─── get_engine singleton ─────────────────────────────────────────────────────

def test_get_engine_returns_singleton():
    _reset_engine()
    with patch("src.db.create_async_engine", return_value=MagicMock()) as mock_create, \
         patch("src.db.sessionmaker", return_value=MagicMock()):
        e1 = db.get_engine()
        e2 = db.get_engine()
    assert e1 is e2
    assert mock_create.call_count == 1
    _reset_engine()


def test_get_engine_creates_session_factory():
    _reset_engine()
    mock_engine = MagicMock()
    with patch("src.db.create_async_engine", return_value=mock_engine), \
         patch("src.db.sessionmaker", return_value=MagicMock()) as mock_sf:
        db.get_engine()
    mock_sf.assert_called_once()
    _reset_engine()


# ─── log_qa_interaction ───────────────────────────────────────────────────────

def test_log_qa_interaction_calls_session_add_commit():
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("src.db._embed", return_value=[[0.1] * 384, [0.2] * 384]), \
         patch("src.db.get_session", return_value=mock_session):
        asyncio.run(db.log_qa_interaction(
            question="¿Qué es CET1?",
            model_answer="CET1 es capital de nivel 1.",
            category="regulatory",
            latency_ms=200,
        ))

    mock_session.add.assert_called_once()
    mock_session.commit.assert_awaited_once()


def test_log_qa_interaction_does_not_raise_on_db_error():
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.commit.side_effect = RuntimeError("DB down")

    with patch("src.db._embed", return_value=[[0.1] * 384, [0.2] * 384]), \
         patch("src.db.get_session", return_value=mock_session):
        try:
            asyncio.run(db.log_qa_interaction("q", "a"))
        except Exception as e:
            pytest.fail(f"log_qa_interaction raised: {e}")


def test_log_qa_interaction_does_not_raise_on_embed_error():
    with patch("src.db._embed", side_effect=RuntimeError("embed model unavailable")):
        try:
            asyncio.run(db.log_qa_interaction("q", "a"))
        except Exception as e:
            pytest.fail(f"log_qa_interaction raised on embed error: {e}")


# ─── store_feedback ───────────────────────────────────────────────────────────

def test_store_feedback_inserts_row():
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("src.db.get_session", return_value=mock_session):
        asyncio.run(db.store_feedback(
            rating="thumbs_up",
            question_text="¿Qué es LCR?",
        ))

    mock_session.add.assert_called_once()
    added = mock_session.add.call_args[0][0]
    assert added.rating == "thumbs_up"
    assert added.question_text == "¿Qué es LCR?"
    mock_session.commit.assert_awaited_once()


def test_store_feedback_thumbs_down():
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("src.db.get_session", return_value=mock_session):
        asyncio.run(db.store_feedback(rating="thumbs_down", question_text="bad q"))

    added = mock_session.add.call_args[0][0]
    assert added.rating == "thumbs_down"


def test_store_feedback_does_not_raise_on_error():
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.commit.side_effect = RuntimeError("DB down")

    with patch("src.db.get_session", return_value=mock_session):
        try:
            asyncio.run(db.store_feedback("thumbs_up"))
        except Exception as e:
            pytest.fail(f"store_feedback raised: {e}")


# ─── search_similar_qa ────────────────────────────────────────────────────────

def test_search_similar_qa_returns_list():
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_row = {"id": 1, "question": "q", "reference_answer": None,
                "model_answer": "a", "category": "reg", "source": None,
                "created_at": None, "distance": 0.05}
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = [mock_row]
    mock_session.execute = AsyncMock(return_value=mock_result)

    with patch("src.db._embed", return_value=[[0.1] * 384]), \
         patch("src.db.get_session", return_value=mock_session):
        results = asyncio.run(db.search_similar_qa("¿Qué es CET1?", top_k=1))

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["question"] == "q"


def test_search_similar_qa_returns_empty_on_error():
    with patch("src.db._embed", side_effect=RuntimeError("embed fail")):
        results = asyncio.run(db.search_similar_qa("cualquier pregunta"))
    assert results == []


# ─── get_db_stats ─────────────────────────────────────────────────────────────

def test_get_db_stats_returns_dict_with_expected_keys():
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_row = {
        "total_queries": 42,
        "avg_confianza": 0.75,
        "avg_latencia_ms": 320.5,
        "queries_hoy": 10,
    }
    mock_result = MagicMock()
    mock_result.mappings.return_value.one.return_value = mock_row
    mock_session.execute = AsyncMock(return_value=mock_result)

    with patch("src.db.get_session", return_value=mock_session):
        stats = asyncio.run(db.get_db_stats())

    assert stats["total_queries"] == 42
    assert stats["avg_confianza"] == 0.75
    assert stats["queries_hoy"] == 10


def test_get_db_stats_returns_zeros_on_error():
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute = AsyncMock(side_effect=RuntimeError("DB error"))

    with patch("src.db.get_session", return_value=mock_session):
        stats = asyncio.run(db.get_db_stats())

    assert stats["total_queries"] == 0
    assert stats["avg_confianza"] == 0.0


# ─── run_db_sync / log_async ──────────────────────────────────────────────────

def test_run_db_sync_executes_coroutine():
    async def _coro():
        return 42

    result = db.run_db_sync(_coro())
    assert result == 42


def test_log_async_does_not_block():
    """log_async() must return immediately without raising."""
    completed = []

    async def _coro():
        completed.append(True)

    db.log_async(_coro())
    # Give the background loop a tick to execute
    import time
    time.sleep(0.05)
    assert completed == [True]

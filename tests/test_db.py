"""
DB integration tests — qa_interactions table + vector functions.

Requires a live PostgreSQL + pgvector instance (uses .env credentials).
All tests are tagged @pytest.mark.integration.

Run with:
  pytest tests/test_db.py -v
  pytest tests/test_db.py -v -m "not slow"   # skip concurrent-write stress test

Online-deployment scenarios covered:
  - Idempotent init (safe to call on every startup)
  - Round-trip insert + verify (basic correctness)
  - Embedding consistency (same text → cosine distance ≈ 0)
  - Semantic search ranking (relevant result ranked above irrelevant)
  - Concurrent writes (N simultaneous inserts don't corrupt the pool)
  - Large payload (10 KB+ question/answer)
  - Null / optional fields (reference_answer, category, latency_ms)
  - Graceful degradation on bad connection (no exception escapes)
  - SQL-injection safety via parameterized queries
  - top_k boundary (k > available rows returns what exists)
"""

import asyncio
import sys
import uuid
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def unique(prefix: str = "test") -> str:
    """Return a unique string to avoid cross-test collisions."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


async def _count_rows(question_prefix: str) -> int:
    """Count qa_interactions rows whose question starts with prefix."""
    from sqlalchemy import text
    from src.db import get_session
    async with get_session() as session:
        result = await session.execute(
            text("SELECT COUNT(*) FROM qa_interactions WHERE question LIKE :p")
            .bindparams(p=f"{question_prefix}%")
        )
        return result.scalar()


async def _delete_rows(question_prefix: str) -> None:
    """Clean up test rows after each test."""
    from sqlalchemy import text
    from src.db import get_session
    async with get_session() as session:
        await session.execute(
            text("DELETE FROM qa_interactions WHERE question LIKE :p")
            .bindparams(p=f"{question_prefix}%")
        )
        await session.commit()


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_engine():
    """
    Dispose the SQLAlchemy async engine BEFORE every test.

    asyncpg connections are tied to the event loop that created them.
    Each asyncio.run() call creates a new event loop, so the singleton engine
    from a previous test would hold stale connections for the new loop.
    Disposing forces a fresh engine on the first DB call inside the test.

    We skip the teardown dispose — closing connections after asyncio.run()
    has already shut down the loop produces benign RuntimeError noise from
    asyncpg's internal transport cleanup.
    """
    from src.db import dispose_engine
    asyncio.run(dispose_engine())
    yield
    # No teardown dispose: engine cleanup happens in the next test's setup.


# ─── 1. Idempotent init ───────────────────────────────────────────────────────

@pytest.mark.integration
def test_init_db_idempotent():
    """Calling init_db() multiple times must not raise or duplicate structure."""
    from src.db import init_db

    async def run():
        await init_db()
        await init_db()   # second call — idempotent (CREATE IF NOT EXISTS)

    asyncio.run(run())


# ─── 2. Basic round-trip insert ───────────────────────────────────────────────

@pytest.mark.integration
def test_log_and_count():
    """log_qa_interaction inserts exactly one row."""
    prefix = unique("rt")

    async def run():
        from src.db import log_qa_interaction
        await log_qa_interaction(
            question=f"{prefix} ¿Qué es el CET1?",
            model_answer="El CET1 es el ratio de capital de máxima calidad.",
            reference_answer="Capital de nivel 1 ordinario.",
            category="regulatory",
            source="test_suite",
            latency_ms=123,
            metadata={"run": "ci"},
        )
        count = await _count_rows(prefix)
        await _delete_rows(prefix)
        return count

    assert asyncio.run(run()) == 1


# ─── 3. Optional fields (NULLs) ───────────────────────────────────────────────

@pytest.mark.integration
def test_log_minimal_fields():
    """Only question + model_answer are required; all others can be None."""
    prefix = unique("min")

    async def run():
        from src.db import log_qa_interaction
        await log_qa_interaction(
            question=f"{prefix} minimal",
            model_answer="respuesta mínima",
        )
        count = await _count_rows(prefix)
        await _delete_rows(prefix)
        return count

    assert asyncio.run(run()) == 1


# ─── 4. Embedding consistency ─────────────────────────────────────────────────

@pytest.mark.integration
def test_embedding_consistency():
    """The same text encoded twice must produce cosine distance ≈ 0."""
    from src.db import _embed
    v1 = _embed(["regulación bancaria española"])
    v2 = _embed(["regulación bancaria española"])
    import numpy as np
    a, b = np.array(v1[0]), np.array(v2[0])
    cosine_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cosine_sim > 0.9999, f"Same text gave cosine_sim={cosine_sim:.6f}"


# ─── 5. Embedding dimensionality ──────────────────────────────────────────────

@pytest.mark.integration
def test_embedding_dimension():
    """Embeddings must be exactly 384-dimensional (model contract)."""
    from src.db import _embed
    vecs = _embed(["test dimensión", "otro texto"])
    assert len(vecs) == 2
    for v in vecs:
        assert len(v) == 384, f"Expected 384 dims, got {len(v)}"


# ─── 6. Semantic search ranking ───────────────────────────────────────────────

@pytest.mark.integration
def test_search_similar_ranking():
    """
    Insert a regulatory question and an unrelated one.
    A regulatory query must rank the regulatory row first.
    """
    prefix = unique("rank")

    async def run():
        from src.db import log_qa_interaction, search_similar_qa
        q_reg = f"{prefix} ¿Qué establece el artículo 92 del CRR sobre requisitos de capital?"
        q_irr = f"{prefix} ¿Cuál es la receta de la paella valenciana?"

        await log_qa_interaction(q_reg, "El artículo 92 exige un CET1 del 4,5%.", category="regulatory", source="test_suite")
        await log_qa_interaction(q_irr, "Arroz, pollo, conejo, judía verde.", category="food", source="test_suite")

        results = await search_similar_qa("requisitos de capital CRR artículo 92", top_k=5)
        await _delete_rows(prefix)
        return results

    results = asyncio.run(run())
    assert len(results) >= 2
    # The regulatory question should appear before the irrelevant one
    questions = [r["question"] for r in results]
    reg_idx = next(i for i, q in enumerate(questions) if "artículo 92" in q)
    irr_idx = next(i for i, q in enumerate(questions) if "paella" in q)
    assert reg_idx < irr_idx, (
        f"Regulatory result ranked at {reg_idx}, irrelevant at {irr_idx} — expected reg first"
    )


# ─── 7. top_k boundary ────────────────────────────────────────────────────────

@pytest.mark.integration
def test_search_top_k_larger_than_available():
    """search_similar_qa with top_k > row count returns all available rows."""
    prefix = unique("topk")

    async def run():
        from src.db import log_qa_interaction, search_similar_qa
        await log_qa_interaction(f"{prefix} solo una pregunta", "una respuesta", source="test_suite")
        results = await search_similar_qa(f"{prefix}", top_k=1000)
        await _delete_rows(prefix)
        return results

    results = asyncio.run(run())
    assert isinstance(results, list)
    assert len(results) >= 1


# ─── 8. Large payload ─────────────────────────────────────────────────────────

@pytest.mark.integration
def test_large_payload():
    """Insert a 10 KB question and answer without error."""
    prefix = unique("large")
    big_question = f"{prefix} " + ("regulación bancaria CRR Basilea IRB " * 300)
    big_answer = "Esta es una respuesta extensa. " * 400

    async def run():
        from src.db import log_qa_interaction
        await log_qa_interaction(
            question=big_question,
            model_answer=big_answer,
            category="regulatory",
            source="test_suite",
        )
        count = await _count_rows(prefix)
        await _delete_rows(prefix)
        return count

    assert asyncio.run(run()) == 1


# ─── 9. Concurrent writes ─────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.slow
def test_concurrent_writes():
    """
    50 simultaneous inserts must all succeed (deployment: multiple users at once).
    Validates connection-pool integrity under load.
    """
    prefix = unique("conc")
    N = 50

    async def run():
        from src.db import log_qa_interaction
        tasks = [
            log_qa_interaction(
                question=f"{prefix} pregunta concurrente {i}",
                model_answer=f"respuesta {i}",
                category="live",
                source="test_suite",
                latency_ms=100 + i,
            )
            for i in range(N)
        ]
        await asyncio.gather(*tasks)
        count = await _count_rows(prefix)
        await _delete_rows(prefix)
        return count

    count = asyncio.run(run())
    assert count == N, f"Expected {N} rows, got {count}"


# ─── 10. Graceful degradation (bad credentials) ───────────────────────────────

@pytest.mark.integration
def test_graceful_degradation_bad_connection():
    """
    log_qa_interaction must not raise even when the DB is unreachable.
    (Simulated by a bad password — the function swallows the error.)
    The fresh_engine fixture already disposed the engine before this test,
    so we just swap the env var, call the function, and it will fail silently.
    """
    import os
    original_pw = os.environ.get("POSTGRES_PASSWORD", "changeme")
    os.environ["POSTGRES_PASSWORD"] = "WRONG_PASSWORD_XYZ"

    async def run():
        from src.db import log_qa_interaction
        # Must not raise — errors are caught and logged internally
        await log_qa_interaction("question with bad creds", "answer")

    try:
        asyncio.run(run())   # should complete without exception
    finally:
        os.environ["POSTGRES_PASSWORD"] = original_pw


# ─── 11. SQL-injection safety ─────────────────────────────────────────────────

@pytest.mark.integration
def test_sql_injection_safety():
    """
    Malicious input in the question field must be stored literally,
    not interpreted as SQL.
    """
    prefix = unique("sqli")
    evil = f"{prefix}'; DROP TABLE qa_interactions; --"

    async def run():
        from src.db import log_qa_interaction, search_similar_qa
        await log_qa_interaction(evil, "respuesta segura", source="test_suite")

        # Table must still exist and be queryable
        results = await search_similar_qa(evil, top_k=5)
        await _delete_rows(prefix)
        return results

    results = asyncio.run(run())
    # If we get here the table was not dropped
    assert isinstance(results, list)


# ─── 12. Result shape ─────────────────────────────────────────────────────────

@pytest.mark.integration
def test_search_result_schema():
    """Each result dict must contain the expected keys for the API/frontend."""
    prefix = unique("schema")

    async def run():
        from src.db import log_qa_interaction, search_similar_qa
        await log_qa_interaction(
            f"{prefix} CET1 ratio capital",
            "El ratio CET1 mínimo es 4,5%.",
            category="regulatory",
            source="test_suite",
        )
        results = await search_similar_qa(f"{prefix} CET1", top_k=3)
        await _delete_rows(prefix)
        return results

    results = asyncio.run(run())
    assert len(results) >= 1
    required_keys = {"id", "question", "model_answer", "category", "source", "created_at", "distance"}
    for row in results:
        missing = required_keys - set(row.keys())
        assert not missing, f"Result missing keys: {missing}"


# ─── 13. Distance ordering ────────────────────────────────────────────────────

@pytest.mark.integration
def test_search_distance_ordering():
    """Results must be sorted ascending by cosine distance (closest first)."""
    prefix = unique("dist")

    async def run():
        from src.db import log_qa_interaction, search_similar_qa
        await log_qa_interaction(f"{prefix} ¿Qué es el IRB avanzado?", "El IRB avanzado usa PD/LGD propios.", source="test_suite")
        await log_qa_interaction(f"{prefix} temperatura del sol", "Unos 5500 grados Celsius.", source="test_suite")
        results = await search_similar_qa("IRB avanzado LGD", top_k=10)
        await _delete_rows(prefix)
        return results

    results = asyncio.run(run())
    if len(results) >= 2:
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances), f"Results not sorted by distance: {distances}"


# ─── 14. Metadata JSONB round-trip ────────────────────────────────────────────

@pytest.mark.integration
def test_metadata_jsonb_roundtrip():
    """Arbitrary metadata dict must survive a write/read cycle intact."""
    prefix = unique("meta")
    payload = {"run_id": "abc123", "score": 0.87, "tags": ["regulatory", "CRR"], "nested": {"k": 1}}

    async def run():
        from src.db import log_qa_interaction
        from sqlalchemy import text
        from src.db import get_session
        await log_qa_interaction(
            f"{prefix} metadata test",
            "respuesta",
            metadata=payload,
            source="test_suite",
        )
        async with get_session() as session:
            result = await session.execute(
                text("SELECT metadata FROM qa_interactions WHERE question LIKE :p LIMIT 1")
                .bindparams(p=f"{prefix}%")
            )
            row = result.mappings().one()
        await _delete_rows(prefix)
        return row["metadata"]

    stored = asyncio.run(run())
    assert stored == payload

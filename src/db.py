"""
Async database layer for query logging.
Uses SQLAlchemy 2.0 async + asyncpg.
"""

import asyncio
import os
import logging
import threading
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, Integer, Text, Float, String, DateTime, JSON,
    func, text,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from pgvector.sqlalchemy import Vector

logger = logging.getLogger(__name__)

# Singleton engine
_engine = None
_session_factory = None


class Base(DeclarativeBase):
    pass


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pregunta = Column(Text, nullable=False)
    respuesta = Column(Text, nullable=False)
    fuentes = Column(JSON)
    score_confianza = Column(Float)
    nivel_confianza = Column(String(20))
    advertencias = Column(JSON)
    modelo = Column(String(200))
    latencia_ms = Column(Integer)
    ip_cliente = Column(String(45))
    created_at = Column(DateTime, server_default=text("now()"))


class QAInteraction(Base):
    __tablename__ = "qa_interactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    reference_answer = Column(Text, nullable=True)
    model_answer = Column(Text, nullable=False)
    question_embedding = Column(Vector(384), nullable=True)
    answer_embedding = Column(Vector(384), nullable=True)
    category = Column(String(100), nullable=True)
    source = Column(String(200), nullable=True)
    latency_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=text("now()"))
    metadata_ = Column("metadata", JSON, nullable=True)


def _build_url() -> str:
    """Build PostgreSQL async connection URL from env vars."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "regllm")
    user = os.getenv("POSTGRES_USER", "regllm")
    password = os.getenv("POSTGRES_PASSWORD", "changeme")
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


def get_engine():
    """Get or create the async engine (singleton)."""
    global _engine, _session_factory
    if _engine is None:
        url = _build_url()
        _engine = create_async_engine(url, echo=False, pool_size=5, max_overflow=10)
        _session_factory = sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
        logger.info("Database engine created")
    return _engine


def get_session() -> AsyncSession:
    """Get a new async session."""
    if _session_factory is None:
        get_engine()
    return _session_factory()


async def init_db():
    """Create tables if they don't exist, and enable pgvector extension."""
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            # Enable pgvector extension (idempotent)
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
            # HNSW index for fast ANN search — works well at any table size
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS qa_interactions_question_emb_idx "
                "ON qa_interactions USING hnsw (question_embedding vector_cosine_ops)"
            ))
        logger.info("Database tables initialized")
    except Exception as e:
        logger.warning(f"Could not initialize database: {e}. Logging will be disabled.")


async def dispose_engine():
    """Dispose the engine on shutdown."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database engine disposed")


async def log_query(
    pregunta: str,
    respuesta: str,
    fuentes: list,
    score_confianza: float,
    nivel_confianza: str,
    advertencias: list,
    modelo: str,
    latencia_ms: int,
    ip_cliente: Optional[str] = None,
):
    """Insert a query log row. Called from BackgroundTasks."""
    try:
        async with get_session() as session:
            entry = QueryLog(
                pregunta=pregunta,
                respuesta=respuesta,
                fuentes=fuentes,
                score_confianza=score_confianza,
                nivel_confianza=nivel_confianza,
                advertencias=advertencias,
                modelo=modelo,
                latencia_ms=latencia_ms,
                ip_cliente=ip_cliente,
            )
            session.add(entry)
            await session.commit()
    except Exception as e:
        logger.error(f"Failed to log query: {e}")


async def get_query_logs(limit: int = 50, offset: int = 0) -> list[dict]:
    """Get paginated query logs, newest first."""
    try:
        async with get_session() as session:
            result = await session.execute(
                text(
                    "SELECT id, pregunta, respuesta, fuentes, score_confianza, "
                    "nivel_confianza, advertencias, modelo, latencia_ms, ip_cliente, created_at "
                    "FROM query_logs ORDER BY id DESC LIMIT :limit OFFSET :offset"
                ).bindparams(limit=limit, offset=offset)
            )
            rows = result.mappings().all()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Failed to fetch query logs: {e}")
        return []


async def get_db_stats() -> dict:
    """Aggregate stats: total queries, avg confidence, avg latency, queries per day."""
    try:
        async with get_session() as session:
            result = await session.execute(
                text(
                    "SELECT "
                    "  COUNT(*) AS total_queries, "
                    "  COALESCE(AVG(score_confianza), 0) AS avg_confianza, "
                    "  COALESCE(AVG(latencia_ms), 0) AS avg_latencia_ms, "
                    "  COALESCE(COUNT(*) FILTER (WHERE created_at >= CURRENT_DATE), 0) AS queries_hoy "
                    "FROM query_logs"
                )
            )
            row = result.mappings().one()
            return {
                "total_queries": row["total_queries"],
                "avg_confianza": round(float(row["avg_confianza"]), 4),
                "avg_latencia_ms": round(float(row["avg_latencia_ms"]), 1),
                "queries_hoy": row["queries_hoy"],
            }
    except Exception as e:
        logger.error(f"Failed to fetch DB stats: {e}")
        return {
            "total_queries": 0,
            "avg_confianza": 0.0,
            "avg_latencia_ms": 0.0,
            "queries_hoy": 0,
        }


# ─── Embedding helper (lazy-loaded) ───────────────────────────────────────────

_embed_model = None
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _embed(texts: list[str]) -> list[list[float]]:
    model = _get_embed_model()
    return model.encode(texts, convert_to_numpy=True).tolist()


# ─── QA Interaction logging ────────────────────────────────────────────────────

async def log_qa_interaction(
    question: str,
    model_answer: str,
    reference_answer: Optional[str] = None,
    category: Optional[str] = None,
    source: Optional[str] = None,
    latency_ms: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Insert a QA interaction row with vector embeddings. Fire-and-forget safe."""
    try:
        vecs = _embed([question, model_answer])
        q_emb, a_emb = vecs[0], vecs[1]

        async with get_session() as session:
            entry = QAInteraction(
                question=question,
                reference_answer=reference_answer,
                model_answer=model_answer,
                question_embedding=q_emb,
                answer_embedding=a_emb,
                category=category,
                source=source,
                latency_ms=latency_ms,
                metadata_=metadata,
            )
            session.add(entry)
            await session.commit()
    except Exception as e:
        logger.error(f"Failed to log QA interaction: {e}")


# ─── Background event loop (for sync callers like Gradio handlers) ─────────────

_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_thread: threading.Thread | None = None
_bg_lock = threading.Lock()


def _get_bg_loop() -> asyncio.AbstractEventLoop:
    """Return the shared background event loop, starting it if needed."""
    global _bg_loop, _bg_thread
    with _bg_lock:
        if _bg_loop is None or not _bg_loop.is_running():
            _bg_loop = asyncio.new_event_loop()
            _bg_thread = threading.Thread(
                target=_bg_loop.run_forever,
                daemon=True,
                name="regllm-db-loop",
            )
            _bg_thread.start()
    return _bg_loop


def run_db_sync(coro, timeout: float = 30):
    """Run a DB coroutine synchronously from any (non-async) thread."""
    loop = _get_bg_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


def log_async(coro) -> None:
    """Fire-and-forget: submit a DB coroutine without blocking the caller."""
    loop = _get_bg_loop()
    asyncio.run_coroutine_threadsafe(coro, loop)


async def search_similar_qa(query_text: str, top_k: int = 5) -> list[dict]:
    """Return top_k most similar QA interactions by cosine distance on question_embedding."""
    try:
        vecs = _embed([query_text])
        q_emb = vecs[0]
        # pgvector <=> is cosine distance (lower = more similar)
        async with get_session() as session:
            result = await session.execute(
                text(
                    "SELECT id, question, reference_answer, model_answer, category, source, "
                    "       created_at, "
                    "       (question_embedding <=> CAST(:emb AS vector)) AS distance "
                    "FROM qa_interactions "
                    "WHERE question_embedding IS NOT NULL "
                    "ORDER BY distance ASC "
                    "LIMIT :top_k"
                ).bindparams(emb=str(q_emb), top_k=top_k)
            )
            rows = result.mappings().all()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Failed to search similar QA: {e}")
        return []

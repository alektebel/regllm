"""
Async database layer for query logging.
Uses SQLAlchemy 2.0 async + asyncpg.
"""

import os
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, Integer, Text, Float, String, DateTime, JSON,
    func, text,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase

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
    """Create tables if they don't exist."""
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
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

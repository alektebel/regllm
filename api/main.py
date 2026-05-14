"""
FastAPI application entry point.

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import dispose_engine, init_db

logger = logging.getLogger(__name__)

# ─── Globals ──────────────────────────────────────────────────────────────────

_chat_engine = None


def get_engine_instance():
    return _chat_engine


# ─── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chat_engine

    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        force=True,
    )

    # Init DB tables
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"DB init failed (logging disabled): {e}")

    # Init RAG + ChatEngine
    backend = os.getenv("REGLLM_BACKEND", "groq")
    try:
        from src.rag_system import RegulatoryRAGSystem
        from src.citation_rag import CitationRAG
        from src.chat_engine import ChatEngine

        rag = RegulatoryRAGSystem()
        logger.info(f"RAG ready — {rag.collection.count()} documents")

        try:
            citation_rag = CitationRAG()
        except Exception as e:
            logger.warning(f"Citation RAG unavailable: {e}")
            citation_rag = None

        _chat_engine = ChatEngine(rag, citation_rag=citation_rag, db_source=f"api_{backend}")
        logger.info("ChatEngine ready")
    except Exception as e:
        logger.error(f"ChatEngine init failed: {e}")

    yield

    await dispose_engine()
    logger.info("Shutdown complete")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RegLLM API",
    description="Spanish banking regulation LLM with hybrid RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routers import auth, conversations, chat  # noqa: E402

app.include_router(auth.router)
app.include_router(conversations.router)
app.include_router(chat.router)


@app.get("/health")
async def health():
    return {"status": "ok", "engine": _chat_engine is not None}

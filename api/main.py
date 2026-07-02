"""FastAPI entry point — DQC generator API.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on sys.path so ``import src...`` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)

app = FastAPI(
    title="RegLLM — DQC Generator",
    description="Data Quality Check generator for banking regulatory compliance (COREP/FINREP).",
    version="0.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:4200").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routers import dqc  # noqa: E402

app.include_router(dqc.router)


@app.get("/health")
def health() -> dict:
    from src.knowledge import get_client
    backend = get_client().detect_backend()
    return {"status": "ok", "llm_backend": backend}

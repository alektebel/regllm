"""
Application-level response cache for RegLLM.

This is NOT the transformer KV (key-value attention) cache — that lives inside
the model and speeds up token generation. This is a semantic deduplication
layer one level above: if a new question is near-identical to one already
answered (cosine distance < CACHE_THRESHOLD), we skip the LLM call entirely
and return the stored answer.

Storage: reuses the existing `qa_interactions` table in PostgreSQL.
Embeddings: same `paraphrase-multilingual-MiniLM-L12-v2` model already
loaded by db.py and rag_system.py — no extra memory.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Questions with cosine distance below this are treated as duplicates.
# 0.08 ≈ "essentially the same question, different words".
# Raise to 0.15 to be more aggressive; lower to 0.04 for exact-only.
CACHE_THRESHOLD = float(os.getenv("QUERY_CACHE_THRESHOLD", "0.08"))
CACHE_ENABLED = os.getenv("QUERY_CACHE_ENABLED", "true").lower() != "false"


async def get_cached_answer(question: str) -> Optional[dict]:
    """
    Look up a semantically similar question in qa_interactions.
    Returns {"answer": str, "sources": list, "cached": True} or None.
    """
    if not CACHE_ENABLED:
        return None

    try:
        from src.db import search_similar_qa

        results = await search_similar_qa(question, top_k=1)
        if not results:
            return None

        best = results[0]
        distance = best.get("distance", 1.0)

        if distance <= CACHE_THRESHOLD:
            logger.info(
                f"Cache HIT (distance={distance:.4f} ≤ {CACHE_THRESHOLD}): "
                f"'{question[:60]}…' → reusing answer from id={best['id']}"
            )
            return {
                "answer": best["model_answer"],
                "sources": [],
                "cached": True,
                "cache_distance": round(distance, 4),
            }

        logger.debug(f"Cache MISS (distance={distance:.4f} > {CACHE_THRESHOLD})")
        return None

    except Exception as e:
        logger.warning(f"Cache lookup failed (continuing without cache): {e}")
        return None


async def store_in_cache(
    question: str,
    answer: str,
    latency_ms: Optional[int] = None,
) -> None:
    """
    Persist a new Q&A pair into qa_interactions so future similar questions
    can hit the cache. Fire-and-forget safe.
    """
    if not CACHE_ENABLED:
        return
    try:
        from src.db import log_qa_interaction

        await log_qa_interaction(
            question=question,
            model_answer=answer,
            category="cached",
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.warning(f"Failed to store in cache: {e}")

"""
SSE streaming chat endpoint.

POST /chat/stream
  → yields: data: {"type":"token","delta":"..."}
            data: {"type":"sources","sources":[...]}
            data: {"type":"done","conversation_id":42}
            data: [DONE]
"""

import asyncio
import json
import logging
import os
import time
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user, get_db
from api.models import ChatRequest
from src.db import Conversation, ConversationMessage, User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ─── Per-backend streaming generators ─────────────────────────────────────────


async def _stream_vllm(messages: list) -> AsyncGenerator[str, None]:
    """Yield tokens from vLLM / HF Inference Endpoint (OpenAI-compatible API)."""
    import httpx

    host = os.getenv("VLLM_HOST", "http://localhost:8080")
    model = os.getenv("VLLM_MODEL", "tgi")  # HF endpoints use "tgi" as model name
    api_key = os.getenv("HF_TOKEN", "")

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST", f"{host}/v1/chat/completions", json=payload, headers=headers
        ) as response:
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError):
                    continue


# ─── Endpoint ─────────────────────────────────────────────────────────────────


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Stream a response as SSE. Creates/updates conversation in DB."""

    from src.chat_engine import ChatEngine
    from api.main import get_engine_instance  # lazy import to avoid circular

    chat_engine: Optional[ChatEngine] = get_engine_instance()
    if chat_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat engine not initialized",
        )

    # ── Topic guard ───────────────────────────────────────────────────────────
    if not chat_engine.check_topic(body.question):
        async def _reject():
            yield _sse({"type": "token", "delta": (
                "Lo siento, solo puedo responder preguntas sobre regulación bancaria "
                "(EBA, CRR/CRD IV, Basilea III/IV). Por favor, reformule su pregunta."
            )})
            yield _sse({"type": "done", "conversation_id": body.conversation_id})
            yield "data: [DONE]\n\n"
        return StreamingResponse(_reject(), media_type="text/event-stream")

    # ── Query cache check ─────────────────────────────────────────────────────
    # NOTE: this is an application-level response cache (semantic deduplication),
    # not the transformer KV-attention cache. If the same question (or a very
    # close paraphrase) was already answered, stream the stored answer directly.
    from src.cache import get_cached_answer

    cached = await get_cached_answer(body.question)
    if cached:
        async def _stream_cached():
            yield _sse({"type": "sources", "sources": []})
            yield _sse({"type": "token", "delta": cached["answer"]})
            yield _sse({"type": "cache_hit", "distance": cached["cache_distance"]})
            yield _sse({"type": "done", "conversation_id": body.conversation_id})
            yield "data: [DONE]\n\n"
        return StreamingResponse(_stream_cached(), media_type="text/event-stream")

    # ── RAG retrieval (sync, run in executor) ─────────────────────────────────
    loop = asyncio.get_event_loop()
    context, sources = await loop.run_in_executor(
        None,
        lambda: chat_engine.build_context(body.question, n_sources=5, hybrid=True),
    )

    # ── Build history from DB ─────────────────────────────────────────────────
    history = []
    if body.conversation_id:
        result = await db.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == body.conversation_id)
            .order_by(ConversationMessage.created_at)
        )
        for msg in result.scalars().all():
            history.append({"role": msg.role, "content": msg.content})

    messages = await loop.run_in_executor(
        None,
        lambda: chat_engine.build_messages(body.question, context, history),
    )

    # ── Ensure conversation exists ────────────────────────────────────────────
    if body.conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == body.conversation_id,
                Conversation.user_id == current_user.id,
            )
        )
        conv = result.scalar_one_or_none()
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # Auto-create a new conversation with a title derived from the question
        title = body.question[:80] + ("…" if len(body.question) > 80 else "")
        conv = Conversation(
            user_id=current_user.id,
            title=title,
            backend=body.backend,
        )
        db.add(conv)
        await db.commit()
        await db.refresh(conv)

    # Save user message
    user_msg = ConversationMessage(
        conversation_id=conv.id,
        role="user",
        content=body.question,
    )
    db.add(user_msg)
    await db.commit()

    # ── SSE generator ─────────────────────────────────────────────────────────
    async def _generate():
        full_response = []
        sources_sent = False

        # Send RAG sources first
        source_list = [
            {
                "source": r.get("metadata", {}).get("source", "?"),
                "text": r.get("texto", r.get("text", r.get("document", "")))[:400],
            }
            for r in sources
        ]
        yield _sse({"type": "sources", "sources": source_list})

        try:
            gen = _stream_vllm(messages)

            async for token in gen:
                full_response.append(token)
                yield _sse({"type": "token", "delta": token})

        except Exception as exc:
            logger.error(f"Streaming error: {exc}")
            yield _sse({"type": "error", "message": str(exc)})

        # Save assistant message to DB + populate cache
        full_text = "".join(full_response)
        try:
            async with db.begin_nested():
                assistant_msg = ConversationMessage(
                    conversation_id=conv.id,
                    role="assistant",
                    content=full_text,
                    sources=source_list,
                )
                db.add(assistant_msg)
            await db.commit()
        except Exception as exc:
            logger.error(f"Failed to save assistant message: {exc}")

        # Store in semantic cache for future near-duplicate questions
        from src.cache import store_in_cache
        await store_in_cache(body.question, full_text)

        yield _sse({"type": "done", "conversation_id": conv.id})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

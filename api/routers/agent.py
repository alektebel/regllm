"""Agentic Q&A router.

Endpoints
---------
POST /agent/ask                    SSE stream of agent events for one question
POST /agent/sas/upload             Upload .sas files into data/sas/{v2|v3}/
POST /agent/docs/reindex           Rebuild the BM25 docs index
GET  /agent/status                 Counts + active LLM backend/model
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SAS_ROOT = _PROJECT_ROOT / "data" / "sas"
_DOCS_ROOT = _PROJECT_ROOT / "data" / "docs"
_INDEX_PATH = _DOCS_ROOT / "index.json"


# ── Request/response models ──────────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    max_iters: int = 8
    temperature: float = 0.1


# ── /agent/ask  (SSE) ────────────────────────────────────────────────────────


def _sse_event(event: dict) -> bytes:
    """Format one SSE message. Single-line ``data:`` is the simplest path."""
    return ("data: " + json.dumps(event, default=str) + "\n\n").encode("utf-8")


async def _stream(question: str, *, max_iters: int, temperature: float) -> AsyncIterator[bytes]:
    from src.agent import SASDiffAgent

    agent = SASDiffAgent(max_iters=max_iters, temperature=temperature)
    try:
        async for ev in agent.run(question):
            yield _sse_event(ev.to_dict())
    except Exception as e:
        logger.exception("Agent run failed")
        yield _sse_event({"type": "error", "stage": "run", "error": str(e)})
    yield _sse_event({"type": "done"})


@router.post("/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream(req.question, max_iters=req.max_iters, temperature=req.temperature),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── /agent/sas/upload ────────────────────────────────────────────────────────


@router.post("/sas/upload")
async def upload_sas(
    files: list[UploadFile] = File(...),
    version: str = Form(...),
) -> dict:
    if version not in {"v2", "v3"}:
        raise HTTPException(400, "version must be 'v2' or 'v3'")
    target_dir = _SAS_ROOT / version
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in {".sas", ".egp"}:
            raise HTTPException(400, f"Only .sas and .egp files allowed (got {ext!r})")
        content = await f.read()
        if ext == ".egp":
            # Extract embedded SAS blocks and save as a single .sas file
            import tempfile
            from src.sas_parser import SASParser
            with tempfile.NamedTemporaryFile(suffix=".egp", delete=False) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)
            try:
                blocks = SASParser().parse(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)
            if not blocks:
                raise HTTPException(422, f"{f.filename}: no SAS code found in .egp")
            sas_code = "\n\n".join(b.code for b in blocks)
            out_name = Path(f.filename or "project").stem + ".sas"
            target = target_dir / out_name
            target.write_text(sas_code, encoding="utf-8")
        else:
            target = target_dir / Path(f.filename or "").name
            target.write_bytes(content)
        saved.append(target.name)
    return {"version": version, "saved": saved, "folder": str(target_dir)}


@router.delete("/sas/{version}/{name}")
def delete_sas(version: str, name: str) -> dict:
    if version not in {"v2", "v3"}:
        raise HTTPException(400, "version must be 'v2' or 'v3'")
    target = _SAS_ROOT / version / name
    if not target.exists():
        raise HTTPException(404, f"{target} not found")
    target.unlink()
    return {"deleted": str(target)}


# ── /agent/docs/reindex ──────────────────────────────────────────────────────


@router.post("/docs/reindex")
def reindex_docs() -> dict:
    from src.agent.docs_index import DocsIndex, reset_default_index

    idx = DocsIndex().build()
    idx.save(_INDEX_PATH)
    reset_default_index()
    return {
        "sections": idx.section_count(),
        "docs_root": str(idx.docs_root),
        "index_path": str(_INDEX_PATH),
    }


@router.post("/docs/upload")
async def upload_docs(
    files: list[UploadFile] = File(...),
    subfolder: str = Form("uploads"),
) -> dict:
    safe = Path(subfolder).name or "uploads"
    target_dir = _DOCS_ROOT / safe
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext != ".md":
            raise HTTPException(400, f"Only .md files allowed (got {ext!r})")
        target = target_dir / Path(f.filename or "").name
        with target.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(target.name)
    # Auto-rebuild
    from src.agent.docs_index import DocsIndex, reset_default_index
    idx = DocsIndex().build()
    idx.save(_INDEX_PATH)
    reset_default_index()
    return {
        "saved": saved,
        "subfolder": str(target_dir),
        "sections_after_reindex": idx.section_count(),
    }


# ── /agent/status ────────────────────────────────────────────────────────────


def _count_sas(version: str) -> dict:
    folder = _SAS_ROOT / version
    if not folder.exists():
        return {"folder": str(folder), "count": 0, "files": []}
    files = sorted(str(p.relative_to(folder)) for p in folder.rglob("*.sas"))
    return {"folder": str(folder), "count": len(files), "files": files}


@router.get("/status")
def status() -> dict:
    from src.agent.docs_index import get_default_index
    from src.agent.tools import TOOL_REGISTRY
    from src.knowledge import get_client

    client = get_client()
    backend = client.detect_backend()
    model = (
        client.litert_model if backend == "litert"
        else client.ollama_model if backend == "ollama"
        else "stub"
    )
    idx = get_default_index()
    return {
        "llm": {"backend": backend, "model": model},
        "tools": list(TOOL_REGISTRY.keys()),
        "sas": {"v2": _count_sas("v2"), "v3": _count_sas("v3")},
        "docs": {
            "root": str(_DOCS_ROOT),
            "sections": idx.section_count(),
            "index_path": str(_INDEX_PATH),
            "index_present": _INDEX_PATH.exists(),
        },
    }

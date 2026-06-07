"""Knowledge-base router — change-log graph upload, reindex, query."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(prefix="/kb", tags=["kb"])

_CHANGELOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "changelog"
_GRAPH_PATH = _CHANGELOG_DIR / "graph.json"

_ALLOWED_EXTS = {".md", ".sql", ".txt"}


@router.get("/graph")
def get_graph() -> dict:
    """Return the current change-log graph (nodes + edges)."""
    from src.knowledge import GraphRAG, graph_to_payload
    rag = GraphRAG(graph_path=_GRAPH_PATH, changelog_dir=_CHANGELOG_DIR)
    return graph_to_payload(rag.graph)


@router.post("/reindex")
def reindex() -> dict:
    """Re-build the graph from on-disk change-log files."""
    from src.knowledge import GraphRAG
    rag = GraphRAG(graph_path=_GRAPH_PATH, changelog_dir=_CHANGELOG_DIR)
    g = rag.reindex(persist=True)
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "path": str(_GRAPH_PATH),
    }


@router.post("/changelog/upload")
async def upload_changelog(files: list[UploadFile] = File(...)) -> dict:
    """Upload one or more ``.md`` / ``.sql`` change-log files; rebuild graph."""
    _CHANGELOG_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in _ALLOWED_EXTS:
            raise HTTPException(400, f"Unsupported file type {ext!r}; allowed: {sorted(_ALLOWED_EXTS)}")
        target = _CHANGELOG_DIR / Path(f.filename or "").name
        with target.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(target.name)

    from src.knowledge import GraphRAG
    rag = GraphRAG(graph_path=_GRAPH_PATH, changelog_dir=_CHANGELOG_DIR)
    g = rag.reindex(persist=True)
    return {
        "saved": saved,
        "graph": {"nodes": g.number_of_nodes(), "edges": g.number_of_edges()},
    }


@router.get("/llm-status")
def llm_status() -> dict:
    """Probe the local-LLM backends and report which is reachable + active model."""
    from src.knowledge import get_client
    c = get_client()
    backend = c.detect_backend()
    active_model = (
        c.litert_model if backend == "litert"
        else c.ollama_model if backend == "ollama"
        else "stub"
    )
    return {
        "backend": backend,
        "model": active_model,
        "litert_url": c.litert_url,
        "litert_model": c.litert_model,
        "ollama_url": c.ollama_url,
        "ollama_model": c.ollama_model,
    }

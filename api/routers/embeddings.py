"""Embedding space visualizer — section embeddings + 2D/3D projections."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


class ProjectRequest(BaseModel):
    method: str = "pca"
    n_components: int = 2
    perplexity: float = 30.0
    n_neighbors: int = 15
    min_dist: float = 0.1
    rebuild: bool = False


@router.get("/methods")
def list_methods() -> dict:
    from src.embeddings import PROJECTION_METHODS

    return {"methods": PROJECTION_METHODS}


@router.get("/sections")
def list_sections(rebuild: bool = False) -> dict:
    """Return embedded section metadata (no coordinates) for the corpus."""
    from src.embeddings.embedder import get_default_embeddings, to_payload

    emb = get_default_embeddings(rebuild=rebuild)
    return {"count": len(emb), "sections": to_payload(emb)}


@router.post("/project")
def project_embeddings(req: ProjectRequest) -> dict:
    """Reduce the corpus embeddings to 2D/3D points for plotting."""
    from src.embeddings import project
    from src.embeddings.embedder import get_default_embeddings, to_payload

    if req.n_components not in (2, 3):
        raise HTTPException(400, "n_components must be 2 or 3")

    emb = get_default_embeddings(rebuild=req.rebuild)
    if len(emb) == 0:
        return {"method": req.method, "n_components": req.n_components, "points": []}

    try:
        coords = project(
            emb.vectors,
            method=req.method,
            n_components=req.n_components,
            perplexity=req.perplexity,
            n_neighbors=req.n_neighbors,
            min_dist=req.min_dist,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    meta = to_payload(emb)
    points = []
    for i, m in enumerate(meta):
        point = dict(m)
        point["x"] = float(coords[i, 0])
        point["y"] = float(coords[i, 1])
        if req.n_components == 3:
            point["z"] = float(coords[i, 2])
        points.append(point)

    return {"method": req.method, "n_components": req.n_components, "points": points}

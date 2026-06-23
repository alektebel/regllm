"""Embedding space visualizer — section embeddings + 2D/3D projections + inspection."""

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


class ClusterRequest(BaseModel):
    method: str = "hdbscan"
    n_clusters: int = 8
    min_cluster_size: int = 3
    rebuild: bool = False


class GraphRequest(BaseModel):
    k: int = 5
    threshold: float = 0.3
    cluster_method: str = "hdbscan"
    min_cluster_size: int = 3
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


# ── Inspection endpoints ─────────────────────────────────────────────────────


@router.get("/inspect/anisotropy")
def inspect_anisotropy(rebuild: bool = False) -> dict:
    """Measure embedding space anisotropy."""
    from src.embeddings.embedder import get_default_embeddings
    from src.embeddings.inspector import anisotropy

    emb = get_default_embeddings(rebuild=rebuild)
    return anisotropy(emb.vectors)


@router.get("/inspect/nn-audit")
def inspect_nn_audit(k: int = 5, rebuild: bool = False) -> dict:
    """Nearest-neighbor audit for each embedded section."""
    from src.embeddings.embedder import get_default_embeddings
    from src.embeddings.inspector import nn_audit

    emb = get_default_embeddings(rebuild=rebuild)
    items = nn_audit(emb.vectors, emb.ids, emb.headings, k=k)
    return {"k": k, "count": len(items), "items": items}


@router.post("/inspect/clusters")
def inspect_clusters(req: ClusterRequest) -> dict:
    """Cluster the embedding space and report cluster membership + silhouette."""
    from src.embeddings.embedder import get_default_embeddings
    from src.embeddings.inspector import cluster

    emb = get_default_embeddings(rebuild=req.rebuild)
    result = cluster(emb.vectors, method=req.method,
                     n_clusters=req.n_clusters,
                     min_cluster_size=req.min_cluster_size)
    return result.summary(emb.ids, emb.headings)


@router.get("/inspect/similarity")
def inspect_similarity(max_points: int = 200, rebuild: bool = False) -> dict:
    """Cosine similarity heatmap data."""
    from src.embeddings.embedder import get_default_embeddings
    from src.embeddings.inspector import similarity_matrix

    emb = get_default_embeddings(rebuild=rebuild)
    return similarity_matrix(emb.vectors, emb.ids, max_points=max_points)


@router.post("/inspect/graph")
def inspect_graph(req: GraphRequest) -> dict:
    """k-NN graph with cluster coloring for the force-directed view."""
    from src.embeddings.embedder import get_default_embeddings
    from src.embeddings.inspector import cluster, knn_graph

    emb = get_default_embeddings(rebuild=req.rebuild)
    cl = cluster(emb.vectors, method=req.cluster_method,
                 min_cluster_size=req.min_cluster_size)
    return knn_graph(
        emb.vectors, emb.ids, emb.headings, emb.paths,
        cluster_labels=cl.labels, k=req.k, threshold=req.threshold,
    )

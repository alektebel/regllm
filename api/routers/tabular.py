"""TabBERT-style tabular embedding endpoints — projection + clustering + search.

Depends on pre-built embeddings from ``scripts/train_tabular_embeddings.py``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/embeddings/tabular", tags=["tabular"])


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


class SearchRequest(BaseModel):
    id: str
    k: int = 10
    rebuild: bool = False


@router.get("/metadata")
def metadata(rebuild: bool = False) -> dict:
    """Return summary stats for the tabular embedding set."""
    from src.embeddings.tabular.embedder import get_default_tabular_embeddings

    emb = get_default_tabular_embeddings(rebuild=rebuild)
    if emb is None:
        raise HTTPException(404, "No tabular embeddings found. Run train_tabular_embeddings.py first.")
    return emb.summary()


@router.get("/fields")
def list_fields(rebuild: bool = False) -> dict:
    """List available metadata fields for coloring/filtering."""
    from src.embeddings.tabular.embedder import get_default_tabular_embeddings

    emb = get_default_tabular_embeddings(rebuild=rebuild)
    if emb is None or not emb.metadata:
        raise HTTPException(404, "No tabular embeddings found.")

    all_keys = set()
    for m in emb.metadata:
        all_keys.update(m.keys())
    return {"fields": sorted(all_keys)}


@router.get("/values/{field}")
def field_values(field: str, rebuild: bool = False) -> dict:
    """Return unique values for a metadata field (for legend construction)."""
    from src.embeddings.tabular.embedder import get_default_tabular_embeddings

    emb = get_default_tabular_embeddings(rebuild=rebuild)
    if emb is None:
        raise HTTPException(404, "No tabular embeddings found.")

    values = sorted(set(str(m.get(field, "NONE")) for m in emb.metadata))
    return {"field": field, "values": values}


@router.post("/project")
def project_embeddings(req: ProjectRequest) -> dict:
    """Reduce tabular embeddings to 2D/3D for plotting."""
    from src.embeddings import project
    from src.embeddings.tabular.embedder import get_default_tabular_embeddings

    if req.n_components not in (2, 3):
        raise HTTPException(400, "n_components must be 2 or 3")

    emb = get_default_tabular_embeddings(rebuild=req.rebuild)
    if emb is None or len(emb) == 0:
        raise HTTPException(404, "No tabular embeddings found.")

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

    points = []
    for i in range(len(emb)):
        point = {
            "id": emb.ids[i],
            **emb.metadata[i],
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
        }
        if req.n_components == 3:
            point["z"] = float(coords[i, 2])
        points.append(point)

    return {
        "method": req.method,
        "n_components": req.n_components,
        "n_points": len(points),
        "points": points,
    }


@router.post("/search")
def search_similar(req: SearchRequest) -> dict:
    """Cosine similarity search for nearest neighbors of a row ID."""
    from src.embeddings.tabular.embedder import get_default_tabular_embeddings

    emb = get_default_tabular_embeddings(rebuild=req.rebuild)
    if emb is None or len(emb) == 0:
        raise HTTPException(404, "No tabular embeddings found.")

    try:
        idx = emb.ids.index(req.id)
    except ValueError:
        raise HTTPException(404, f"Row ID '{req.id}' not found.")

    norms = emb.vectors / (  # type: ignore[operator]
        (emb.vectors ** 2).sum(axis=1, keepdims=True) ** 0.5
    )
    sims = norms @ norms[idx]
    sims[idx] = -2.0  # exclude self
    top_k = sims.argsort()[::-1][: req.k]

    results = []
    for j in top_k:
        results.append({
            "id": emb.ids[j],
            "metadata": emb.metadata[j],
            "cosine": float(round(sims[j], 4)),
        })

    return {
        "query_id": req.id,
        "query_metadata": emb.metadata[idx],
        "k": len(results),
        "results": results,
    }


@router.post("/project/plot-url")
def plot_url(
    method: str = Query("pca"),
    color_by: str = Query("TERMINACION"),
    n_components: int = Query(2),
    rebuild: bool = Query(False),
) -> dict:
    """Return projected points suitable for a scatter plot, colored by field.

    This endpoint is designed for the frontend visualization: it returns the
    same data as /project but with a recommended ``color_field`` so the UI
    knows which column to use for color mapping.
    """
    from src.embeddings import project
    from src.embeddings.tabular.embedder import get_default_tabular_embeddings

    emb = get_default_tabular_embeddings(rebuild=rebuild)
    if emb is None or len(emb) == 0:
        raise HTTPException(404, "No tabular embeddings found.")

    try:
        coords = project(emb.vectors, method=method, n_components=n_components)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    color_labels = [str(m.get(color_by, "NONE")) for m in emb.metadata]
    unique_colors = sorted(set(color_labels))

    points = []
    for i in range(len(emb)):
        points.append({
            "id": emb.ids[i],
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            color_by.lower(): color_labels[i],
        })

    return {
        "method": method,
        "n_points": len(points),
        "color_field": color_by,
        "color_values": unique_colors,
        "points": points,
    }


@router.post("/inspect/anisotropy")
def inspect_anisotropy(rebuild: bool = False) -> dict:
    """Measure embedding space anisotropy."""
    from src.embeddings.inspector import anisotropy
    from src.embeddings.tabular.embedder import get_default_tabular_embeddings

    emb = get_default_tabular_embeddings(rebuild=rebuild)
    if emb is None:
        raise HTTPException(404, "No tabular embeddings found.")
    return anisotropy(emb.vectors)


@router.post("/inspect/clusters")
def inspect_clusters(req: ClusterRequest) -> dict:
    """Cluster the tabular embedding space."""
    from src.embeddings.inspector import cluster
    from src.embeddings.tabular.embedder import get_default_tabular_embeddings

    emb = get_default_tabular_embeddings(rebuild=req.rebuild)
    if emb is None:
        raise HTTPException(404, "No tabular embeddings found.")

    result = cluster(
        emb.vectors,
        method=req.method,
        n_clusters=req.n_clusters,
        min_cluster_size=req.min_cluster_size,
    )
    return result.summary(emb.ids, [m.get("ID_CONTR_CICLO_LGD", "") for m in emb.metadata])

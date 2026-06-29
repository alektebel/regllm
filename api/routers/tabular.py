"""TabBERT-style tabular embedding endpoints — upload, projection, demo UI."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

router = APIRouter(prefix="/embeddings/tabular", tags=["tabular"])


class ProjectRequest(BaseModel):
    method: str = "pca"
    n_components: int = 2
    perplexity: float = 30.0
    n_neighbors: int = 15
    min_dist: float = 0.1
    rebuild: bool = False
    session_id: str | None = None


class ClusterRequest(BaseModel):
    method: str = "hdbscan"
    n_clusters: int = 8
    min_cluster_size: int = 3
    rebuild: bool = False
    session_id: str | None = None


class SearchRequest(BaseModel):
    id: str
    k: int = 10
    rebuild: bool = False
    session_id: str | None = None


class BuildRequest(BaseModel):
    session_id: str
    id_column: str
    pk_column: str | None = None
    sample: int = Field(default=1500, ge=10, le=10000)
    strategy: str = "colon"


class DemoRequest(BaseModel):
    session_id: str | None = None
    use_default: bool = False
    method: str = "umap"
    color_field: str | None = None
    id_column: str | None = None
    pk_column: str | None = None
    sample: int = Field(default=1500, ge=10, le=10000)
    graph_k: int = Field(default=5, ge=1, le=20)
    graph_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    rebuild: bool = False


def _resolve_embeddings(session_id: str | None, rebuild: bool = False):
    from src.embeddings.tabular.embedder import get_embeddings_for_session

    emb = get_embeddings_for_session(session_id, rebuild=rebuild)
    if emb is None or len(emb) == 0:
        raise HTTPException(
            404,
            "No tabular embeddings found. Upload a file and call /build, or use use_default=true.",
        )
    return emb


@router.post("/upload")
async def upload_table(file: UploadFile = File(...)) -> dict:
    """Upload CSV/XLSX and return column preview + suggested ID column."""
    import uuid

    from src.embeddings.tabular.embedder import UPLOADS_ROOT, create_upload_session

    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(400, f"Only .csv and .xlsx allowed (got {ext!r})")

    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    target = UPLOADS_ROOT / f"{session_id}{ext}"
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    session = create_upload_session(target, file.filename or "upload", session_id=session_id)
    return {
        "session_id": session.session_id,
        "filename": session.filename,
        "columns": session.columns,
        "row_count": session.row_count,
        "preview": session.preview,
        "suggested_id_column": session.suggested_id_column,
    }


@router.post("/build")
def build_embeddings(req: BuildRequest) -> dict:
    """Embed uploaded rows with chosen ID / PK columns."""
    from src.embeddings.tabular.embedder import (
        build_tabular_embeddings_from_path,
        get_session,
        set_session_embeddings,
    )

    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(404, f"Session '{req.session_id}' not found")

    if req.id_column not in session.columns:
        raise HTTPException(400, f"Column '{req.id_column}' not in uploaded file")

    pk = req.pk_column or req.id_column
    if pk not in session.columns:
        raise HTTPException(400, f"Column '{pk}' not in uploaded file")

    emb = build_tabular_embeddings_from_path(
        session.file_path,
        strategy=req.strategy,
        sample=req.sample,
        id_column=req.id_column,
        pk_column=pk,
    )
    set_session_embeddings(req.session_id, emb)
    return {"session_id": req.session_id, **emb.summary()}


@router.post("/demo")
def tabular_demo(req: DemoRequest) -> dict:
    """Full demo payload: 2D projection, k-NN graph, separating vars, outliers."""
    from src.embeddings import project
    from src.embeddings.inspector import cluster, knn_graph
    from src.embeddings.tabular.analysis import outlier_scores, top_separating_variables
    from src.embeddings.tabular.embedder import (
        build_tabular_embeddings,
        build_tabular_embeddings_from_path,
        get_session,
        set_session_embeddings,
    )

    if req.use_default:
        emb = build_tabular_embeddings(sample=req.sample)
    elif req.session_id:
        session = get_session(req.session_id)
        if session is None:
            raise HTTPException(404, f"Session '{req.session_id}' not found")
        if session.embeddings and not req.rebuild:
            emb = session.embeddings
        else:
            id_col = req.id_column or session.suggested_id_column
            if not id_col:
                raise HTTPException(400, "id_column required")
            pk = req.pk_column or id_col
            emb = build_tabular_embeddings_from_path(
                session.file_path,
                sample=req.sample,
                id_column=id_col,
                pk_column=pk,
            )
            set_session_embeddings(req.session_id, emb)
    else:
        emb = _resolve_embeddings(None, rebuild=req.rebuild)

    if len(emb) == 0:
        raise HTTPException(400, "No rows to embed")

    try:
        coords = project(
            emb.vectors,
            method=req.method,
            n_components=2,
            perplexity=30.0,
            n_neighbors=15,
            min_dist=0.1,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    cluster_result = cluster(emb.vectors, method="hdbscan", min_cluster_size=max(3, len(emb) // 100))
    labels = cluster_result.labels

    skip = {"_id", "_pk", emb.id_column, emb.pk_column}
    separating = top_separating_variables(emb.metadata, labels, top_k=5, skip_fields=skip)

    color_field = req.color_field
    if not color_field:
        color_field = separating[0]["field"] if separating else None
    if color_field and color_field not in (emb.metadata[0] if emb.metadata else {}):
        color_field = emb.id_column or None

    outliers = outlier_scores(emb.vectors, k=min(10, len(emb) - 1))

    headings = [str(m.get("_pk", m.get("_id", emb.ids[i]))) for i, m in enumerate(emb.metadata)]
    paths = [str(m.get(emb.id_column, emb.ids[i])) for i, m in enumerate(emb.metadata)]

    graph = knn_graph(
        emb.vectors,
        emb.ids,
        headings,
        paths,
        cluster_labels=labels,
        k=req.graph_k,
        threshold=req.graph_threshold,
    )

    color_labels = [
        str(m.get(color_field, "NONE")) if color_field else str(int(labels[i]))
        for i, m in enumerate(emb.metadata)
    ]
    unique_colors = sorted(set(color_labels))

    points = []
    for i in range(len(emb)):
        points.append({
            "id": emb.ids[i],
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "color": color_labels[i],
            "cluster": int(labels[i]),
            "outlier_score": round(float(outliers[i]), 4),
            "heading": headings[i],
            "metadata": emb.metadata[i],
        })

    for node in graph["nodes"]:
        idx = emb.ids.index(node["id"])
        node["color"] = color_labels[idx]
        node["outlier_score"] = round(float(outliers[idx]), 4)
        node["metadata"] = emb.metadata[idx]

    return {
        "session_id": req.session_id,
        "summary": emb.summary(),
        "method": req.method,
        "color_field": color_field,
        "color_values": unique_colors,
        "separating_variables": separating,
        "n_clusters": cluster_result.n_clusters,
        "silhouette": round(cluster_result.silhouette, 4) if cluster_result.silhouette is not None else None,
        "points": points,
        "graph": graph,
    }


@router.get("/metadata")
def metadata(rebuild: bool = False, session_id: str | None = None) -> dict:
    emb = _resolve_embeddings(session_id, rebuild=rebuild)
    return emb.summary()


@router.get("/fields")
def list_fields(rebuild: bool = False, session_id: str | None = None) -> dict:
    emb = _resolve_embeddings(session_id, rebuild=rebuild)
    all_keys = set()
    for m in emb.metadata:
        all_keys.update(m.keys())
    all_keys -= {"_id", "_pk"}
    return {"fields": sorted(all_keys)}


@router.get("/values/{field}")
def field_values(field: str, rebuild: bool = False, session_id: str | None = None) -> dict:
    emb = _resolve_embeddings(session_id, rebuild=rebuild)
    values = sorted(set(str(m.get(field, "NONE")) for m in emb.metadata))
    return {"field": field, "values": values}


@router.post("/project")
def project_embeddings(req: ProjectRequest) -> dict:
    from src.embeddings import project

    emb = _resolve_embeddings(req.session_id, rebuild=req.rebuild)
    if req.n_components not in (2, 3):
        raise HTTPException(400, "n_components must be 2 or 3")

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
        point = {"id": emb.ids[i], **emb.metadata[i], "x": float(coords[i, 0]), "y": float(coords[i, 1])}
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
    emb = _resolve_embeddings(req.session_id, rebuild=req.rebuild)

    try:
        idx = emb.ids.index(req.id)
    except ValueError:
        raise HTTPException(404, f"Row ID '{req.id}' not found.")

    norms = emb.vectors / ((emb.vectors ** 2).sum(axis=1, keepdims=True) ** 0.5)
    sims = norms @ norms[idx]
    sims[idx] = -2.0
    top_k = sims.argsort()[::-1][: req.k]

    results = [{"id": emb.ids[j], "metadata": emb.metadata[j], "cosine": float(round(sims[j], 4))} for j in top_k]
    return {"query_id": req.id, "query_metadata": emb.metadata[idx], "k": len(results), "results": results}


@router.post("/project/plot-url")
def plot_url(
    method: str = Query("pca"),
    color_by: str = Query("TERMINACION"),
    n_components: int = Query(2),
    rebuild: bool = Query(False),
    session_id: str | None = Query(None),
) -> dict:
    from src.embeddings import project

    emb = _resolve_embeddings(session_id, rebuild=rebuild)
    try:
        coords = project(emb.vectors, method=method, n_components=n_components)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    color_labels = [str(m.get(color_by, "NONE")) for m in emb.metadata]
    points = [
        {"id": emb.ids[i], "x": float(coords[i, 0]), "y": float(coords[i, 1]), color_by.lower(): color_labels[i]}
        for i in range(len(emb))
    ]
    return {
        "method": method,
        "n_points": len(points),
        "color_field": color_by,
        "color_values": sorted(set(color_labels)),
        "points": points,
    }


@router.post("/inspect/anisotropy")
def inspect_anisotropy(rebuild: bool = False, session_id: str | None = None) -> dict:
    from src.embeddings.inspector import anisotropy

    emb = _resolve_embeddings(session_id, rebuild=rebuild)
    return anisotropy(emb.vectors)


@router.post("/inspect/clusters")
def inspect_clusters(req: ClusterRequest) -> dict:
    from src.embeddings.inspector import cluster

    emb = _resolve_embeddings(req.session_id, rebuild=req.rebuild)
    result = cluster(
        emb.vectors,
        method=req.method,
        n_clusters=req.n_clusters,
        min_cluster_size=req.min_cluster_size,
    )
    headings = [str(m.get("_pk", m.get("_id", ""))) for m in emb.metadata]
    return result.summary(emb.ids, headings)

"""Embedding space visualizer — docs KB embeddings + tabular data (Excel/CSV)."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

_ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".tsv"}


# ── Docs / KB embeddings ─────────────────────────────────────────────────────

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
    from src.embeddings.embedder import get_default_embeddings, to_payload

    emb = get_default_embeddings(rebuild=rebuild)
    return {"count": len(emb), "sections": to_payload(emb)}


@router.post("/project")
def project_embeddings(req: ProjectRequest) -> dict:
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


# ── Tabular / spreadsheet embeddings ─────────────────────────────────────────

class SpreadsheetProjectRequest(BaseModel):
    upload_id: str
    id_col: str
    feature_cols: list[str] | None = None
    color_col: str | None = None    # optional column whose value is used for point colour
    method: str = "pca"
    n_components: int = 3
    perplexity: float = 30.0
    n_neighbors: int = 15
    min_dist: float = 0.1
    rebuild: bool = False


@router.post("/upload-spreadsheet")
async def upload_spreadsheet(file: UploadFile = File(...)) -> dict:
    """Upload an Excel or CSV file; return upload_id + column metadata."""
    from pathlib import Path

    from src.embeddings.tabular import _ALLOWED_EXTENSIONS, column_info, save_upload

    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type {ext!r}; allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    import pandas as pd

    content = await file.read()
    upload_id, saved_path = save_upload(content, file.filename or "upload")

    try:
        if ext in {".xlsx", ".xls"}:
            df = pd.read_excel(saved_path)
        elif ext == ".tsv":
            df = pd.read_csv(saved_path, sep="\t", nrows=5)
        else:
            df = pd.read_csv(saved_path, nrows=5)
    except Exception as e:
        raise HTTPException(422, f"Could not parse file: {e}") from e

    # reload full frame for column info
    try:
        if ext in {".xlsx", ".xls"}:
            full_df = pd.read_excel(saved_path)
        elif ext == ".tsv":
            full_df = pd.read_csv(saved_path, sep="\t")
        else:
            full_df = pd.read_csv(saved_path)
    except Exception:
        full_df = df

    return {
        "upload_id": upload_id,
        "filename": file.filename,
        "n_rows": len(full_df),
        "columns": column_info(full_df),
        "sample": df.head(3).fillna("").to_dict(orient="records"),
    }


@router.post("/project-spreadsheet")
def project_spreadsheet(req: SpreadsheetProjectRequest) -> dict:
    """Compute tabular embeddings + 2D/3D projection for an uploaded file."""
    from src.embeddings import project
    from src.embeddings.embedder import to_payload
    from src.embeddings.tabular import get_or_build

    if req.n_components not in (2, 3):
        raise HTTPException(400, "n_components must be 2 or 3")

    try:
        emb = get_or_build(
            req.upload_id,
            id_col=req.id_col,
            feature_cols=req.feature_cols,
            rebuild=req.rebuild,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

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

    # Load raw df once to resolve optional color_col values
    from src.embeddings.tabular import load_upload

    raw_df = load_upload(req.upload_id) if req.color_col else None
    color_values: list[str] = []
    if raw_df is not None and req.color_col and req.color_col in raw_df.columns:
        color_values = raw_df[req.color_col].astype(str).tolist()

    # count occurrences per id to flag "evolving" records (id appears > 1×)
    id_counts: dict[str, int] = {}
    for m in meta:
        id_counts[m["heading"]] = id_counts.get(m["heading"], 0) + 1

    points = []
    for i, m in enumerate(meta):
        point = dict(m)
        point["x"] = float(coords[i, 0])
        point["y"] = float(coords[i, 1])
        if req.n_components == 3:
            point["z"] = float(coords[i, 2])
        point["evolving"] = id_counts[m["heading"]] > 1
        point["colorGroup"] = color_values[i] if color_values else None
        points.append(point)

    return {
        "method": req.method,
        "n_components": req.n_components,
        "id_col": req.id_col,
        "points": points,
    }

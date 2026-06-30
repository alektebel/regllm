"""Embedding builder for tabular data (Excel / CSV).

Numeric columns → StandardScaler, categorical → OneHotEncoder.
All features are concatenated into a dense matrix, then reduced to
at most 64 dimensions with TruncatedSVD so t-SNE/UMAP have a
manageable input. Each *row* becomes one point in embedding space.

"Evolving registers" = rows that share an ID across versions (e.g.
CICLO_ID in V2 vs V3). The visualizer highlights matched pairs when the
same ID appears more than once in the uploaded data.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.embeddings.embedder import EmbeddingSet

_UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "_uploads"
_LATENT_DIM = 64
_MAX_CATEGORIES = 32          # OHE cap: columns with more unique values become text
_ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".tsv"}


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_upload(content: bytes, filename: str) -> tuple[str, Path]:
    """Save an uploaded file under a fresh UUID, return (upload_id, path)."""
    upload_id = uuid.uuid4().hex
    dest_dir = _UPLOAD_DIR / upload_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(filename).name
    dest.write_bytes(content)
    return upload_id, dest


def load_upload(upload_id: str) -> pd.DataFrame:
    """Read the dataframe for a previously uploaded file."""
    upload_dir = _UPLOAD_DIR / upload_id
    if not upload_dir.exists():
        raise FileNotFoundError(f"No upload with id {upload_id!r}")
    files = list(upload_dir.iterdir())
    if not files:
        raise FileNotFoundError(f"Upload {upload_id!r} is empty")
    return _read_file(files[0])


def _read_file(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype_backend="numpy_nullable")
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def column_info(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return per-column metadata for the frontend column picker."""
    info = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
        elif s.nunique(dropna=True) <= _MAX_CATEGORIES:
            kind = "categorical"
        else:
            kind = "text"
        info.append({"name": col, "kind": kind, "n_unique": int(s.nunique(dropna=True))})
    return info


# ── Tabular embedding ─────────────────────────────────────────────────────────

def embed_dataframe(
    df: pd.DataFrame,
    id_col: str,
    feature_cols: list[str] | None = None,
) -> EmbeddingSet:
    """Build an EmbeddingSet where every row is one point."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer

    if id_col not in df.columns:
        raise ValueError(f"ID column {id_col!r} not found in dataframe")

    all_cols = [c for c in df.columns if c != id_col]
    if feature_cols:
        feat_cols = [c for c in feature_cols if c in df.columns and c != id_col]
    else:
        feat_cols = all_cols

    if not feat_cols:
        raise ValueError("No feature columns available after excluding the ID column")

    numeric = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [
        c for c in feat_cols
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) <= _MAX_CATEGORIES
    ]

    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))

    if not transformers:
        raise ValueError("No numeric or low-cardinality categorical columns to embed")

    ct = ColumnTransformer(transformers, remainder="drop")
    dense = ct.fit_transform(df[feat_cols].fillna(0))

    n_components = max(1, min(_LATENT_DIM, dense.shape[0] - 1, dense.shape[1] - 1))
    if n_components >= dense.shape[1]:
        vectors = np.asarray(dense, dtype=np.float64)
    else:
        vectors = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(dense)

    id_vals = df[id_col].astype(str).tolist()

    def _row_snippet(row: pd.Series) -> str:
        parts = [f"{c}={row[c]}" for c in feat_cols[:8] if c in row.index]
        return "  ".join(parts)

    snippets = [_row_snippet(df.iloc[i]) for i in range(len(df))]

    return EmbeddingSet(
        ids=id_vals,
        vectors=np.asarray(vectors, dtype=np.float64),
        paths=[id_col] * len(df),          # reuse "path" as the color group label
        headings=id_vals,
        snippets=snippets,
        levels=[1] * len(df),
    )


# ── In-memory cache keyed by upload_id ───────────────────────────────────────

@dataclass
class _CachedSet:
    emb: EmbeddingSet
    id_col: str
    feature_cols: list[str]


_CACHE: dict[str, _CachedSet] = {}


def get_or_build(
    upload_id: str,
    id_col: str,
    feature_cols: list[str] | None = None,
    rebuild: bool = False,
) -> EmbeddingSet:
    key = f"{upload_id}:{id_col}:{','.join(sorted(feature_cols or []))}"
    if key in _CACHE and not rebuild:
        return _CACHE[key].emb
    df = load_upload(upload_id)
    emb = embed_dataframe(df, id_col=id_col, feature_cols=feature_cols)
    _CACHE[key] = _CachedSet(emb=emb, id_col=id_col, feature_cols=feature_cols or [])
    return emb

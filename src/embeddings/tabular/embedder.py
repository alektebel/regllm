"""TabBERT-style embedder for tabular data.

Loads tabular rows, serializes each to text via a configurable strategy
(TabBERT-style colon format by default), and embeds via Ollama
``nomic-embed-text`` with a sklearn hash-vector fallback when Ollama is
unavailable.
"""

from __future__ import annotations

import csv
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings.tabular.io import load_table, suggest_id_column, table_columns
from src.embeddings.tabular.serializer import SerializationStrategy, serialize_row
from src.knowledge.embeddings import get_embedding_service

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "tabular" / "embeddings.json"
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "samples" / "recuperatory_cycles.csv"
UPLOADS_ROOT = PROJECT_ROOT / "data" / "uploads" / "tabular"


@dataclass
class TabularEmbeddingSet:
    """Container for tabular row embeddings + metadata."""

    ids: list[str] = field(default_factory=list)
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    metadata: list[dict[str, Any]] = field(default_factory=list)
    strategy: str = "colon"
    n_rows: int = 0
    id_column: str = ""
    pk_column: str = ""
    embed_backend: str = "ollama"
    source_path: str = ""

    def __len__(self) -> int:
        return len(self.ids)

    def save(self, path: Path | None = None) -> Path:
        path = path or DEFAULT_EMBEDDINGS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "strategy": self.strategy,
            "n_rows": self.n_rows,
            "id_column": self.id_column,
            "pk_column": self.pk_column,
            "embed_backend": self.embed_backend,
            "source_path": self.source_path,
            "ids": self.ids,
            "vectors": self.vectors.tolist(),
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        logger.info("Saved %d tabular embeddings to %s", len(self), path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "TabularEmbeddingSet | None":
        path = path or DEFAULT_EMBEDDINGS_PATH
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            ids=data["ids"],
            vectors=np.array(data["vectors"], dtype=np.float64),
            metadata=data["metadata"],
            strategy=data.get("strategy", "colon"),
            n_rows=data.get("n_rows", len(data["ids"])),
            id_column=data.get("id_column", ""),
            pk_column=data.get("pk_column", ""),
            embed_backend=data.get("embed_backend", "ollama"),
            source_path=data.get("source_path", ""),
        )

    def get_labels(self, field: str) -> list[str]:
        return [str(m.get(field, "NONE")) for m in self.metadata]

    def summary(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_embedded": len(self),
            "dim": self.vectors.shape[1] if len(self.vectors.shape) > 1 else 0,
            "strategy": self.strategy,
            "id_column": self.id_column,
            "pk_column": self.pk_column,
            "embed_backend": self.embed_backend,
            "source_path": self.source_path,
        }


@dataclass
class TabularSession:
    session_id: str
    file_path: Path
    filename: str
    columns: list[str]
    row_count: int
    preview: list[dict[str, Any]]
    suggested_id_column: str | None
    embeddings: TabularEmbeddingSet | None = None


def _load_rows_csv(csv_path: Path) -> list[dict[str, Any]]:
    with open(csv_path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _embed_texts(texts: list[str], batch_size: int = 32) -> tuple[np.ndarray, str]:
    """Embed serialized rows; fall back to hash vectors when Ollama is down."""
    embedder = get_embedding_service()
    if embedder.available:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(embedder.embed(batch))
            logger.info("  embedded %d / %d rows", min(i + batch_size, len(texts)), len(texts))
        vecs = np.array(all_embeddings, dtype=np.float64)
        if vecs.size and np.any(vecs != 0):
            return vecs, "ollama"

    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.preprocessing import normalize

    logger.warning("Ollama unavailable — using hash-vector fallback for tabular demo")
    hv = HashingVectorizer(n_features=512, alternate_sign=False)
    X = hv.transform(texts)
    return normalize(X).astype(np.float64).toarray(), "hash"


def _build_metadata(row: dict[str, Any], id_column: str, pk_column: str) -> dict[str, Any]:
    meta = {str(k): v for k, v in row.items()}
    meta["_id"] = str(row.get(id_column, ""))
    meta["_pk"] = str(row.get(pk_column, row.get(id_column, "")))
    return meta


def build_tabular_embeddings(
    csv_path: Path | None = None,
    strategy: SerializationStrategy | str = SerializationStrategy.COLON,
    sample: int | None = None,
    batch_size: int = 32,
    id_column: str | None = None,
    pk_column: str | None = None,
) -> TabularEmbeddingSet:
    """Build embeddings for rows in a CSV file."""
    if csv_path is None:
        csv_path = DEFAULT_CSV_PATH

    rows = _load_rows_csv(csv_path)
    if sample:
        rows = rows[:sample]

    if not rows:
        return TabularEmbeddingSet(source_path=str(csv_path))

    cols = list(rows[0].keys())
    id_col = id_column or suggest_id_column(cols) or cols[0]
    pk_col = pk_column or id_col

    if isinstance(strategy, str):
        strategy = SerializationStrategy(strategy)

    texts = [serialize_row(r, strategy) for r in rows]
    vectors, backend = _embed_texts(texts, batch_size=batch_size)

    ids = [str(r.get(id_col, j)) for j, r in enumerate(rows)]
    metadata = [_build_metadata(r, id_col, pk_col) for r in rows]

    return TabularEmbeddingSet(
        ids=ids,
        vectors=vectors,
        metadata=metadata,
        strategy=strategy.value,
        n_rows=len(rows),
        id_column=id_col,
        pk_column=pk_col,
        embed_backend=backend,
        source_path=str(csv_path),
    )


def build_tabular_embeddings_from_path(
    path: Path,
    strategy: SerializationStrategy | str = SerializationStrategy.COLON,
    sample: int | None = None,
    batch_size: int = 32,
    id_column: str | None = None,
    pk_column: str | None = None,
) -> TabularEmbeddingSet:
    """Build embeddings from CSV or Excel."""
    rows = load_table(path, max_rows=sample)
    if not rows:
        return TabularEmbeddingSet(source_path=str(path))

    cols = list(rows[0].keys())
    id_col = id_column or suggest_id_column(cols) or cols[0]
    pk_col = pk_column or id_col

    if isinstance(strategy, str):
        strategy = SerializationStrategy(strategy)

    texts = [serialize_row(r, strategy) for r in rows]
    vectors, backend = _embed_texts(texts, batch_size=batch_size)

    ids = [str(r.get(id_col, j)) for j, r in enumerate(rows)]
    metadata = [_build_metadata(r, id_col, pk_col) for r in rows]

    return TabularEmbeddingSet(
        ids=ids,
        vectors=vectors,
        metadata=metadata,
        strategy=strategy.value,
        n_rows=len(rows),
        id_column=id_col,
        pk_column=pk_col,
        embed_backend=backend,
        source_path=str(path),
    )


# ── Session store (in-memory, per API process) ───────────────────────────────

_sessions: dict[str, TabularSession] = {}


def create_upload_session(
    file_path: Path,
    filename: str,
    session_id: str | None = None,
) -> TabularSession:
    session_id = session_id or uuid.uuid4().hex[:12]
    rows = load_table(file_path, max_rows=5)
    all_rows = load_table(file_path)
    columns = table_columns(file_path) if not rows else list(rows[0].keys())
    session = TabularSession(
        session_id=session_id,
        file_path=file_path,
        filename=filename,
        columns=columns,
        row_count=len(all_rows),
        preview=rows[:5],
        suggested_id_column=suggest_id_column(columns),
    )
    _sessions[session_id] = session
    return session


def get_session(session_id: str) -> TabularSession | None:
    return _sessions.get(session_id)


def set_session_embeddings(session_id: str, emb: TabularEmbeddingSet) -> None:
    session = _sessions.get(session_id)
    if session:
        session.embeddings = emb


# ── Module-level cache (default dataset) ────────────────────────────────────

_default: TabularEmbeddingSet | None = None


def get_default_tabular_embeddings(rebuild: bool = False) -> TabularEmbeddingSet | None:
    global _default
    if _default is None and not rebuild:
        _default = TabularEmbeddingSet.load()
    if _default is None or rebuild:
        if DEFAULT_CSV_PATH.exists():
            _default = build_tabular_embeddings(sample=1500)
            _default.save()
        else:
            return None
    return _default


def get_embeddings_for_session(session_id: str | None, rebuild: bool = False) -> TabularEmbeddingSet | None:
    if session_id:
        session = get_session(session_id)
        if session and session.embeddings:
            return session.embeddings
        if session and rebuild:
            emb = build_tabular_embeddings_from_path(session.file_path, id_column=session.suggested_id_column or None)
            session.embeddings = emb
            return emb
    return get_default_tabular_embeddings(rebuild=rebuild)


def reset_default_tabular_embeddings() -> None:
    global _default
    _default = None

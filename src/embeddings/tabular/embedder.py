"""TabBERT-style embedder for tabular data.

Loads a CSV of recuperatory cycles, serializes each row to text via a
configurable strategy (TabBERT-style colon format by default), and embeds
via the existing nomic-embed-text Ollama service.

Output is a TabularEmbeddingSet that stores embeddings alongside row
metadata for projection, clustering, and visualization.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings.tabular.serializer import SerializationStrategy, serialize_row
from src.knowledge.embeddings import get_embedding_service

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "tabular" / "embeddings.json"


@dataclass
class TabularEmbeddingSet:
    """Container for tabular row embeddings + metadata."""

    ids: list[str] = field(default_factory=list)
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    metadata: list[dict[str, Any]] = field(default_factory=list)
    strategy: str = "colon"
    n_rows: int = 0

    def __len__(self) -> int:
        return len(self.ids)

    def save(self, path: Path | None = None) -> Path:
        path = path or DEFAULT_EMBEDDINGS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "strategy": self.strategy,
            "n_rows": self.n_rows,
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
        )

    def get_labels(self, field: str) -> list[str]:
        """Extract labels for a metadata field across all rows."""
        return [str(m.get(field, "NONE")) for m in self.metadata]

    def summary(self) -> dict[str, Any]:
        """Return a summary of the embedding set."""
        return {
            "n_rows": self.n_rows,
            "n_embedded": len(self),
            "dim": self.vectors.shape[1] if len(self.vectors.shape) > 1 else 0,
            "strategy": self.strategy,
        }


def _load_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Load CSV rows as a list of dicts."""
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_tabular_embeddings(
    csv_path: Path | None = None,
    strategy: SerializationStrategy | str = SerializationStrategy.COLON,
    sample: int | None = None,
    batch_size: int = 32,
) -> TabularEmbeddingSet:
    """Build embeddings for all rows in the recuperatory cycles CSV.

    Args:
        csv_path: Path to the CSV.  Defaults to the generated dataset.
        strategy: Serialization strategy (colon, flat, json, col_val).
        sample: If set, only embed N rows (for testing).
        batch_size: Rows per embedding API call.

    Returns:
        TabularEmbeddingSet with ids, vectors, and metadata.
    """
    if csv_path is None:
        csv_path = PROJECT_ROOT / "data" / "samples" / "recuperatory_cycles.csv"

    rows = _load_rows(csv_path)
    if sample:
        rows = rows[:sample]

    if isinstance(strategy, str):
        strategy = SerializationStrategy(strategy)

    embedder = get_embedding_service()

    texts = [serialize_row(r, strategy) for r in rows]

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embs = embedder.embed(batch)
        all_embeddings.extend(batch_embs)
        logger.info("  embedded %d / %d rows", min(i + batch_size, len(texts)), len(texts))

    ids = [r.get("ID_CONTR_CICLO_LGD", str(j)) for j, r in enumerate(rows)]
    metadata = [_build_metadata(r) for r in rows]

    return TabularEmbeddingSet(
        ids=ids,
        vectors=np.array(all_embeddings, dtype=np.float64),
        metadata=metadata,
        strategy=strategy.value,
        n_rows=len(rows),
    )


def _build_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata fields for projection coloring / filtering."""
    return {
        "ID_CONTR_CICLO_LGD": row.get("ID_CONTR_CICLO_LGD", ""),
        "ID_FUSION_FINAL": row.get("ID_FUSION_FINAL", ""),
        "SEGMENTO": row.get("SEGMENTO", ""),
        "CALIBRATION_SEGMENT": row.get("CALIBRATION_SEGMENT", ""),
        "PRODUCTO": row.get("PRODUCTO", ""),
        "TERMINACION": row.get("TERMINACION", ""),
        "ESTADO_CICLO": row.get("ESTADO_CICLO", ""),
        "STAGE_IFRS9": row.get("STAGE_IFRS9", ""),
        "MES_CICLO": int(row.get("MES_CICLO", 0)),
        "CURE_FLAG": int(row.get("CURE_FLAG", 0)),
        "ADJUDICACION_FLAG": int(row.get("ADJUDICACION_FLAG", 0)),
        "SEGMENTO_PRODUCTO": f"{row.get('SEGMENTO', '')}_{row.get('PRODUCTO', '')}",
    }


# ── Module-level cache ───────────────────────────────────────────────────────

_default: TabularEmbeddingSet | None = None


def get_default_tabular_embeddings(rebuild: bool = False) -> TabularEmbeddingSet | None:
    global _default
    if _default is None and not rebuild:
        _default = TabularEmbeddingSet.load()
    if _default is None or rebuild:
        _default = build_tabular_embeddings()
        _default.save()
    return _default


def reset_default_tabular_embeddings() -> None:
    global _default
    _default = None

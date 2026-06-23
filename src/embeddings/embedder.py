"""Builds dense embedding vectors for the markdown docs corpus.

Uses TF-IDF + truncated SVD (latent semantic analysis) rather than a
downloaded neural embedding model — the docs corpus is small (banking
glossary / change-log sections) and this keeps the pipeline dependency-light
and fully offline, while still producing vectors dense enough to cluster
semantically related sections for the 2D/3D projection UI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.agent.docs_index import DocsIndex, get_default_index

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_EMBEDDINGS_PATH = _PROJECT_ROOT / "data" / "docs" / "embeddings.json"

_LATENT_DIM = 64


@dataclass
class EmbeddingSet:
    ids: list[str] = field(default_factory=list)
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    paths: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    snippets: list[str] = field(default_factory=list)
    levels: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)

    def save(self, path: Path | None = None) -> Path:
        path = path or _DEFAULT_EMBEDDINGS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "ids": self.ids,
            "vectors": self.vectors.tolist(),
            "paths": self.paths,
            "headings": self.headings,
            "snippets": self.snippets,
            "levels": self.levels,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "EmbeddingSet | None":
        path = path or _DEFAULT_EMBEDDINGS_PATH
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            ids=data["ids"],
            vectors=np.array(data["vectors"], dtype=np.float64),
            paths=data["paths"],
            headings=data["headings"],
            snippets=data["snippets"],
            levels=data["levels"],
        )


def build_embeddings(index: DocsIndex | None = None) -> EmbeddingSet:
    """Build embeddings for every section in the docs index."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    idx = index or get_default_index()
    sections = idx.sections
    if not sections:
        return EmbeddingSet()

    texts = [f"{s.heading}\n{s.body}" for s in sections]
    vectorizer = TfidfVectorizer(max_features=4096, stop_words="english")
    tfidf = vectorizer.fit_transform(texts)

    n_components = max(1, min(_LATENT_DIM, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
    if n_components < 1 or tfidf.shape[0] < 2:
        vectors = tfidf.toarray()
    else:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        vectors = svd.fit_transform(tfidf)

    return EmbeddingSet(
        ids=[f"{s.path}#{i}" for i, s in enumerate(sections)],
        vectors=np.asarray(vectors, dtype=np.float64),
        paths=[s.path for s in sections],
        headings=[s.heading for s in sections],
        snippets=[s.body[:280] for s in sections],
        levels=[s.level for s in sections],
    )


def to_payload(emb: EmbeddingSet) -> list[dict[str, Any]]:
    return [
        {
            "id": emb.ids[i],
            "path": emb.paths[i],
            "heading": emb.headings[i],
            "snippet": emb.snippets[i],
            "level": emb.levels[i],
        }
        for i in range(len(emb))
    ]


# ── Module-level cache (mirrors src.agent.docs_index pattern) ───────────────

_default: EmbeddingSet | None = None


def get_default_embeddings(rebuild: bool = False) -> EmbeddingSet:
    global _default
    if _default is None and not rebuild:
        _default = EmbeddingSet.load()
    if _default is None or rebuild:
        _default = build_embeddings()
        _default.save()
    return _default


def reset_default_embeddings() -> None:
    global _default
    _default = None

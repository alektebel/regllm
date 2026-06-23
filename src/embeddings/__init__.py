"""Embedding generation + projection + inspection for the docs/KB corpus."""

from src.embeddings.embedder import EmbeddingSet, get_default_embeddings, reset_default_embeddings
from src.embeddings.projector import PROJECTION_METHODS, project

__all__ = [
    "EmbeddingSet",
    "get_default_embeddings",
    "reset_default_embeddings",
    "PROJECTION_METHODS",
    "project",
]

"""TabBERT-style embeddings for tabular data (recuperatory cycles).

Serializes tabular rows into text using configurable strategies, then embeds
them via the existing nomic-embed-text Ollama service.
"""

from src.embeddings.tabular.serializer import SerializationStrategy, serialize_row
from src.embeddings.tabular.embedder import TabularEmbeddingSet, build_tabular_embeddings

__all__ = [
    "SerializationStrategy",
    "serialize_row",
    "TabularEmbeddingSet",
    "build_tabular_embeddings",
]

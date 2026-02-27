"""
Shared ChromaDB PersistentClient singleton.

Using a single client per database path prevents concurrent SQLite
connections from the same process, which cause "database is locked" errors.
All RAG modules should call get_client() instead of creating their own
PersistentClient instances.
"""

import atexit
import logging
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)

_clients: dict[str, chromadb.ClientAPI] = {}


def get_client(path: str = "./vector_db/chroma_db") -> chromadb.ClientAPI:
    """Return the shared PersistentClient for *path*, creating it on first call."""
    abs_path = str(Path(path).resolve())
    if abs_path not in _clients:
        Path(abs_path).mkdir(parents=True, exist_ok=True)
        _clients[abs_path] = chromadb.PersistentClient(
            path=abs_path,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB client created: {abs_path}")
    return _clients[abs_path]


def close_all() -> None:
    """Close all open ChromaDB clients, flushing SQLite WAL to disk."""
    for path, client in list(_clients.items()):
        try:
            client._system.stop()
            logger.info(f"ChromaDB client closed: {path}")
        except Exception as e:
            logger.debug(f"ChromaDB close warning ({path}): {e}")
    _clients.clear()


atexit.register(close_all)

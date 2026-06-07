"""Local LLM + change-log GraphRAG."""

from .change_log_graph import (
    build_graph,
    graph_to_payload,
    load_graph,
    save_graph,
)
from .llm_client import (
    ChatResponse,
    GemmaClient,        # legacy alias
    LocalLLMClient,
    get_client,
    reset_client,
)
from .graph_rag import (
    FieldJustification,
    GraphRAG,
    GraphRAGReport,
    collect_evidence,
    field_subgraph,
    linearise_subgraph,
)

__all__ = [
    "build_graph", "graph_to_payload", "load_graph", "save_graph",
    "ChatResponse", "LocalLLMClient", "GemmaClient", "get_client", "reset_client",
    "FieldJustification", "GraphRAG", "GraphRAGReport",
    "collect_evidence", "field_subgraph", "linearise_subgraph",
]

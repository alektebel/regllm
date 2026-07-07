"""Local LLM + change-log GraphRAG + regulatory knowledge graph."""

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
from .ontology import (
    EdgeType,
    KGEdge,
    KGNode,
    NodeType,
)

# GraphStore is intentionally NOT imported here: it requires ``kuzu`` (a
# compiled native dependency), and every real caller already imports it from
# the submodule directly (``from src.knowledge.graph_store import
# GraphStore``). Re-exporting it from the package root would force kuzu to be
# installed just to call get_client()/GraphRAG() — which don't use it — and
# is the only thing standing between the DQC deployment and a slim image
# (no torch/kuzu/chromadb/sklearn needed for the /dqc endpoints).

__all__ = [
    "build_graph", "graph_to_payload", "load_graph", "save_graph",
    "ChatResponse", "LocalLLMClient", "GemmaClient", "get_client", "reset_client",
    "FieldJustification", "GraphRAG", "GraphRAGReport",
    "collect_evidence", "field_subgraph", "linearise_subgraph",
    "EdgeType", "KGEdge", "KGNode", "NodeType",
]

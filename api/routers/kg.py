"""Knowledge Graph API endpoints.

Exposes the regulatory knowledge graph for querying, searching,
and ingestion.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kg", tags=["knowledge-graph"])

# Lazy-load to avoid import errors when kuzu isn't installed
_store = None
_experience_store = None


def _get_store():
    global _store
    if _store is None:
        from src.knowledge.graph_store import GraphStore
        _store = GraphStore()
    return _store


def _get_experience_store():
    global _experience_store
    if _experience_store is None:
        from src.knowledge.experience_store import ExperienceStore
        _experience_store = ExperienceStore(_get_store())
    return _experience_store


@router.get("/stats")
def kg_stats() -> dict[str, Any]:
    """Node and edge counts by type."""
    return _get_store().stats()


@router.get("/node/{node_id:path}")
def kg_node(node_id: str) -> dict[str, Any]:
    """Get a node and its immediate neighbors."""
    store = _get_store()
    node = store.get_node(node_id)
    if node is None:
        raise HTTPException(404, f"Node not found: {node_id}")
    neighbors = store.get_neighbors(node_id, hops=1)
    edges = store.get_edges(node_id)
    return {
        "node": node.model_dump(),
        "neighbors": [n.model_dump() for n in neighbors],
        "edges": [e.model_dump() for e in edges],
    }


@router.get("/subgraph")
def kg_subgraph(
    center: str = Query(..., description="Center node ID"),
    hops: int = Query(2, ge=1, le=5),
    types: str | None = Query(None, description="Comma-separated node types to filter"),
) -> dict[str, Any]:
    """Extract a subgraph around a center node."""
    return _get_store().subgraph(center, hops=hops)


@router.get("/path")
def kg_path(
    from_id: str = Query(..., alias="from", description="Source node ID"),
    to_id: str = Query(..., alias="to", description="Target node ID"),
    max_hops: int = Query(5, ge=1, le=10),
) -> dict[str, Any]:
    """Shortest path between two nodes."""
    path = _get_store().shortest_path(from_id, to_id, max_hops=max_hops)
    if path is None:
        return {"found": False, "from": from_id, "to": to_id, "path": None}
    return {"found": True, "from": from_id, "to": to_id, "path": path}


@router.get("/search")
def kg_search(
    q: str = Query(..., description="Search query"),
    types: str | None = Query(None, description="Comma-separated node types"),
    limit: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """Search nodes by label."""
    from src.knowledge.ontology import NodeType
    node_types = None
    if types:
        node_types = [NodeType(t.strip()) for t in types.split(",")]
    results = _get_store().search_nodes(
        node_types=node_types,
        label_contains=q,
        limit=limit,
    )
    return {"query": q, "results": [n.model_dump() for n in results], "count": len(results)}


@router.post("/ingest")
def kg_ingest() -> dict[str, Any]:
    """Trigger full re-ingestion of all knowledge sources (regex-based)."""
    from src.knowledge.ingestors.regulation_ingestor import ingest_regulations
    from src.knowledge.ingestors.schema_ingestor import ingest_schema
    from src.knowledge.ingestors.changelog_ingestor import ingest_changelog

    store = _get_store()
    counts: dict[str, Any] = {}
    counts["regulation"] = ingest_regulations(store)
    counts["schema"] = ingest_schema(store)
    counts["changelog"] = ingest_changelog(store)
    counts["final_stats"] = store.stats()
    return counts


@router.post("/build")
def kg_build_llm(
    regulation_dir: str = "data/regulation",
    schema_file: str | None = "data/samples/irb_schema.sql",
) -> dict[str, Any]:
    """LLM-driven full graph build (qwen3:32b extracts entities + relations)."""
    from src.knowledge.graph_builder import GraphBuilder
    from src.knowledge.vector_store import VectorStore
    from src.knowledge.embeddings import get_embedding_service

    store = _get_store()
    vs = VectorStore()
    embed = get_embedding_service()
    builder = GraphBuilder(store, vs, embed)
    result = builder.build_from_directory(regulation_dir, schema_file)
    result["final_stats"] = store.stats()
    return result


@router.post("/reflect")
def kg_reflect(session_log: str, session_id: str = "") -> dict[str, Any]:
    """Reflect on an agent session — extract insights, quirks, new connections."""
    from src.knowledge.graph_builder import GraphBuilder
    from src.knowledge.vector_store import VectorStore
    from src.knowledge.embeddings import get_embedding_service

    store = _get_store()
    vs = VectorStore()
    embed = get_embedding_service()
    builder = GraphBuilder(store, vs, embed)
    return builder.reflect_on_session(session_log, session_id)


@router.get("/gaps")
def kg_gaps() -> dict[str, Any]:
    """Detect knowledge gaps in the graph (SENATOR pattern)."""
    from src.knowledge.graph_builder import GraphBuilder

    store = _get_store()
    builder = GraphBuilder(store)
    return builder.detect_gaps()


@router.get("/experiences")
def kg_experiences(
    q: str | None = Query(None, description="Optional search query"),
    limit: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """List recent experiences and insights."""
    es = _get_experience_store()
    if q:
        results = es.search(q, k=limit)
    else:
        from src.knowledge.ontology import NodeType
        results = _get_store().search_nodes(
            node_types=[NodeType.EXPERIENCE, NodeType.INSIGHT],
            limit=limit,
        )
    return {"results": [n.model_dump() for n in results], "count": len(results)}


@router.get("/experience/{node_id:path}")
def kg_experience_detail(node_id: str) -> dict[str, Any]:
    """Get an experience/insight with its graph neighborhood."""
    es = _get_experience_store()
    node = _get_store().get_node(node_id)
    if node is None:
        raise HTTPException(404, f"Node not found: {node_id}")
    neighborhood = es.get_related(node_id, hops=2)
    return {
        "node": node.model_dump(),
        "neighborhood": neighborhood,
    }

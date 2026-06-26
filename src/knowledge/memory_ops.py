"""Graph mutation operations for sleep-cycle memory organization."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .graph_entropy import compute_graph_entropy
from .graph_store import GraphStore
from .ontology import Concept, EdgeType, Insight, KGEdge, KGNode, NodeType


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:40] or "theme"


def merge_nodes(store: GraphStore, source: str, target: str) -> dict[str, Any]:
    """Merge ``source`` into ``target``: rewire edges, enrich target, delete source."""
    src = store.get_node(source)
    tgt = store.get_node(target)
    if src is None:
        return {"merged": False, "error": f"source not found: {source}"}
    if tgt is None:
        return {"merged": False, "error": f"target not found: {target}"}
    if source == target:
        return {"merged": False, "error": "source and target are the same"}

    if isinstance(tgt, Insight) and isinstance(src, Insight):
        merged_tags = sorted(set((tgt.tags or []) + (src.tags or [])))
        note = f" [merged from {source}]"
        summary = tgt.summary or ""
        if src.summary and src.summary not in summary:
            summary = f"{summary}{note}: {src.summary}"[:500]
        store.add_node(Insight(
            id=tgt.id,
            label=tgt.label,
            summary=summary,
            tags=merged_tags,
            priority=max(tgt.priority, src.priority),
            feedback_type=tgt.feedback_type or src.feedback_type,
            original_claim=tgt.original_claim or src.original_claim,
            corrected_understanding=tgt.corrected_understanding or src.corrected_understanding,
        ))

    for edge in store.get_edges(source, direction="out"):
        if edge.dst_id != target:
            store.add_edge(KGEdge(
                src_id=target,
                dst_id=edge.dst_id,
                edge_type=edge.edge_type,
                confidence=edge.confidence,
                metadata=edge.metadata,
            ))
    for edge in store.get_edges(source, direction="in"):
        if edge.src_id != target:
            store.add_edge(KGEdge(
                src_id=edge.src_id,
                dst_id=target,
                edge_type=edge.edge_type,
                confidence=edge.confidence,
                metadata=edge.metadata,
            ))

    store.delete_node(source)
    return {"merged": True, "source": source, "target": target}


def add_tags(store: GraphStore, node: str, tags: list[str]) -> dict[str, Any]:
    n = store.get_node(node)
    if n is None:
        return {"updated": False, "error": f"node not found: {node}"}
    if not isinstance(n, Insight):
        return {"updated": False, "error": f"tags only supported on Insight nodes, got {n.node_type}"}
    merged = sorted(set((n.tags or []) + tags))
    store.add_node(Insight(
        id=n.id,
        label=n.label,
        summary=n.summary,
        tags=merged,
        priority=n.priority,
        feedback_type=n.feedback_type,
        original_claim=n.original_claim,
        corrected_understanding=n.corrected_understanding,
    ))
    return {"updated": True, "node": node, "tags": merged}


def link_nodes(
    store: GraphStore,
    source: str,
    target: str,
    relation: str = "RELATES_TO",
) -> dict[str, Any]:
    if store.get_node(source) is None:
        return {"linked": False, "error": f"source not found: {source}"}
    if store.get_node(target) is None:
        return {"linked": False, "error": f"target not found: {target}"}
    try:
        edge_type = EdgeType(relation)
    except ValueError:
        edge_type = EdgeType.RELATES_TO
    store.add_edge(KGEdge(src_id=source, dst_id=target, edge_type=edge_type))
    return {"linked": True, "source": source, "target": target, "relation": edge_type.value}


def flag_conflict(store: GraphStore, node_a: str, node_b: str, reason: str = "") -> dict[str, Any]:
    if store.get_node(node_a) is None or store.get_node(node_b) is None:
        return {"flagged": False, "error": "one or both nodes not found"}
    store.add_edge(KGEdge(
        src_id=node_a,
        dst_id=node_b,
        edge_type=EdgeType.NEEDS_HUMAN_REVIEW,
        metadata={"reason": reason or "conflicting insights"},
    ))
    return {"flagged": True, "node_a": node_a, "node_b": node_b, "reason": reason}


def create_concept(
    store: GraphStore,
    label: str,
    definition: str,
    covers: list[str],
    links_to: list[str] | None = None,
    concept_id: str | None = None,
) -> dict[str, Any]:
    """Create an experience-derived Concept that compresses related insights."""
    if len(covers) < 2:
        return {
            "created": False,
            "error": "create_concept requires at least 2 insights (no compression otherwise)",
        }

    slug = _slug(label)
    cid = concept_id or f"concept:mem_{slug}_{hashlib.sha256(label.encode()).hexdigest()[:8]}"
    if store.get_node(cid) is not None:
        cid = f"{cid}_{hashlib.sha256(definition.encode()).hexdigest()[:6]}"

    before_nodes = _experience_subgraph_nodes(store, covers)
    entropy_before = compute_graph_entropy(before_nodes).total
    # Virtual consolidation: measure redundancy/conflict relief without penalizing
    # the extra concept node in the structure term.
    entropy_after_virtual = compute_graph_entropy(
        before_nodes, concept_covers={cid: covers},
    ).total

    store.add_node(Concept(id=cid, label=label, definition=definition))

    linked_insights = 0
    for ins_id in covers:
        if store.get_node(ins_id) is None:
            continue
        store.add_edge(KGEdge(src_id=cid, dst_id=ins_id, edge_type=EdgeType.CONTAINS))
        linked_insights += 1

    field_links = 0
    for ref in links_to or []:
        upper = ref.upper().replace("CONCEPT:", "").replace("COL:", "")
        for candidate in (f"col:{upper}", f"concept:{upper}", ref):
            if store.get_node(candidate):
                store.add_edge(KGEdge(src_id=cid, dst_id=candidate, edge_type=EdgeType.RELATES_TO))
                field_links += 1
                break

    delta = round(entropy_before - entropy_after_virtual, 4)

    return {
        "created": True,
        "id": cid,
        "label": label,
        "covers_linked": linked_insights,
        "field_links": field_links,
        "entropy_before": entropy_before,
        "entropy_delta": delta,
    }


def link_to_concept(store: GraphStore, insight_id: str, concept_id: str) -> dict[str, Any]:
    if store.get_node(insight_id) is None:
        return {"linked": False, "error": f"insight not found: {insight_id}"}
    concept = store.get_node(concept_id)
    if concept is None or concept.node_type != NodeType.CONCEPT:
        return {"linked": False, "error": f"concept not found: {concept_id}"}
    store.add_edge(KGEdge(src_id=concept_id, dst_id=insight_id, edge_type=EdgeType.CONTAINS))
    return {"linked": True, "insight_id": insight_id, "concept_id": concept_id}


def _experience_subgraph_nodes(store: GraphStore, insight_ids: list[str]) -> list[KGNode]:
    nodes: list[KGNode] = []
    for iid in insight_ids:
        n = store.get_node(iid)
        if n is not None:
            nodes.append(n)
    return nodes

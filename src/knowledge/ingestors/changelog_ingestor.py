"""Ingest changelog markdown into the knowledge graph.

Migrates the existing ``change_log_graph.py`` NetworkX-based ingestion
to write into KuzuDB instead, reusing the same parsing functions.

Usage::

    python -m src.knowledge.ingestors.changelog_ingestor
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..change_log_graph import _extract_fields, _extract_renames, _split_markdown
from ..graph_store import GraphStore
from ..ontology import (
    Concept,
    EdgeType,
    KGEdge,
    KGNode,
    NodeType,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_CHANGELOG_DIR = _PROJECT_ROOT / "data" / "changelog"


class _DocNode(KGNode):
    """Lightweight node for changelog documents."""
    node_type: NodeType = NodeType.INSIGHT  # reuse Insight type for docs
    source: str = ""


def ingest_changelog(
    store: GraphStore,
    changelog_dir: Path | None = None,
) -> dict[str, int]:
    """Ingest changelog markdown into the knowledge graph.

    Returns counts of created nodes and edges.
    """
    cdir = changelog_dir or _DEFAULT_CHANGELOG_DIR
    counts = {"nodes_created": 0, "edges_created": 0}

    if not cdir.is_dir():
        logger.warning("Changelog dir not found: %s", cdir)
        return counts

    for md_path in sorted(cdir.glob("*.md")):
        _ingest_one(md_path, store, counts)

    logger.info(
        "Changelog ingestion complete: %d nodes, %d edges",
        counts["nodes_created"], counts["edges_created"],
    )
    return counts


def _ingest_one(
    md_path: Path,
    store: GraphStore,
    counts: dict[str, int],
) -> None:
    """Ingest a single changelog markdown file."""
    text = md_path.read_text(encoding="utf-8")
    stem = md_path.stem

    # Document node
    doc_id = f"changelog:{stem}"
    store.add_node(KGNode(
        id=doc_id,
        node_type=NodeType.INSIGHT,
        label=stem,
    ))
    counts["nodes_created"] += 1

    sections = _split_markdown(text)
    for idx, (heading, level, body) in enumerate(sections):
        if not body and heading == "(intro)":
            continue

        # Section as a lightweight node
        sec_id = f"changelog:{stem}:s{idx}"
        store.add_node(KGNode(
            id=sec_id,
            node_type=NodeType.INSIGHT,
            label=heading,
        ))
        counts["nodes_created"] += 1

        store.add_edge(KGEdge(
            src_id=doc_id,
            dst_id=sec_id,
            edge_type=EdgeType.CONTAINS,
        ))
        counts["edges_created"] += 1

        # Field mentions → link to concepts
        full_text = f"{heading}\n{body}"
        fields = _extract_fields(full_text)
        for field_name in fields:
            concept_id = f"concept:{field_name}"
            store.add_node(Concept(
                id=concept_id,
                label=field_name,
            ))
            counts["nodes_created"] += 1

            store.add_edge(KGEdge(
                src_id=sec_id,
                dst_id=concept_id,
                edge_type=EdgeType.MENTIONS_FIELD,
            ))
            counts["edges_created"] += 1

        # Rename edges
        for src_field, dst_field in _extract_renames(full_text):
            for f in (src_field, dst_field):
                store.add_node(Concept(id=f"concept:{f}", label=f))
            store.add_edge(KGEdge(
                src_id=f"concept:{src_field}",
                dst_id=f"concept:{dst_field}",
                edge_type=EdgeType.SUPERSEDES,
            ))
            counts["edges_created"] += 1


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = GraphStore()
    counts = ingest_changelog(store)
    print(f"Done: {counts}")

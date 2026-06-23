"""Ingest regulation markdown into the knowledge graph.

Reads ``data/regulation/*.md`` and ``data/docs/regulation/*.md``,
creates ``Regulation`` nodes per article, ``Concept`` nodes for defined
terms (uppercase field references), and typed edges between them.

Also embeds each section into the vector store for semantic search.

Usage::

    python -m src.knowledge.ingestors.regulation_ingestor
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ..change_log_graph import _extract_fields, _split_markdown
from ..embeddings import EmbeddingService, get_embedding_service
from ..graph_store import GraphStore
from ..ontology import (
    Concept,
    EdgeType,
    KGEdge,
    Regulation,
)
from ..vector_store import VectorStore

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_REGULATION_DIRS = [
    _PROJECT_ROOT / "data" / "regulation",
    _PROJECT_ROOT / "data" / "docs" / "regulation",
]

# Patterns for cross-references: "artículo X", "Article X", "Art. X"
_ARTICLE_REF_RE = re.compile(
    r"(?:art[íi]culo|article|art\.?)\s+(\d+)",
    re.IGNORECASE,
)


def ingest_regulations(
    store: GraphStore,
    vector_store: VectorStore | None = None,
    embed_service: EmbeddingService | None = None,
    regulation_dirs: list[Path] | None = None,
) -> dict[str, int]:
    """Ingest all regulation markdown into the knowledge graph.

    Returns counts: {nodes_created, edges_created, sections_embedded}.
    """
    dirs = regulation_dirs or _REGULATION_DIRS
    embed = embed_service or get_embedding_service()

    counts = {"nodes_created": 0, "edges_created": 0, "sections_embedded": 0}
    article_ids: dict[str, str] = {}  # article_number -> node_id

    for reg_dir in dirs:
        if not reg_dir.is_dir():
            continue
        for md_path in sorted(reg_dir.glob("*.md")):
            _ingest_one(md_path, store, vector_store, embed, counts, article_ids)

    # Cross-reference edges between articles
    _link_cross_references(store, article_ids, counts)

    logger.info(
        "Regulation ingestion complete: %d nodes, %d edges, %d embedded",
        counts["nodes_created"], counts["edges_created"], counts["sections_embedded"],
    )
    return counts


def _ingest_one(
    md_path: Path,
    store: GraphStore,
    vector_store: VectorStore | None,
    embed: EmbeddingService,
    counts: dict[str, int],
    article_ids: dict[str, str],
) -> None:
    """Ingest a single regulation markdown file."""
    text = md_path.read_text(encoding="utf-8")
    stem = md_path.stem  # e.g. "art_15_dotaciones_minimas"

    # Extract article number from filename
    art_match = re.search(r"art[_\s]*(\d+)", stem, re.IGNORECASE)
    article_num = art_match.group(1) if art_match else stem

    # Create Regulation node
    sections = _split_markdown(text)
    title = sections[0][0] if sections else stem
    if title == "(intro)" and len(sections) > 1:
        title = sections[1][0]

    reg_id = f"reg:{stem}"
    store.add_node(Regulation(
        id=reg_id,
        label=title,
        article_id=f"art{article_num}",
        source_document=md_path.name,
        jurisdiction="ES",
    ))
    counts["nodes_created"] += 1
    article_ids[article_num] = reg_id

    # Process each section
    for idx, (heading, level, body) in enumerate(sections):
        if not body and heading == "(intro)":
            continue

        full_text = f"{heading}\n{body}"

        # Extract field/concept mentions
        fields = _extract_fields(full_text)
        for field_name in fields:
            concept_id = f"concept:{field_name}"
            store.add_node(Concept(
                id=concept_id,
                label=field_name,
                definition="",
            ))
            counts["nodes_created"] += 1

            # Regulation DEFINITION_IN concept
            store.add_edge(KGEdge(
                src_id=concept_id,
                dst_id=reg_id,
                edge_type=EdgeType.DEFINITION_IN,
            ))
            counts["edges_created"] += 1

        # Embed section into vector store
        if vector_store and embed.available:
            doc_id = f"reg:{stem}:s{idx}"
            embedding = embed.embed_one(full_text[:2000])
            vector_store.upsert(
                doc_id=doc_id,
                text=full_text[:2000],
                embedding=embedding,
                metadata={
                    "source": md_path.name,
                    "heading": heading,
                    "article_id": f"art{article_num}",
                    "node_type": "Regulation",
                    "section_idx": idx,
                },
            )
            counts["sections_embedded"] += 1


def _link_cross_references(
    store: GraphStore,
    article_ids: dict[str, str],
    counts: dict[str, int],
) -> None:
    """Find cross-references between articles and create SUBJECT_TO edges."""
    for art_num, reg_id in article_ids.items():
        node = store.get_node(reg_id)
        if node is None:
            continue
        # Get the source text by reading the props
        # We'll search for references to other articles in all sections
        # that mention this regulation
        neighbors = store.get_neighbors(reg_id, hops=1)
        for neighbor in neighbors:
            if hasattr(neighbor, "definition"):
                # Check concept labels for article references
                pass

    # Instead, do a simpler approach: check all regulation nodes' source files
    for art_num, reg_id in article_ids.items():
        node = store.get_node(reg_id)
        if node is None or not isinstance(node, Regulation):
            continue
        source_file = node.source_document
        for reg_dir in _REGULATION_DIRS:
            src_path = reg_dir / source_file
            if src_path.exists():
                text = src_path.read_text(encoding="utf-8")
                for ref_match in _ARTICLE_REF_RE.finditer(text):
                    ref_num = ref_match.group(1)
                    if ref_num != art_num and ref_num in article_ids:
                        store.add_edge(KGEdge(
                            src_id=reg_id,
                            dst_id=article_ids[ref_num],
                            edge_type=EdgeType.SUBJECT_TO,
                        ))
                        counts["edges_created"] += 1
                break


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    store = GraphStore()
    counts = ingest_regulations(store)
    print(f"Done: {counts}")

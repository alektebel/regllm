"""Ingest database schema (SQL DDL + audit mapping) into the knowledge graph.

Creates ``Table`` and ``Column`` nodes, then links them to ``Regulation``
nodes via ``GOVERNS``/``IMPLEMENTS`` edges using confidence scores from
the audit variable mapper.

Usage::

    python -m src.knowledge.ingestors.schema_ingestor
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..graph_store import GraphStore
from ..ontology import (
    Column,
    EdgeType,
    KGEdge,
    NodeType,
    Table,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SCHEMA_FILE = _PROJECT_ROOT / "data" / "samples" / "irb_schema.sql"
_MAPPING_FILE = _PROJECT_ROOT / "data" / "audit" / "mapping.json"

# SQL CREATE TABLE regex
_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*?)\);",
    re.IGNORECASE | re.DOTALL,
)
_COLUMN_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z]+(?:\([^)]*\))?)")
_SQL_KEYWORDS = {
    "PRIMARY", "FOREIGN", "KEY", "REFERENCES", "CONSTRAINT",
    "UNIQUE", "INDEX", "CHECK", "DEFAULT", "NOT", "NULL",
}


def ingest_schema(
    store: GraphStore,
    schema_file: Path | None = None,
    mapping_file: Path | None = None,
) -> dict[str, int]:
    """Ingest schema DDL and audit mapping into the graph.

    Returns counts of created nodes and edges.
    """
    sf = schema_file or _SCHEMA_FILE
    mf = mapping_file or _MAPPING_FILE

    counts = {"nodes_created": 0, "edges_created": 0}

    # 1. Parse DDL
    if sf.exists():
        _ingest_ddl(store, sf, counts)
    else:
        logger.warning("Schema file not found: %s", sf)

    # 2. Load audit mapping and create regulation → column edges
    if mf.exists():
        _ingest_mapping(store, mf, counts)
    else:
        logger.warning("Mapping file not found: %s", mf)

    logger.info(
        "Schema ingestion complete: %d nodes, %d edges",
        counts["nodes_created"], counts["edges_created"],
    )
    return counts


def _ingest_ddl(store: GraphStore, path: Path, counts: dict[str, int]) -> None:
    """Parse SQL DDL and create Table/Column nodes."""
    text = path.read_text(encoding="utf-8")

    for tm in _CREATE_TABLE_RE.finditer(text):
        table_name = tm.group(1)
        body = tm.group(2)

        # Create Table node
        table_id = f"table:{table_name.lower()}"
        store.add_node(Table(
            id=table_id,
            label=table_name,
            schema_name=table_name.split(".")[0] if "." in table_name else "",
        ))
        counts["nodes_created"] += 1

        # Parse columns
        for line in body.split(","):
            cm = _COLUMN_RE.match(line)
            if not cm:
                continue
            col_name = cm.group(1).upper()
            col_type = cm.group(2)
            if col_name in _SQL_KEYWORDS:
                continue

            col_id = f"col:{col_name}"
            nullable = "NOT NULL" not in line.upper()

            store.add_node(Column(
                id=col_id,
                label=col_name,
                table_name=table_name,
                data_type=col_type,
                nullable=nullable,
            ))
            counts["nodes_created"] += 1

            # Table CONTAINS column
            store.add_edge(KGEdge(
                src_id=table_id,
                dst_id=col_id,
                edge_type=EdgeType.CONTAINS,
            ))
            counts["edges_created"] += 1


def _ingest_mapping(store: GraphStore, path: Path, counts: dict[str, int]) -> None:
    """Load audit mapping and create regulation → column edges."""
    raw = json.loads(path.read_text(encoding="utf-8"))

    for col_name, entry in raw.items():
        col_id = f"col:{col_name}"

        # Ensure column node exists
        if store.get_node(col_id) is None:
            store.add_node(Column(
                id=col_id,
                label=col_name,
                table_name=entry.get("table_name", ""),
                data_type="",
            ))
            counts["nodes_created"] += 1

        confidence = entry.get("confidence", 0.5)

        # Link to regulation articles
        for article_ref in entry.get("articles", []):
            # Find matching regulation node
            reg_nodes = store.search_nodes(
                node_types=[NodeType.REGULATION],
                label_contains=article_ref.replace("_", " "),
            )
            # Also try by article_id pattern
            stem = article_ref.replace(".md", "")
            reg_id = f"reg:{stem}"
            reg_node = store.get_node(reg_id)

            if reg_node:
                store.add_edge(KGEdge(
                    src_id=reg_id,
                    dst_id=col_id,
                    edge_type=EdgeType.GOVERNS,
                    confidence=confidence,
                    metadata={
                        "regulation_concept": entry.get("regulation_concept", ""),
                        "regulation_description": entry.get("regulation_description", ""),
                    },
                ))
                counts["edges_created"] += 1
            elif reg_nodes:
                store.add_edge(KGEdge(
                    src_id=reg_nodes[0].id,
                    dst_id=col_id,
                    edge_type=EdgeType.GOVERNS,
                    confidence=confidence,
                ))
                counts["edges_created"] += 1

        # Link column to concepts with same name
        concept_id = f"concept:{col_name}"
        concept = store.get_node(concept_id)
        if concept:
            store.add_edge(KGEdge(
                src_id=col_id,
                dst_id=concept_id,
                edge_type=EdgeType.IMPLEMENTS,
                confidence=confidence,
            ))
            counts["edges_created"] += 1

        # Flag low-confidence mappings
        if entry.get("needs_review") or confidence < 0.70:
            store.add_edge(KGEdge(
                src_id=col_id,
                dst_id=col_id,
                edge_type=EdgeType.NEEDS_HUMAN_REVIEW,
                confidence=confidence,
                metadata={"reason": "low_confidence_mapping"},
            ))
            counts["edges_created"] += 1


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = GraphStore()
    counts = ingest_schema(store)
    print(f"Done: {counts}")

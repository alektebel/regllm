"""
Structural hierarchy extractor.

Reads chunk metadata fields (titulo, capitulo, seccion, articulo, documento_id)
and builds the document_hierarchy tree, then links each chunk to its node.
"""

from __future__ import annotations
import logging
import re
import psycopg2
import psycopg2.extras

log = logging.getLogger(__name__)

# Maps documento_id prefix → human-readable source name (populated from metadata)
# Falls back to documento_id if not found.


def _slug(text: str) -> str:
    """Lower-case alphanumeric slug from text, trimmed to 40 chars."""
    s = re.sub(r"[^a-z0-9]+", "_", text.lower().strip()).strip("_")
    return s[:40]


def _node_id(doc_id: str, level: str, label: str) -> str:
    return f"{doc_id}__{level}_{_slug(label)}"


def _doc_node_id(doc_id: str) -> str:
    return f"{doc_id}__doc"


class HierarchyExtractor:
    """
    Builds document_hierarchy rows from document_chunks.metadata and
    updates document_chunks.hierarchy_node_id.
    """

    def __init__(self, dsn: str):
        self.dsn = dsn

    # ── public entry point ───────────────────────────────────────────────────

    def build(self) -> dict[str, int]:
        """
        Full build:
        1. Read all chunks and their metadata.
        2. Derive hierarchy nodes (doc → titulo → capitulo → seccion → article).
        3. Insert nodes into document_hierarchy.
        4. Update document_chunks.hierarchy_node_id.

        Returns counts: {"nodes": N, "links": N}
        """
        conn = psycopg2.connect(self.dsn)
        try:
            chunks = self._load_chunks(conn)
            nodes, chunk_node_map = self._derive_nodes(chunks)
            n_nodes = self._upsert_nodes(conn, nodes)
            n_links = self._link_chunks(conn, chunk_node_map)
            self._update_article_counts(conn)
            return {"nodes": n_nodes, "links": n_links}
        finally:
            conn.close()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _load_chunks(self, conn) -> list[dict]:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT chunk_id, metadata FROM document_chunks")
            return cur.fetchall()

    def _derive_nodes(self, chunks: list[dict]):
        """
        Walk chunks, extract hierarchy levels from metadata, build node dicts.
        Returns (nodes_dict keyed by node_id, chunk→leaf_node_id map).
        """
        nodes: dict[str, dict] = {}  # node_id → row dict
        chunk_node_map: dict[str, str] = {}  # chunk_id → leaf node_id

        for row in chunks:
            meta = row["metadata"] or {}
            chunk_id: str = row["chunk_id"]
            doc_id: str = meta.get("documento_id") or chunk_id.rsplit("_", 1)[0]
            source: str = meta.get("source", doc_id)

            # Level values (empty string means absent)
            titulo   = (meta.get("titulo")   or "").strip()
            capitulo = (meta.get("capitulo") or "").strip()
            seccion  = (meta.get("seccion")  or "").strip()
            articulo = (meta.get("articulo") or "").strip()

            # ── document node ────────────────────────────────────────────────
            doc_nid = _doc_node_id(doc_id)
            if doc_nid not in nodes:
                nodes[doc_nid] = {
                    "node_id": doc_nid,
                    "doc_id": doc_id,
                    "level_type": "document",
                    "number": None,
                    "label": source,
                    "parent_id": None,
                    "position": 0,
                }

            parent_id = doc_nid
            position_counters: dict[str, int] = {}

            # ── titulo ───────────────────────────────────────────────────────
            if titulo:
                nid = _node_id(doc_id, "title", titulo)
                if nid not in nodes:
                    nodes[nid] = {
                        "node_id": nid,
                        "doc_id": doc_id,
                        "level_type": "part",
                        "number": _extract_number(titulo),
                        "label": titulo[:200],
                        "parent_id": doc_nid,
                        "position": len([n for n in nodes.values()
                                         if n["parent_id"] == doc_nid]),
                    }
                parent_id = nid

            # ── capitulo ─────────────────────────────────────────────────────
            if capitulo:
                nid = _node_id(doc_id, "chapter", capitulo)
                if nid not in nodes:
                    nodes[nid] = {
                        "node_id": nid,
                        "doc_id": doc_id,
                        "level_type": "chapter",
                        "number": _extract_number(capitulo),
                        "label": capitulo[:200],
                        "parent_id": parent_id,
                        "position": len([n for n in nodes.values()
                                         if n["parent_id"] == parent_id]),
                    }
                parent_id = nid

            # ── seccion ──────────────────────────────────────────────────────
            if seccion:
                nid = _node_id(doc_id, "section", seccion)
                if nid not in nodes:
                    nodes[nid] = {
                        "node_id": nid,
                        "doc_id": doc_id,
                        "level_type": "section",
                        "number": _extract_number(seccion),
                        "label": seccion[:200],
                        "parent_id": parent_id,
                        "position": len([n for n in nodes.values()
                                         if n["parent_id"] == parent_id]),
                    }
                parent_id = nid

            # ── article ──────────────────────────────────────────────────────
            if articulo:
                art_title = meta.get("art_title", "")
                label = f"{articulo} {art_title}".strip() if art_title else articulo
                nid = _node_id(doc_id, "article", articulo)
                if nid not in nodes:
                    nodes[nid] = {
                        "node_id": nid,
                        "doc_id": doc_id,
                        "level_type": "article",
                        "number": _extract_number(articulo),
                        "label": label[:200],
                        "parent_id": parent_id,
                        "position": len([n for n in nodes.values()
                                         if n["parent_id"] == parent_id]),
                    }
                chunk_node_map[chunk_id] = nid
            else:
                # Link chunk to deepest available node
                chunk_node_map[chunk_id] = parent_id

        return nodes, chunk_node_map

    def _upsert_nodes(self, conn, nodes: dict) -> int:
        inserted = 0
        with conn, conn.cursor() as cur:
            for node in nodes.values():
                cur.execute(
                    """
                    INSERT INTO document_hierarchy
                        (node_id, doc_id, level_type, number, label, parent_id, position)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (node_id) DO UPDATE
                        SET label = EXCLUDED.label,
                            parent_id = EXCLUDED.parent_id
                    """,
                    (
                        node["node_id"], node["doc_id"], node["level_type"],
                        node["number"], node["label"],
                        node["parent_id"], node["position"],
                    ),
                )
                inserted += 1
        log.info("Upserted %d hierarchy nodes", inserted)
        return inserted

    def _link_chunks(self, conn, chunk_node_map: dict[str, str]) -> int:
        linked = 0
        with conn, conn.cursor() as cur:
            for chunk_id, node_id in chunk_node_map.items():
                cur.execute(
                    "UPDATE document_chunks SET hierarchy_node_id = %s WHERE chunk_id = %s",
                    (node_id, chunk_id),
                )
                linked += 1
        conn.commit()
        log.info("Linked %d chunks to hierarchy nodes", linked)
        return linked

    def _update_article_counts(self, conn) -> None:
        """Update article_count for each node = number of descendant article nodes."""
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE document_hierarchy dh
                SET article_count = sub.cnt
                FROM (
                    SELECT parent_id, COUNT(*) AS cnt
                    FROM document_hierarchy
                    WHERE level_type = 'article'
                    GROUP BY parent_id
                ) sub
                WHERE dh.node_id = sub.parent_id
                """
            )
        conn.commit()


def _extract_number(label: str) -> str | None:
    """Pull the first number/roman-numeral token from a label string."""
    m = re.search(r"\b([IVXLCDM]+|\d+[\w.]*)\b", label, re.IGNORECASE)
    return m.group(1) if m else None

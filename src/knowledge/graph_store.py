"""KuzuDB-backed knowledge graph store.

Provides CRUD and Cypher query access over the regulatory knowledge graph.
Uses a single ``Entity`` node table with a ``node_type`` discriminator and
a JSON ``props`` column for type-specific fields.  This keeps the schema
simple and avoids KuzuDB's ALTER TABLE limitations.

Usage::

    store = GraphStore("data/knowledge/regllm.kuzu")
    store.add_node(Regulation(id="reg_art15", label="Article 15", article_id="art15"))
    store.add_node(Column(id="col_lgd", label="LGD_ESTIMADA", table_name="cycles"))
    store.add_edge(KGEdge(src_id="reg_art15", dst_id="col_lgd", edge_type=EdgeType.GOVERNS))
    neighbors = store.get_neighbors("reg_art15", hops=2)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import kuzu

from .ontology import (
    NODE_CLASSES,
    EdgeType,
    KGEdge,
    KGNode,
    NodeType,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


class GraphStore:
    """Thin wrapper around a KuzuDB database for the regulatory KG."""

    def __init__(self, db_path: str | Path = "data/knowledge/regllm.kuzu") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(str(self._db_path))
        self._conn = kuzu.Connection(self._db)
        self._ensure_schema()

    # ── Schema management ────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        """Create node/edge tables if they don't exist."""
        existing = self._list_tables()
        if "Entity" not in existing:
            self._conn.execute(
                "CREATE NODE TABLE Entity("
                "  id STRING,"
                "  node_type STRING,"
                "  label STRING,"
                "  props STRING,"
                "  PRIMARY KEY (id)"
                ")"
            )
            logger.info("Created Entity node table")
        if "Edge" not in existing:
            self._conn.execute(
                "CREATE REL TABLE Edge("
                "  FROM Entity TO Entity,"
                "  edge_type STRING,"
                "  confidence DOUBLE,"
                "  props STRING"
                ")"
            )
            logger.info("Created Edge rel table")
        if "Meta" not in existing:
            self._conn.execute(
                "CREATE NODE TABLE Meta("
                "  key STRING,"
                "  value STRING,"
                "  PRIMARY KEY (key)"
                ")"
            )
            self._conn.execute(
                f"CREATE (m:Meta {{key: 'schema_version', value: '{_SCHEMA_VERSION}'}})"
            )

    def _list_tables(self) -> set[str]:
        result = self._conn.execute("CALL show_tables() RETURN *")
        names: set[str] = set()
        while result.has_next():
            row = result.get_next()
            names.add(row[1])  # name is second column
        return names

    @property
    def schema_version(self) -> int:
        result = self._conn.execute(
            "MATCH (m:Meta {key: 'schema_version'}) RETURN m.value"
        )
        if result.has_next():
            return int(result.get_next()[0])
        return 0

    # ── Node CRUD ────────────────────────────────────────────────────────

    def add_node(self, node: KGNode) -> None:
        """Insert or update a node."""
        params = {
            "id": node.id,
            "node_type": node.node_type.value,
            "label": node.label,
            "props": node.props_json(),
        }
        self._conn.execute(
            "MERGE (e:Entity {id: $id})"
            " ON CREATE SET e.node_type = $node_type,"
            "   e.label = $label,"
            "   e.props = $props"
            " ON MATCH SET e.node_type = $node_type,"
            "   e.label = $label,"
            "   e.props = $props",
            params,
        )

    def add_nodes(self, nodes: list[KGNode]) -> None:
        """Batch insert nodes."""
        for node in nodes:
            self.add_node(node)

    def get_node(self, node_id: str) -> KGNode | None:
        """Retrieve a node by id, returns typed Pydantic model."""
        result = self._conn.execute(
            "MATCH (e:Entity {id: $id}) "
            "RETURN e.id, e.node_type, e.label, e.props",
            {"id": node_id},
        )
        if not result.has_next():
            return None
        row = result.get_next()
        return self._row_to_node(row)

    def delete_node(self, node_id: str) -> None:
        """Delete a node and all its edges."""
        self._conn.execute(
            "MATCH (e:Entity {id: $id}) DETACH DELETE e",
            {"id": node_id},
        )

    def search_nodes(
        self,
        node_types: list[NodeType] | None = None,
        label_contains: str | None = None,
        limit: int = 50,
    ) -> list[KGNode]:
        """Search nodes by type and/or label substring."""
        clauses: list[str] = []
        if node_types:
            types_str = ", ".join(f"'{t.value}'" for t in node_types)
            clauses.append(f"e.node_type IN [{types_str}]")
        if label_contains:
            esc = label_contains.replace("'", "''")
            clauses.append(f"e.label CONTAINS '{esc}'")

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"MATCH (e:Entity){where} RETURN e.id, e.node_type, e.label, e.props LIMIT {limit}"
        return self._query_nodes(query)

    # ── Edge CRUD ────────────────────────────────────────────────────────

    def add_edge(self, edge: KGEdge) -> None:
        """Insert an edge between two existing nodes."""
        params = {
            "src": edge.src_id,
            "dst": edge.dst_id,
            "edge_type": edge.edge_type.value,
            "confidence": edge.confidence if edge.confidence is not None else 1.0,
            "props": edge.props_json(),
        }
        self._conn.execute(
            "MATCH (a:Entity {id: $src}), (b:Entity {id: $dst}) "
            "CREATE (a)-[:Edge {"
            "  edge_type: $edge_type,"
            "  confidence: $confidence,"
            "  props: $props"
            "}]->(b)",
            params,
        )

    def add_edges(self, edges: list[KGEdge]) -> None:
        for edge in edges:
            self.add_edge(edge)

    def get_edges(
        self,
        node_id: str,
        edge_types: list[EdgeType] | None = None,
        direction: str = "both",
    ) -> list[KGEdge]:
        """Get edges connected to a node."""
        edges: list[KGEdge] = []
        if direction in ("out", "both"):
            edges.extend(self._get_edges_directional(node_id, edge_types, "out"))
        if direction in ("in", "both"):
            edges.extend(self._get_edges_directional(node_id, edge_types, "in"))
        return edges

    def _get_edges_directional(
        self,
        node_id: str,
        edge_types: list[EdgeType] | None,
        direction: str,
    ) -> list[KGEdge]:
        ret = "a.id, b.id, r.edge_type, r.confidence, r.props"
        if direction == "out":
            pattern = "(a:Entity {id: $nid})-[r:Edge]->(b:Entity)"
        else:
            pattern = "(a:Entity)-[r:Edge]->(b:Entity {id: $nid})"

        where = ""
        if edge_types:
            types_str = ", ".join(f"'{t.value}'" for t in edge_types)
            where = f" WHERE r.edge_type IN [{types_str}]"

        query = f"MATCH {pattern}{where} RETURN {ret}"
        result = self._conn.execute(query, {"nid": node_id})
        edges: list[KGEdge] = []
        while result.has_next():
            row = result.get_next()
            props = json.loads(row[4]) if row[4] else {}
            meta = json.loads(props.get("metadata", "{}")) if isinstance(props.get("metadata"), str) else props.get("metadata", {})
            edges.append(KGEdge(
                src_id=row[0],
                dst_id=row[1],
                edge_type=EdgeType(row[2]),
                confidence=row[3],
                metadata=meta,
            ))
        return edges

    # ── Graph traversal ──────────────────────────────────────────────────

    def get_neighbors(
        self,
        node_id: str,
        hops: int = 1,
        edge_types: list[EdgeType] | None = None,
    ) -> list[KGNode]:
        """Get nodes within ``hops`` of ``node_id``."""
        where = ""
        if edge_types:
            types_str = ", ".join(f"'{t.value}'" for t in edge_types)
            where = f" WHERE r.edge_type IN [{types_str}]"

        query = (
            f"MATCH (a:Entity {{id: $nid}})-[r:Edge*1..{hops}]-(b:Entity)"
            f"{where}"
            " RETURN DISTINCT b.id, b.node_type, b.label, b.props"
        )
        return self._query_nodes(query, {"nid": node_id})

    def subgraph(
        self,
        center_id: str,
        hops: int = 2,
        edge_types: list[EdgeType] | None = None,
    ) -> dict[str, Any]:
        """Extract a subgraph as {nodes: [...], edges: [...]} dicts."""
        center = self.get_node(center_id)
        if center is None:
            return {"nodes": [], "edges": []}

        neighbors = self.get_neighbors(center_id, hops=hops, edge_types=edge_types)
        all_nodes = [center] + neighbors
        node_ids = {n.id for n in all_nodes}

        # Collect edges between subgraph nodes
        all_edges: list[KGEdge] = []
        for nid in node_ids:
            for edge in self.get_edges(nid, edge_types=edge_types, direction="out"):
                if edge.dst_id in node_ids:
                    all_edges.append(edge)

        return {
            "nodes": [n.model_dump() for n in all_nodes],
            "edges": [e.model_dump() for e in all_edges],
        }

    def shortest_path(
        self,
        from_id: str,
        to_id: str,
        max_hops: int = 5,
    ) -> list[dict[str, Any]] | None:
        """Find shortest path between two nodes. Returns list of
        alternating node/edge dicts, or None if no path exists."""
        # KuzuDB uses "* SHORTEST 1..N" syntax for shortest path
        query = (
            f"MATCH p = (a:Entity {{id: $from}})"
            f"-[r:Edge* SHORTEST 1..{max_hops}]-"
            f"(b:Entity {{id: $to}})"
            f" RETURN nodes(p), rels(p)"
        )
        try:
            result = self._conn.execute(query, {"from": from_id, "to": to_id})
        except Exception:
            return None
        if not result.has_next():
            return None
        row = result.get_next()
        nodes_raw, rels_raw = row[0], row[1]

        path: list[dict[str, Any]] = []
        for i, n in enumerate(nodes_raw):
            path.append({"kind": "node", "id": n["id"], "node_type": n["node_type"], "label": n["label"]})
            if i < len(rels_raw):
                r = rels_raw[i]
                path.append({"kind": "edge", "edge_type": r["edge_type"]})
        return path

    # ── Raw Cypher ───────────────────────────────────────────────────────

    def query_cypher(self, cypher: str) -> list[list[Any]]:
        """Execute raw Cypher and return rows."""
        result = self._conn.execute(cypher)
        rows: list[list[Any]] = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    # ── NetworkX conversion (backward compat) ────────────────────────────

    def to_networkx(
        self,
        center_id: str | None = None,
        hops: int = 2,
    ) -> Any:
        """Convert (sub)graph to a NetworkX DiGraph.

        If ``center_id`` is given, extract the ``hops``-hop neighborhood.
        Otherwise convert the entire graph (use with care on large graphs).
        """
        import networkx as nx

        g = nx.DiGraph()

        if center_id:
            sg = self.subgraph(center_id, hops=hops)
            for n in sg["nodes"]:
                g.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
            for e in sg["edges"]:
                g.add_edge(e["src_id"], e["dst_id"], **{k: v for k, v in e.items() if k not in ("src_id", "dst_id")})
        else:
            # Full graph
            for node in self.search_nodes(limit=10_000):
                d = node.model_dump()
                nid = d.pop("id")
                g.add_node(nid, **d)
            for row in self.query_cypher(
                "MATCH (a:Entity)-[r:Edge]->(b:Entity) "
                "RETURN a.id, b.id, r.edge_type, r.confidence"
            ):
                g.add_edge(row[0], row[1], edge_type=row[2], confidence=row[3])

        return g

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        """Return node/edge counts by type."""
        counts: dict[str, int] = {}
        result = self._conn.execute(
            "MATCH (e:Entity) RETURN e.node_type, count(*) ORDER BY e.node_type"
        )
        while result.has_next():
            row = result.get_next()
            counts[f"node:{row[0]}"] = row[1]
        result = self._conn.execute(
            "MATCH ()-[r:Edge]->() RETURN r.edge_type, count(*) ORDER BY r.edge_type"
        )
        while result.has_next():
            row = result.get_next()
            counts[f"edge:{row[0]}"] = row[1]
        return counts

    # ── Internals ────────────────────────────────────────────────────────

    def _row_to_node(self, row: list[Any]) -> KGNode:
        """Convert a query row [id, node_type, label, props] to a typed node."""
        nid, ntype_str, label, props_str = row[0], row[1], row[2], row[3]
        props = json.loads(props_str) if props_str else {}
        node_type = NodeType(ntype_str)
        cls = NODE_CLASSES.get(node_type, KGNode)
        return cls(id=nid, node_type=node_type, label=label, **props)

    def _query_nodes(self, cypher: str, params: dict[str, Any] | None = None) -> list[KGNode]:
        """Execute Cypher returning [id, node_type, label, props] rows."""
        result = self._conn.execute(cypher, params or {})
        nodes: list[KGNode] = []
        while result.has_next():
            nodes.append(self._row_to_node(result.get_next()))
        return nodes

    def close(self) -> None:
        """Close the database connection."""
        # KuzuDB connections are closed on garbage collection,
        # but this makes intent explicit.
        self._conn = None  # type: ignore[assignment]
        self._db = None  # type: ignore[assignment]

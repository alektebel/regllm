"""Tests for the KuzuDB-backed knowledge graph store."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from src.knowledge.ontology import (
    Column,
    Concept,
    EdgeType,
    Experience,
    Insight,
    Interpretation,
    KGEdge,
    NodeType,
    Quirk,
    Regulation,
    Table,
    ValidationRule,
)
from src.knowledge.graph_store import GraphStore


@pytest.fixture()
def store(tmp_path: Path) -> GraphStore:
    db_path = tmp_path / "test.kuzu"
    return GraphStore(db_path=db_path)


@pytest.fixture()
def populated_store(store: GraphStore) -> GraphStore:
    """Store with a small regulation → concept → column chain."""
    store.add_node(Regulation(
        id="reg_art15",
        label="Article 15 – Minimum provisions",
        article_id="art15",
        source_document="Circular 6/2016",
        jurisdiction="ES",
    ))
    store.add_node(Concept(
        id="con_lgd_floor",
        label="LGD Floor",
        definition="Minimum LGD value per segment",
    ))
    store.add_node(Column(
        id="col_lgd",
        label="LGD_ESTIMADA",
        table_name="cycles",
        data_type="FLOAT",
        nullable=False,
    ))
    store.add_node(ValidationRule(
        id="vr_lgd_min",
        label="LGD >= floor",
        severity="ERROR",
        framework_version="4.0",
        rule_text="LGD_ESTIMADA must be >= segment floor",
    ))

    store.add_edge(KGEdge(
        src_id="reg_art15", dst_id="con_lgd_floor",
        edge_type=EdgeType.DEFINITION_IN,
    ))
    store.add_edge(KGEdge(
        src_id="con_lgd_floor", dst_id="col_lgd",
        edge_type=EdgeType.IMPLEMENTS,
        confidence=0.92,
    ))
    store.add_edge(KGEdge(
        src_id="reg_art15", dst_id="col_lgd",
        edge_type=EdgeType.GOVERNS,
        confidence=0.95,
    ))
    store.add_edge(KGEdge(
        src_id="vr_lgd_min", dst_id="col_lgd",
        edge_type=EdgeType.VALIDATES,
    ))
    return store


# ── Node CRUD ────────────────────────────────────────────────────────────────


class TestNodeCRUD:
    def test_add_and_get(self, store: GraphStore) -> None:
        reg = Regulation(id="r1", label="Art 8", article_id="art8")
        store.add_node(reg)
        got = store.get_node("r1")
        assert got is not None
        assert got.id == "r1"
        assert got.label == "Art 8"
        assert isinstance(got, Regulation)
        assert got.article_id == "art8"

    def test_get_missing_returns_none(self, store: GraphStore) -> None:
        assert store.get_node("nonexistent") is None

    def test_upsert_updates_existing(self, store: GraphStore) -> None:
        store.add_node(Regulation(id="r1", label="Old", article_id="old"))
        store.add_node(Regulation(id="r1", label="New", article_id="new"))
        got = store.get_node("r1")
        assert got is not None
        assert got.label == "New"
        assert isinstance(got, Regulation)
        assert got.article_id == "new"

    def test_delete_node(self, store: GraphStore) -> None:
        store.add_node(Concept(id="c1", label="Test"))
        store.delete_node("c1")
        assert store.get_node("c1") is None

    def test_all_node_types(self, store: GraphStore) -> None:
        """Every node type can be stored and retrieved with correct class."""
        nodes = [
            Regulation(id="n1", label="Reg"),
            Interpretation(id="n2", label="Interp", confidence=0.8, source="EBA_QA"),
            Concept(id="n3", label="Concept", definition="some def"),
            Table(id="n4", label="Table"),
            Column(id="n5", label="Column", table_name="t", data_type="INT"),
            ValidationRule(id="n6", label="VR", severity="WARNING"),
            Experience(id="n7", label="Exp", session_id="s1"),
            Quirk(id="n8", label="Quirk", severity="high"),
            Insight(id="n9", label="Insight", tags=["tag1"]),
        ]
        store.add_nodes(nodes)
        for n in nodes:
            got = store.get_node(n.id)
            assert got is not None
            assert type(got) is type(n), f"Expected {type(n)}, got {type(got)}"

    def test_label_with_quotes(self, store: GraphStore) -> None:
        store.add_node(Concept(id="c1", label="It's a test"))
        got = store.get_node("c1")
        assert got is not None
        assert got.label == "It's a test"


# ── Search ───────────────────────────────────────────────────────────────────


class TestSearch:
    def test_search_by_type(self, populated_store: GraphStore) -> None:
        regs = populated_store.search_nodes(node_types=[NodeType.REGULATION])
        assert len(regs) == 1
        assert regs[0].id == "reg_art15"

    def test_search_by_label(self, populated_store: GraphStore) -> None:
        results = populated_store.search_nodes(label_contains="LGD")
        labels = {n.label for n in results}
        assert "LGD Floor" in labels
        assert "LGD_ESTIMADA" in labels

    def test_search_combined(self, populated_store: GraphStore) -> None:
        results = populated_store.search_nodes(
            node_types=[NodeType.COLUMN],
            label_contains="LGD",
        )
        assert len(results) == 1
        assert results[0].id == "col_lgd"


# ── Edge CRUD ────────────────────────────────────────────────────────────────


class TestEdgeCRUD:
    def test_add_and_get_edges(self, populated_store: GraphStore) -> None:
        edges = populated_store.get_edges("reg_art15", direction="out")
        assert len(edges) >= 2
        edge_types = {e.edge_type for e in edges}
        assert EdgeType.DEFINITION_IN in edge_types
        assert EdgeType.GOVERNS in edge_types

    def test_get_edges_filtered(self, populated_store: GraphStore) -> None:
        edges = populated_store.get_edges(
            "reg_art15",
            edge_types=[EdgeType.GOVERNS],
            direction="out",
        )
        assert len(edges) == 1
        assert edges[0].dst_id == "col_lgd"
        assert edges[0].confidence == pytest.approx(0.95)

    def test_get_edges_inbound(self, populated_store: GraphStore) -> None:
        edges = populated_store.get_edges("col_lgd", direction="in")
        src_ids = {e.src_id for e in edges}
        assert "reg_art15" in src_ids
        assert "con_lgd_floor" in src_ids


# ── Graph traversal ──────────────────────────────────────────────────────────


class TestTraversal:
    def test_neighbors_1hop(self, populated_store: GraphStore) -> None:
        neighbors = populated_store.get_neighbors("reg_art15", hops=1)
        ids = {n.id for n in neighbors}
        assert "con_lgd_floor" in ids
        assert "col_lgd" in ids

    def test_neighbors_2hop(self, populated_store: GraphStore) -> None:
        # From validation rule, 2 hops should reach concept and regulation
        neighbors = populated_store.get_neighbors("vr_lgd_min", hops=2)
        ids = {n.id for n in neighbors}
        assert "col_lgd" in ids
        # Through col_lgd we should reach reg_art15 and con_lgd_floor
        assert "reg_art15" in ids or "con_lgd_floor" in ids

    def test_subgraph(self, populated_store: GraphStore) -> None:
        sg = populated_store.subgraph("col_lgd", hops=1)
        node_ids = {n["id"] for n in sg["nodes"]}
        assert "col_lgd" in node_ids
        assert "reg_art15" in node_ids
        assert len(sg["edges"]) >= 1

    def test_subgraph_missing_center(self, populated_store: GraphStore) -> None:
        sg = populated_store.subgraph("nonexistent")
        assert sg == {"nodes": [], "edges": []}


# ── Shortest path ────────────────────────────────────────────────────────────


class TestShortestPath:
    def test_direct_path(self, populated_store: GraphStore) -> None:
        path = populated_store.shortest_path("reg_art15", "col_lgd")
        assert path is not None
        # Should contain at least 2 nodes and 1 edge
        node_entries = [p for p in path if p["kind"] == "node"]
        assert len(node_entries) >= 2
        assert node_entries[0]["id"] == "reg_art15"
        assert node_entries[-1]["id"] == "col_lgd"

    def test_no_path(self, store: GraphStore) -> None:
        store.add_node(Concept(id="a", label="A"))
        store.add_node(Concept(id="b", label="B"))
        path = store.shortest_path("a", "b")
        assert path is None


# ── NetworkX conversion ──────────────────────────────────────────────────────


class TestNetworkX:
    def test_to_networkx_subgraph(self, populated_store: GraphStore) -> None:
        g = populated_store.to_networkx(center_id="col_lgd", hops=1)
        assert "col_lgd" in g.nodes
        assert "reg_art15" in g.nodes
        assert g.number_of_edges() >= 1

    def test_to_networkx_full(self, populated_store: GraphStore) -> None:
        g = populated_store.to_networkx()
        assert g.number_of_nodes() == 4
        assert g.number_of_edges() >= 4


# ── Stats ────────────────────────────────────────────────────────────────────


class TestStats:
    def test_stats(self, populated_store: GraphStore) -> None:
        s = populated_store.stats()
        assert s.get("node:Regulation") == 1
        assert s.get("node:Column") == 1
        assert s.get("edge:GOVERNS") == 1
        assert s.get("edge:VALIDATES") == 1


# ── Raw Cypher ───────────────────────────────────────────────────────────────


class TestCypher:
    def test_raw_query(self, populated_store: GraphStore) -> None:
        rows = populated_store.query_cypher(
            "MATCH (e:Entity) WHERE e.node_type = 'Regulation' RETURN e.id"
        )
        assert len(rows) == 1
        assert rows[0][0] == "reg_art15"

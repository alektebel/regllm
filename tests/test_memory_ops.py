"""Tests for memory graph operations (merge, concept creation)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.knowledge.graph_store import GraphStore
from src.knowledge.memory_ops import create_concept, merge_nodes
from src.knowledge.ontology import EdgeType, Insight, NodeType


@pytest.fixture()
def store(tmp_path: Path) -> GraphStore:
    return GraphStore(db_path=tmp_path / "mem_ops.kuzu")


def test_create_concept_links_insights(store: GraphStore) -> None:
    for i, seg in enumerate(("retail", "CORP")):
        store.add_node(Insight(
            id=f"ins_{i}",
            label=f"LGD {seg}",
            summary=f"LGD floor {seg} = {0.45 + i * 0.05}",
            tags=["LGD", seg],
        ))
    result = create_concept(
        store,
        label="LGD floor por segmento",
        definition="LGD floor varía por segmento de calibración",
        covers=["ins_0", "ins_1"],
        links_to=["LGD_ESTIMADA"],
    )
    assert result["created"] is True
    assert result["covers_linked"] == 2
    assert result["entropy_delta"] >= 0
    concept = store.get_node(result["id"])
    assert concept is not None
    assert concept.node_type == NodeType.CONCEPT


def test_create_concept_rejects_single_cover(store: GraphStore) -> None:
    store.add_node(Insight(id="ins_0", label="solo", summary="único insight"))
    result = create_concept(store, "Tema", "def", covers=["ins_0"])
    assert result["created"] is False


def test_merge_nodes(store: GraphStore) -> None:
    store.add_node(Insight(id="keep", label="canonical", summary="texto base", tags=["LGD"]))
    store.add_node(Insight(id="dup", label="dup", summary="texto dup", tags=["LGD"]))
    store.add_edge(__import__("src.knowledge.ontology", fromlist=["KGEdge"]).KGEdge(
        src_id="dup", dst_id="keep", edge_type=EdgeType.RELATES_TO,
    ))
    out = merge_nodes(store, "dup", "keep")
    assert out["merged"] is True
    assert store.get_node("dup") is None
    assert store.get_node("keep") is not None

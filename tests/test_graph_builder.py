"""Tests for the LLM-driven knowledge graph builder.

Uses a mock LLM client so tests run without Ollama.
Verifies the full pipeline: regulation ingestion, schema mapping,
session reflection, auto-linking, and the nuanced many-to-many
schema-regulation relationships.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.knowledge.graph_builder import GraphBuilder, _chunk_text, _make_id
from src.knowledge.graph_store import GraphStore
from src.knowledge.ontology import (
    Column,
    Concept,
    EdgeType,
    KGEdge,
    NodeType,
    Regulation,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_db(tmp_path):
    """Create a temporary KuzuDB database."""
    db_path = tmp_path / "test.kuzu"
    store = GraphStore(str(db_path))
    yield store


@pytest.fixture()
def mock_llm():
    """A mock LLM client that returns canned JSON responses."""
    client = MagicMock()
    client.detect_backend.return_value = "stub"
    return client


@pytest.fixture()
def mock_embed():
    """A mock embedding service."""
    svc = MagicMock()
    svc.available = False
    svc.embed_one.return_value = [0.0] * 768
    return svc


@pytest.fixture()
def builder(tmp_db, mock_llm, mock_embed):
    return GraphBuilder(tmp_db, vector_store=None, embed_service=mock_embed, llm=mock_llm)


# ── Helper ───────────────────────────────────────────────────────────────────


def _set_llm_response(mock_llm, response: dict[str, Any]):
    """Configure the mock LLM to return a specific JSON dict."""
    mock_llm.chat_json.return_value = response


# ── Unit tests: helpers ──────────────────────────────────────────────────────


class TestChunkText:
    def test_short_text_single_chunk(self):
        assert _chunk_text("hello", 100) == ["hello"]

    def test_splits_on_paragraph_boundaries(self):
        text = "Para one.\n\nPara two.\n\nPara three.\n\nPara four."
        chunks = _chunk_text(text, 25)
        assert len(chunks) >= 2
        # All content preserved
        rejoined = "\n\n".join(chunks)
        assert "Para one." in rejoined
        assert "Para four." in rejoined

    def test_no_paragraphs_stays_single(self):
        text = "x" * 5000
        chunks = _chunk_text(text, 3000)
        # No paragraph breaks → can't split, returns as-is
        assert len(chunks) == 1

    def test_empty_text(self):
        assert _chunk_text("", 100) == [""]


class TestMakeId:
    def test_deterministic(self):
        a = _make_id("Regulation", "Article 15")
        b = _make_id("Regulation", "Article 15")
        assert a == b

    def test_different_types_different_prefix(self):
        reg = _make_id("Regulation", "foo")
        con = _make_id("Concept", "foo")
        assert reg.startswith("reg:")
        assert con.startswith("concept:")

    def test_truncates_long_names(self):
        long_name = "A" * 200
        result = _make_id("Insight", long_name)
        # prefix:slug_hash — slug capped at 60 chars
        slug_part = result.split(":")[1].rsplit("_", 1)[0]
        assert len(slug_part) <= 60


# ── Integration: regulation ingestion ────────────────────────────────────────


class TestRegulationIngestion:
    def test_creates_nodes_and_edges(self, builder, tmp_db, mock_llm):
        # Use full _make_id-style IDs so _resolve_node_id can find them
        art15_id = _make_id("Regulation", "art15")
        lgd_id = _make_id("Concept", "lgd_floor")
        rule_id = _make_id("ValidationRule", "lgd_rule_045")

        mock_llm.chat_json.side_effect = [
            # Extraction
            {
                "entities": [
                    {"id": "art15", "type": "Regulation", "label": "Article 15 — Minimum provisions",
                     "props": {"article_id": "art15", "jurisdiction": "ES"}},
                    {"id": "lgd_floor", "type": "Concept", "label": "LGD floor",
                     "props": {"definition": "Minimum LGD value for downturn estimation"}},
                    {"id": "lgd_rule_045", "type": "ValidationRule", "label": "LGD >= 0.45 for corporate",
                     "props": {"rule_text": "LGD >= 0.45", "severity": "ERROR"}},
                ],
                "relationships": [
                    # Use the generated IDs so resolution works
                    {"source": art15_id, "target": lgd_id, "type": "GOVERNS", "confidence": 0.95,
                     "reason": "Article 15 sets the LGD floor"},
                    {"source": lgd_id, "target": rule_id, "type": "RELATES_TO", "confidence": 0.8,
                     "reason": "Floor concept materialised as validation rule"},
                ],
            },
            # Auto-link (no extra links)
            {"links": []},
        ]

        result = builder.ingest_regulation(
            "Article 15. Minimum provisions for LGD...",
            source="circular_4_2022",
        )

        assert result["nodes"] >= 3
        assert result["edges"] >= 2

        # Verify nodes exist in the graph
        nodes = tmp_db.search_nodes(limit=100)
        labels = {n.label for n in nodes}
        assert "Article 15 — Minimum provisions" in labels
        assert "LGD floor" in labels

    def test_handles_llm_error_gracefully(self, builder, mock_llm):
        _set_llm_response(mock_llm, {"error": "invalid_json", "raw": "garbage"})

        result = builder.ingest_regulation("Some text", source="test")
        # Should not crash, just return zero counts
        assert result["nodes"] == 0
        assert result["edges"] == 0

    def test_chunks_long_text(self, builder, mock_llm):
        _set_llm_response(mock_llm, {"entities": [], "relationships": []})

        long_text = "\n\n".join([f"Paragraph {i}. " * 100 for i in range(20)])
        builder.ingest_regulation(long_text, source="long_doc")

        # Should have been called multiple times (once per chunk + auto-link)
        assert mock_llm.chat_json.call_count >= 2


# ── Integration: schema mapping (the nuanced part) ──────────────────────────


class TestSchemaMapping:
    """Tests for the many-to-many, context-dependent schema mapping."""

    def test_many_to_many_mapping(self, builder, tmp_db, mock_llm):
        """A single column can be governed by multiple regulations,
        and a single regulation can govern multiple columns."""
        # Pre-populate some regulation nodes
        tmp_db.add_node(Regulation(
            id="reg:art15", label="Article 15 — LGD floors",
            article_id="art15", jurisdiction="ES",
        ))
        tmp_db.add_node(Regulation(
            id="reg:art22", label="Article 22 — Downturn calibration",
            article_id="art22", jurisdiction="ES",
        ))
        tmp_db.add_node(Column(
            id="col:LGD_ESTIMADA", label="LGD_ESTIMADA",
            table_name="cycles", data_type="DECIMAL",
        ))

        _set_llm_response(mock_llm, {
            "mappings": [
                {
                    "column": "LGD_ESTIMADA",
                    "regulations": [
                        {"regulation_id": "reg:art15", "edge_type": "GOVERNS",
                         "confidence": 0.95,
                         "reasoning": "Art 15 sets minimum LGD floor of 0.45 for corporate exposures"},
                        {"regulation_id": "reg:art22", "edge_type": "GOVERNS",
                         "confidence": 0.80,
                         "reasoning": "Art 22 requires downturn calibration affecting LGD estimates"},
                    ],
                    "concepts": ["LGD floor", "downturn adjustment"],
                    "validation_rules": [
                        {"rule": "LGD_ESTIMADA >= 0.45 WHERE ASSET_CLASS = 'CORPORATE'",
                         "severity": "ERROR", "source_article": "art15"},
                        {"rule": "LGD_ESTIMADA >= LGD_FLOOR_APLICADO",
                         "severity": "ERROR", "source_article": "art15"},
                    ],
                },
            ],
        })

        ddl = "CREATE TABLE cycles (LGD_ESTIMADA DECIMAL(10,6) NOT NULL);"
        result = builder.map_schema_to_regulations(ddl)

        # Should create edges from both regulations to the column
        assert result["edges"] >= 2

        # Verify the edges exist
        edges = tmp_db.get_edges("col:LGD_ESTIMADA", direction="in")
        governing_regs = {e.src_id for e in edges if e.edge_type == EdgeType.GOVERNS}
        assert "reg:art15" in governing_regs
        assert "reg:art22" in governing_regs

        # Should create validation rules
        assert result["rules"] >= 2

        # Should create concept nodes
        assert result["concepts"] >= 2

    def test_conditional_mapping_with_confidence(self, builder, tmp_db, mock_llm):
        """Mapping confidence varies by context — asset class, vintage, etc."""
        tmp_db.add_node(Regulation(
            id="reg:art15", label="Article 15", article_id="art15",
        ))
        tmp_db.add_node(Column(
            id="col:PD_ESTIMADA", label="PD_ESTIMADA",
            table_name="cycles", data_type="DECIMAL",
        ))

        _set_llm_response(mock_llm, {
            "mappings": [{
                "column": "PD_ESTIMADA",
                "regulations": [
                    {"regulation_id": "reg:art15", "edge_type": "GOVERNS",
                     "confidence": 0.6,
                     "reasoning": "Art 15 governs PD only indirectly through the EL = PD * LGD formula; "
                                  "the direct PD constraint is in Art 10"},
                ],
                "concepts": ["probability of default"],
                "validation_rules": [],
            }],
        })

        result = builder.map_schema_to_regulations("CREATE TABLE cycles (PD_ESTIMADA DECIMAL);")
        edges = tmp_db.get_edges("col:PD_ESTIMADA", direction="in")
        assert len(edges) >= 1
        # Confidence should be preserved (0.6, not 1.0)
        governs_edges = [e for e in edges if e.edge_type == EdgeType.GOVERNS]
        assert governs_edges[0].confidence == pytest.approx(0.6)

    def test_no_mappings_returned(self, builder, mock_llm):
        """LLM finds no relevant mappings — should not crash."""
        _set_llm_response(mock_llm, {"mappings": []})
        result = builder.map_schema_to_regulations("CREATE TABLE foo (bar INT);")
        assert result["edges"] == 0

    def test_uses_existing_graph_as_context(self, builder, tmp_db, mock_llm):
        """The mapping call should pass existing Regulation nodes as context."""
        tmp_db.add_node(Regulation(
            id="reg:art15", label="Article 15", article_id="art15",
        ))

        _set_llm_response(mock_llm, {"mappings": []})
        builder.map_schema_to_regulations("CREATE TABLE t (c INT);")

        # Verify the LLM was called with regulation context
        call_args = mock_llm.chat_json.call_args
        user_msg = call_args[0][1]  # second positional arg
        assert "reg:art15" in user_msg or "Article 15" in user_msg


# ── Integration: session reflection ──────────────────────────────────────────


class TestSessionReflection:
    def test_creates_insights_and_quirks(self, builder, tmp_db, mock_llm):
        _set_llm_response(mock_llm, {
            "insights": [
                {
                    "summary": "LGD floor changed from 0.45 to 0.50 for retail exposures in 2024",
                    "related_fields": ["LGD_ESTIMADA", "LGD_FLOOR_APLICADO"],
                    "related_articles": ["art15"],
                    "tags": ["lgd", "retail", "threshold_change"],
                    "confidence": 0.85,
                },
            ],
            "quirks": [
                {
                    "description": "CURE_FLAG=1 rows have LGD=0 which bypasses the floor check",
                    "severity": "high",
                    "related_fields": ["CURE_FLAG", "LGD_ESTIMADA"],
                    "suggested_validation": "WHERE CURE_FLAG=1 AND LGD_ESTIMADA > 0",
                },
            ],
            "new_connections": [],
        })

        result = builder.reflect_on_session(
            "User asked about LGD differences. Found that...",
            session_id="test_session_001",
        )

        assert result["insights"] == 1
        assert result["quirks"] == 1

        # Verify insight node
        insights = tmp_db.search_nodes(node_types=[NodeType.INSIGHT], limit=10)
        assert len(insights) >= 1
        assert "LGD floor" in insights[0].label

        # Verify quirk node
        quirks = tmp_db.search_nodes(node_types=[NodeType.QUIRK], limit=10)
        assert len(quirks) >= 1
        assert "CURE_FLAG" in quirks[0].label

        # Verify experience node
        exp = tmp_db.get_node("exp:test_session_001")
        assert exp is not None

    def test_creates_cross_links_between_existing_nodes(self, builder, tmp_db, mock_llm):
        """Reflection can discover new connections between existing nodes."""
        # Pre-populate
        tmp_db.add_node(Regulation(id="reg:art15", label="Article 15", article_id="art15"))
        tmp_db.add_node(Column(id="col:LGD_ESTIMADA", label="LGD_ESTIMADA", table_name="cycles"))

        _set_llm_response(mock_llm, {
            "insights": [],
            "quirks": [],
            "new_connections": [
                {
                    "source_type": "Regulation",
                    "source_label": "reg:art15",
                    "target_type": "Column",
                    "target_label": "col:LGD_ESTIMADA",
                    "edge_type": "GOVERNS",
                    "confidence": 0.9,
                    "reason": "Session revealed Art 15 directly constrains LGD_ESTIMADA",
                },
            ],
        })

        result = builder.reflect_on_session("Session log...", session_id="sess_002")
        assert result["connections"] == 1

        # Verify the edge was created
        edges = tmp_db.get_edges("col:LGD_ESTIMADA", direction="in")
        assert any(e.src_id == "reg:art15" and e.edge_type == EdgeType.GOVERNS for e in edges)

    def test_empty_reflection(self, builder, mock_llm):
        """Session with nothing interesting — should not crash."""
        _set_llm_response(mock_llm, {"insights": [], "quirks": [], "new_connections": []})
        result = builder.reflect_on_session("Trivial question answered.", session_id="sess_003")
        assert result["insights"] == 0
        assert result["quirks"] == 0
        assert result["connections"] == 0


# ── Integration: auto-linking ────────────────────────────────────────────────


class TestAutoLinking:
    def test_auto_link_connects_new_to_existing(self, builder, tmp_db, mock_llm):
        """After ingesting new entities, auto-link finds connections to existing graph."""
        # Existing node
        tmp_db.add_node(Column(id="col:LGD_ESTIMADA", label="LGD_ESTIMADA", table_name="cycles"))

        lgd_concept_id = _make_id("Concept", "lgd_concept")

        # First call: regulation extraction; second: auto-link
        mock_llm.chat_json.side_effect = [
            {
                "entities": [
                    {"id": "lgd_concept", "type": "Concept", "label": "LGD floor"},
                ],
                "relationships": [],
            },
            {
                "links": [
                    {
                        "source_id": "col:LGD_ESTIMADA",
                        "target_id": lgd_concept_id,
                        "edge_type": "IMPLEMENTS",
                        "confidence": 0.85,
                        "reason": "Column implements the LGD floor concept",
                    },
                ],
            },
        ]

        result = builder.ingest_regulation("LGD floor provisions...", source="test")

        # The auto-link should have created an edge
        edges = tmp_db.get_edges("col:LGD_ESTIMADA", direction="out")
        impl_edges = [e for e in edges if e.edge_type == EdgeType.IMPLEMENTS]
        assert len(impl_edges) >= 1


# ── Integration: node resolution ─────────────────────────────────────────────


class TestNodeResolution:
    def test_resolves_by_prefix(self, builder, tmp_db):
        tmp_db.add_node(Regulation(id="reg:art15", label="Article 15"))
        assert builder._resolve_node_id("art15") == "reg:art15"

    def test_resolves_by_direct_id(self, builder, tmp_db):
        tmp_db.add_node(Concept(id="concept:LGD_FLOOR", label="LGD floor"))
        assert builder._resolve_node_id("concept:LGD_FLOOR") == "concept:LGD_FLOOR"

    def test_resolves_by_label_search(self, builder, tmp_db):
        tmp_db.add_node(Regulation(id="reg:art15_provisions", label="Article 15 — Provisions"))
        result = builder._resolve_node_id("Article 15")
        assert result == "reg:art15_provisions"

    def test_returns_none_for_unknown(self, builder, tmp_db):
        assert builder._resolve_node_id("nonexistent_node") is None

    def test_returns_none_for_empty(self, builder):
        assert builder._resolve_node_id("") is None


# ── Integration: full build ──────────────────────────────────────────────────


class TestFullBuild:
    def test_build_from_directory(self, builder, tmp_db, mock_llm, tmp_path):
        """Full pipeline: regulation files → graph."""
        # Create temp regulation files
        reg_dir = tmp_path / "regulation"
        reg_dir.mkdir()
        (reg_dir / "art_15_lgd.md").write_text(
            "# Article 15 — LGD Floor\n\nMinimum LGD of 0.45 for corporate.\n"
        )
        (reg_dir / "art_22_downturn.md").write_text(
            "# Article 22 — Downturn\n\nDownturn adjustment required.\n"
        )

        # Schema file
        schema_file = tmp_path / "schema.sql"
        schema_file.write_text(
            "CREATE TABLE cycles (LGD_ESTIMADA DECIMAL(10,6) NOT NULL);\n"
        )

        art15_id = _make_id("Regulation", "art15")

        # Default response for any LLM call — keeps things simple
        _empty_extraction = {"entities": [], "relationships": []}
        _empty_links = {"links": []}

        def _chat_json_side_effect(*args, **kwargs):
            """Return canned responses based on prompt content."""
            user_msg = args[1] if len(args) > 1 else ""
            if "art_15_lgd" in user_msg or "Article 15" in user_msg:
                return {"entities": [{"id": "art15", "type": "Regulation", "label": "Art 15"}],
                        "relationships": []}
            if "art_22_downturn" in user_msg or "Article 22" in user_msg:
                return {"entities": [{"id": "art22", "type": "Regulation", "label": "Art 22"}],
                        "relationships": []}
            if "DDL" in user_msg or "DATABASE SCHEMA" in user_msg:
                return {"mappings": [
                    {"column": "LGD_ESTIMADA",
                     "regulations": [{"regulation_id": art15_id, "edge_type": "GOVERNS",
                                      "confidence": 0.9, "reasoning": "LGD floor"}],
                     "concepts": [], "validation_rules": []},
                ]}
            if "EXISTING NODES" in user_msg and "NEW NODES" in user_msg:
                return {"links": []}
            return _empty_extraction

        mock_llm.chat_json.side_effect = _chat_json_side_effect

        # Need to pre-create column for mapping edge to work
        tmp_db.add_node(Column(id="col:LGD_ESTIMADA", label="LGD_ESTIMADA", table_name="cycles"))

        result = builder.build_from_directory(
            regulation_dir=str(reg_dir),
            schema_file=str(schema_file),
        )

        assert result["nodes"] >= 2  # At least the 2 regulation nodes
        stats = tmp_db.stats()
        assert sum(v for k, v in stats.items() if k.startswith("node:")) >= 3


# ── Edge case: LLM returns unexpected shapes ────────────────────────────────


class TestLLMEdgeCases:
    def test_missing_entities_key(self, builder, mock_llm):
        _set_llm_response(mock_llm, {"something_else": []})
        result = builder.ingest_regulation("text", source="test")
        assert result["nodes"] == 0

    def test_entity_without_label(self, builder, tmp_db, mock_llm):
        _set_llm_response(mock_llm, {
            "entities": [{"id": "mystery", "type": "Concept"}],
            "relationships": [],
        })
        result = builder.ingest_regulation("text", source="test")
        # Should still create the node using id as label
        assert result["nodes"] == 1

    def test_relationship_with_invalid_edge_type(self, builder, tmp_db, mock_llm):
        """Invalid edge type should fall back to RELATES_TO."""
        tmp_db.add_node(Concept(id="concept:a", label="A"))
        tmp_db.add_node(Concept(id="concept:b", label="B"))

        _set_llm_response(mock_llm, {
            "entities": [],
            "relationships": [
                {"source": "concept:a", "target": "concept:b",
                 "type": "TOTALLY_MADE_UP", "confidence": 0.5},
            ],
        })
        result = builder.ingest_regulation("text", source="test")
        # Should create edge with RELATES_TO fallback
        edges = tmp_db.get_edges("concept:a", direction="out")
        assert any(e.edge_type == EdgeType.RELATES_TO for e in edges)

    def test_reflection_with_malformed_response(self, builder, mock_llm):
        _set_llm_response(mock_llm, {"error": "invalid_json", "raw": "{bad"})
        result = builder.reflect_on_session("session log", session_id="bad_sess")
        assert result["insights"] == 0

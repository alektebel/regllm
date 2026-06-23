"""Tests for the knowledge graph ingestors."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.knowledge.graph_store import GraphStore
from src.knowledge.ontology import EdgeType, NodeType
from src.knowledge.ingestors.regulation_ingestor import ingest_regulations
from src.knowledge.ingestors.schema_ingestor import ingest_schema
from src.knowledge.ingestors.changelog_ingestor import ingest_changelog


@pytest.fixture()
def store(tmp_path: Path) -> GraphStore:
    return GraphStore(db_path=tmp_path / "test.kuzu")


@pytest.fixture()
def reg_dir(tmp_path: Path) -> Path:
    """Create a minimal regulation directory."""
    d = tmp_path / "regulation"
    d.mkdir()
    (d / "art_15_dotaciones_minimas.md").write_text(
        "# Artículo 15 — Dotaciones Mínimas\n\n"
        "## Suelos mínimos de LGD\n\n"
        "El campo `LGD_ESTIMADA` no puede ser inferior al suelo.\n"
        "Ver también artículo 12 para períodos.\n\n"
        "## Suelo mínimo de PD\n\n"
        "`PD_ESTIMADA` no puede ser inferior al 0.05%.\n",
        encoding="utf-8",
    )
    (d / "art_12_periodos_dotacion.md").write_text(
        "# Artículo 12 — Períodos de Dotación\n\n"
        "El campo `PROVISION_PERIOD_MONTHS` define la duración.\n"
        "Según artículo 15, se aplican suelos.\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture()
def schema_file(tmp_path: Path) -> Path:
    """Create a minimal SQL DDL file."""
    f = tmp_path / "schema.sql"
    f.write_text(
        "CREATE TABLE ciclos_recuperacion (\n"
        "  CICLO_ID VARCHAR(20),\n"
        "  LGD_ESTIMADA FLOAT,\n"
        "  PD_ESTIMADA FLOAT,\n"
        "  EAD FLOAT NOT NULL,\n"
        "  PROVISION_PERIOD_MONTHS INTEGER\n"
        ");\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture()
def mapping_file(tmp_path: Path) -> Path:
    """Create a minimal mapping file."""
    import json
    f = tmp_path / "mapping.json"
    f.write_text(json.dumps({
        "LGD_ESTIMADA": {
            "regulation_concept": "LGD floor",
            "articles": ["art_15_dotaciones_minimas"],
            "confidence": 0.95,
            "needs_review": False,
            "regulation_description": "LGD must be >= floor",
        },
        "PROVISION_PERIOD_MONTHS": {
            "regulation_concept": "Provision period",
            "articles": ["art_12_periodos_dotacion"],
            "confidence": 0.55,
            "needs_review": True,
            "regulation_description": "Duration of provision",
        },
    }), encoding="utf-8")
    return f


@pytest.fixture()
def changelog_dir(tmp_path: Path) -> Path:
    """Create a minimal changelog directory."""
    d = tmp_path / "changelog"
    d.mkdir()
    (d / "2025-q1-pd-recalibration.md").write_text(
        "# PD Recalibration Q1 2025\n\n"
        "## Changes\n\n"
        "Applied 1.15x multiplier to PD_ESTIMADA for RATING_GRADO 1-2.\n",
        encoding="utf-8",
    )
    return d


# ── Regulation ingestor ──────────────────────────────────────────────────────


class TestRegulationIngestor:
    def test_creates_regulation_nodes(self, store: GraphStore, reg_dir: Path) -> None:
        counts = ingest_regulations(store, regulation_dirs=[reg_dir])
        assert counts["nodes_created"] > 0

        regs = store.search_nodes(node_types=[NodeType.REGULATION])
        assert len(regs) == 2
        labels = {r.label for r in regs}
        assert any("15" in l for l in labels)
        assert any("12" in l for l in labels)

    def test_creates_concept_nodes(self, store: GraphStore, reg_dir: Path) -> None:
        ingest_regulations(store, regulation_dirs=[reg_dir])

        concepts = store.search_nodes(node_types=[NodeType.CONCEPT])
        concept_labels = {c.label for c in concepts}
        assert "LGD_ESTIMADA" in concept_labels
        assert "PD_ESTIMADA" in concept_labels

    def test_creates_definition_edges(self, store: GraphStore, reg_dir: Path) -> None:
        ingest_regulations(store, regulation_dirs=[reg_dir])

        edges = store.get_edges("concept:LGD_ESTIMADA", direction="out")
        edge_types = {e.edge_type for e in edges}
        assert EdgeType.DEFINITION_IN in edge_types

    def test_cross_references(self, store: GraphStore, reg_dir: Path) -> None:
        counts = ingest_regulations(store, regulation_dirs=[reg_dir])

        # Art 15 references Art 12 and vice versa
        edges_15 = store.get_edges(
            "reg:art_15_dotaciones_minimas",
            edge_types=[EdgeType.SUBJECT_TO],
            direction="out",
        )
        edges_12 = store.get_edges(
            "reg:art_12_periodos_dotacion",
            edge_types=[EdgeType.SUBJECT_TO],
            direction="out",
        )
        # At least one cross-reference should exist
        assert len(edges_15) > 0 or len(edges_12) > 0


# ── Schema ingestor ──────────────────────────────────────────────────────────


class TestSchemaIngestor:
    def test_creates_table_and_columns(
        self, store: GraphStore, schema_file: Path, mapping_file: Path,
    ) -> None:
        counts = ingest_schema(store, schema_file=schema_file, mapping_file=mapping_file)
        assert counts["nodes_created"] > 0

        tables = store.search_nodes(node_types=[NodeType.TABLE])
        assert len(tables) == 1
        assert tables[0].label == "ciclos_recuperacion"

        columns = store.search_nodes(node_types=[NodeType.COLUMN])
        col_labels = {c.label for c in columns}
        assert "LGD_ESTIMADA" in col_labels
        assert "EAD" in col_labels

    def test_table_contains_column_edges(
        self, store: GraphStore, schema_file: Path, mapping_file: Path,
    ) -> None:
        ingest_schema(store, schema_file=schema_file, mapping_file=mapping_file)

        edges = store.get_edges("table:ciclos_recuperacion", direction="out")
        dst_ids = {e.dst_id for e in edges}
        assert "col:LGD_ESTIMADA" in dst_ids

    def test_mapping_creates_governs_edges(
        self, store: GraphStore, reg_dir: Path, schema_file: Path, mapping_file: Path,
    ) -> None:
        # Ingest regulations first so regulation nodes exist
        ingest_regulations(store, regulation_dirs=[reg_dir])
        ingest_schema(store, schema_file=schema_file, mapping_file=mapping_file)

        # Check GOVERNS edge from regulation to column
        edges = store.get_edges("col:LGD_ESTIMADA", direction="in")
        governs = [e for e in edges if e.edge_type == EdgeType.GOVERNS]
        assert len(governs) >= 1
        assert governs[0].confidence == pytest.approx(0.95)

    def test_low_confidence_flagged(
        self, store: GraphStore, reg_dir: Path, schema_file: Path, mapping_file: Path,
    ) -> None:
        ingest_regulations(store, regulation_dirs=[reg_dir])
        ingest_schema(store, schema_file=schema_file, mapping_file=mapping_file)

        # PROVISION_PERIOD_MONTHS has confidence 0.55 → NEEDS_HUMAN_REVIEW
        edges = store.get_edges("col:PROVISION_PERIOD_MONTHS", direction="out")
        review_edges = [e for e in edges if e.edge_type == EdgeType.NEEDS_HUMAN_REVIEW]
        assert len(review_edges) >= 1


# ── Changelog ingestor ───────────────────────────────────────────────────────


class TestChangelogIngestor:
    def test_creates_nodes(self, store: GraphStore, changelog_dir: Path) -> None:
        counts = ingest_changelog(store, changelog_dir=changelog_dir)
        assert counts["nodes_created"] > 0

    def test_mentions_fields(self, store: GraphStore, changelog_dir: Path) -> None:
        ingest_changelog(store, changelog_dir=changelog_dir)

        concepts = store.search_nodes(node_types=[NodeType.CONCEPT])
        labels = {c.label for c in concepts}
        assert "PD_ESTIMADA" in labels or "RATING_GRADO" in labels


# ── Integration: full pipeline ───────────────────────────────────────────────


class TestFullIngestion:
    def test_regulation_to_column_path(
        self, store: GraphStore, reg_dir: Path, schema_file: Path,
        mapping_file: Path, changelog_dir: Path,
    ) -> None:
        """After full ingestion, we can trace from regulation to column."""
        ingest_regulations(store, regulation_dirs=[reg_dir])
        ingest_schema(store, schema_file=schema_file, mapping_file=mapping_file)
        ingest_changelog(store, changelog_dir=changelog_dir)

        stats = store.stats()
        assert sum(v for k, v in stats.items() if k.startswith("node:")) >= 5
        assert sum(v for k, v in stats.items() if k.startswith("edge:")) >= 3

        # Multi-hop: regulation → column
        neighbors = store.get_neighbors("reg:art_15_dotaciones_minimas", hops=2)
        neighbor_ids = {n.id for n in neighbors}
        # Should reach column nodes through concepts or direct GOVERNS edges
        assert any(nid.startswith("col:") or nid.startswith("concept:") for nid in neighbor_ids)

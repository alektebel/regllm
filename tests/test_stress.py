"""Stress tests — large codebases, .egp parsing, graph scale, LLM thinking-mode.

These tests verify the app handles realistic production loads:
- Large SAS programs (10k+ lines)
- .egp files with many code blocks
- Graph ingestion with hundreds of nodes
- LLM client thinking-mode tag stripping
- KuzuDB lock retry
"""

from __future__ import annotations

import io
import shutil
import tempfile
import textwrap
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _big_sas_code(n_steps: int = 50, vars_per_step: int = 20) -> str:
    """Generate a large but syntactically valid SAS DATA step program."""
    blocks = []
    for step in range(n_steps):
        assigns = "\n".join(
            f"  VAR_{step}_{v} = VAR_{step}_{v - 1} + {v * 0.01};"
            if v > 0
            else f"  VAR_{step}_0 = INPUT_FIELD_{step};"
            for v in range(vars_per_step)
        )
        blocks.append(
            f"DATA work.step_{step};\n"
            f"  SET work.{'raw' if step == 0 else f'step_{step - 1}'};\n"
            f"{assigns}\n"
            f"RUN;\n"
        )
    return "\n".join(blocks)


def _make_egp(n_tasks: int = 20, lines_per_task: int = 50) -> bytes:
    """Build a synthetic .egp ZIP with project.xml + embedded SAS blocks."""
    root = ET.Element("SASProject")
    for i in range(n_tasks):
        task = ET.SubElement(root, "CodeTask")
        name = ET.SubElement(task, "Name")
        name.text = f"Task_{i:03d}_Processing"
        code_el = ET.SubElement(task, "Text")
        lines = [f"/* Task {i} */"]
        lines.append(f"DATA work.out_{i};")
        lines.append(f"  SET work.in_{i};")
        for j in range(lines_per_task):
            lines.append(f"  FIELD_{j} = FIELD_{j} * {1 + j * 0.001:.4f};")
        lines.append("RUN;")
        code_el.text = "\n".join(lines)

    xml_bytes = ET.tostring(root, encoding="unicode").encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.xml", xml_bytes)
        # Also add a few standalone .sas files
        for i in range(3):
            zf.writestr(
                f"programs/extra_{i}.sas",
                f"PROC PRINT DATA=work.out_{i}; RUN;\n",
            )
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
# 1. Large SAS parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestLargeSASParsing:
    """SASLogicTree must handle programs with thousands of assignments."""

    def test_parse_large_program(self):
        from src.sas_logic_tree import SASLogicTree

        code = _big_sas_code(n_steps=40, vars_per_step=30)
        assert len(code) > 30_000, f"Generated code should be >30KB, got {len(code)}"

        tree = SASLogicTree()
        nodes = tree.parse(code)
        assert len(nodes) > 0, "Parser should produce AST nodes"

        # Lineage should not crash
        lg = tree.lineage(nodes)
        assert len(lg.nodes) > 0

    def test_parse_deeply_nested_ifs(self):
        from src.sas_logic_tree import SASLogicTree

        depth = 30
        code = "DATA work.deep;\n  SET work.input;\n"
        for i in range(depth):
            code += "  " * (i + 1) + f"IF COND_{i} > 0 THEN DO;\n"
            code += "  " * (i + 2) + f"RESULT_{i} = {i};\n"
        for i in range(depth - 1, -1, -1):
            code += "  " * (i + 1) + "END;\n"
        code += "RUN;\n"

        tree = SASLogicTree()
        nodes = tree.parse(code)
        assert len(nodes) > 0

    def test_evaluate_large_row(self):
        from src.sas_logic_tree import SASLogicTree

        code = "DATA work.out;\n  SET work.inp;\n"
        for i in range(200):
            code += f"  F_{i} = F_{i} + 1;\n"
        code += "RUN;\n"

        tree = SASLogicTree()
        nodes = tree.parse(code)
        row = {f"F_{i}": float(i) for i in range(200)}
        trace = tree.evaluate(nodes, row)
        assert trace is not None
        assert trace.final.get("F_199") == 200.0


# ═══════════════════════════════════════════════════════════════════════════
# 2. .egp parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestEGPParsing:
    """SASParser.parse() on .egp files must handle many embedded blocks."""

    def test_parse_large_egp(self, tmp_path):
        from src.sas_parser import SASParser

        egp_bytes = _make_egp(n_tasks=20, lines_per_task=50)
        egp_path = tmp_path / "big_project.egp"
        egp_path.write_bytes(egp_bytes)

        parser = SASParser()
        blocks = parser.parse(egp_path)

        # Should extract tasks from XML + standalone .sas files
        assert len(blocks) >= 20, f"Expected >=20 blocks, got {len(blocks)}"

        # Most blocks should have non-trivial code (external .sas may be short)
        long_blocks = [b for b in blocks if len(b.code) > 50]
        assert len(long_blocks) >= 15, f"Expected >=15 long blocks, got {len(long_blocks)}"

    def test_egp_deduplication(self, tmp_path):
        """Duplicate code in XML and .sas should be deduplicated."""
        from src.sas_parser import SASParser

        code = "DATA work.test;\n  SET work.input;\n  X = 1;\nRUN;\n"
        root = ET.Element("SASProject")
        task = ET.SubElement(root, "CodeTask")
        ET.SubElement(task, "Name").text = "Test Task"
        ET.SubElement(task, "Text").text = code

        xml_bytes = ET.tostring(root, encoding="unicode").encode("utf-8")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("project.xml", xml_bytes)
            zf.writestr("programs/test.sas", code)  # Same code

        egp_path = tmp_path / "dupe.egp"
        egp_path.write_bytes(buf.getvalue())

        blocks = SASParser().parse(egp_path)
        # Should only appear once due to fingerprint dedup
        assert len(blocks) == 1

    def test_empty_egp(self, tmp_path):
        """An .egp with no code blocks should return empty list."""
        from src.sas_parser import SASParser

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("project.xml", "<SASProject></SASProject>")
        egp_path = tmp_path / "empty.egp"
        egp_path.write_bytes(buf.getvalue())

        blocks = SASParser().parse(egp_path)
        assert blocks == []

    def test_corrupt_xml_in_egp(self, tmp_path):
        """Corrupt project.xml should not crash — just skip XML blocks."""
        from src.sas_parser import SASParser

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("project.xml", "<broken><<<not xml")
            zf.writestr("programs/valid.sas", "PROC PRINT DATA=work.x; RUN;\n")

        egp_path = tmp_path / "corrupt.egp"
        egp_path.write_bytes(buf.getvalue())

        blocks = SASParser().parse(egp_path)
        assert len(blocks) == 1  # Only the standalone .sas


# ═══════════════════════════════════════════════════════════════════════════
# 3. Graph store at scale
# ═══════════════════════════════════════════════════════════════════════════


class TestGraphStoreScale:
    """GraphStore must handle hundreds of nodes and edges efficiently."""

    @pytest.fixture()
    def store(self, tmp_path):
        from src.knowledge.graph_store import GraphStore
        return GraphStore(db_path=tmp_path / "scale.kuzu")

    def test_bulk_insert_500_nodes(self, store):
        from src.knowledge.ontology import Concept, EdgeType, KGEdge

        # Insert 500 concepts
        for i in range(500):
            store.add_node(Concept(
                id=f"concept:{i}", label=f"Test Concept {i}",
                definition=f"Definition for concept {i}" * 5,
            ))

        stats = store.stats()
        assert stats.get("node:Concept", 0) == 500

        # Add 1000 edges
        for i in range(500):
            target = (i + 1) % 500
            store.add_edge(KGEdge(
                src_id=f"concept:{i}", dst_id=f"concept:{target}",
                edge_type=EdgeType.RELATES_TO, confidence=0.8,
            ))
            store.add_edge(KGEdge(
                src_id=f"concept:{i}", dst_id=f"concept:{(i + 100) % 500}",
                edge_type=EdgeType.GOVERNS, confidence=0.6,
            ))

        stats = store.stats()
        total_edges = sum(v for k, v in stats.items() if k.startswith("edge:"))
        assert total_edges == 1000

    def test_search_scales(self, store):
        from src.knowledge.ontology import Concept

        for i in range(200):
            store.add_node(Concept(
                id=f"con:{i}", label=f"LGD Floor Variant {i}",
            ))

        results = store.search_nodes(label_contains="LGD Floor", limit=50)
        assert len(results) == 50

    def test_subgraph_with_many_hops(self, store):
        from src.knowledge.ontology import Concept, EdgeType, KGEdge

        # Build a chain: c:0 → c:1 → c:2 → ... → c:19
        for i in range(20):
            store.add_node(Concept(id=f"c:{i}", label=f"Chain Node {i}"))
        for i in range(19):
            store.add_edge(KGEdge(
                src_id=f"c:{i}", dst_id=f"c:{i + 1}",
                edge_type=EdgeType.RELATES_TO,
            ))

        neighbors = store.get_neighbors("c:0", hops=5)
        assert len(neighbors) >= 5


# ═══════════════════════════════════════════════════════════════════════════
# 4. LLM thinking-mode fix
# ═══════════════════════════════════════════════════════════════════════════


class TestThinkingModeStrip:
    """_strip_think_tags and _inject_no_think must handle all edge cases."""

    def test_strip_simple(self):
        from src.knowledge.llm_client import _strip_think_tags

        text = '<think>internal reasoning here</think>{"result": true}'
        assert _strip_think_tags(text) == '{"result": true}'

    def test_strip_multiline(self):
        from src.knowledge.llm_client import _strip_think_tags

        text = (
            "<think>\nLet me think about this...\n"
            "Step 1: extract entities\n"
            "Step 2: build relationships\n"
            "</think>\n"
            '{"entities": []}'
        )
        assert _strip_think_tags(text) == '{"entities": []}'

    def test_strip_no_tags(self):
        from src.knowledge.llm_client import _strip_think_tags

        text = '{"entities": [{"id": "test"}]}'
        assert _strip_think_tags(text) == text

    def test_strip_empty_think(self):
        from src.knowledge.llm_client import _strip_think_tags

        text = "<think></think>OK"
        assert _strip_think_tags(text) == "OK"

    def test_inject_no_think(self):
        from src.knowledge.llm_client import _inject_no_think

        msgs = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Extract entities."},
        ]
        result = _inject_no_think(msgs)
        assert result[0]["content"] == "You are a helper."  # Unchanged
        assert result[1]["content"].startswith("/no_think\n")

    def test_inject_no_think_empty(self):
        from src.knowledge.llm_client import _inject_no_think

        assert _inject_no_think([]) == []

    def test_safe_json_with_think_tags(self):
        from src.knowledge.llm_client import _safe_json

        text = '<think>reasoning</think>{"entities": [{"id": "x"}]}'
        result = _safe_json(text)
        assert result == {"entities": [{"id": "x"}]}

    def test_safe_json_only_think_no_json(self):
        from src.knowledge.llm_client import _safe_json

        text = "<think>lots of thinking but no JSON output</think>"
        result = _safe_json(text)
        assert "error" in result

    def test_is_thinking_model(self):
        from src.knowledge.llm_client import LocalLLMClient

        client = LocalLLMClient(ollama_model="qwen3.6:27b", prefer="stub")
        assert client._is_thinking_model()

        client2 = LocalLLMClient(ollama_model="qwen2.5:14b", prefer="stub")
        assert not client2._is_thinking_model()

        client3 = LocalLLMClient(ollama_model="deepseek-r1:70b", prefer="stub")
        assert client3._is_thinking_model()


# ═══════════════════════════════════════════════════════════════════════════
# 5. KuzuDB lock retry
# ═══════════════════════════════════════════════════════════════════════════


class TestKuzuLockRetry:
    """GraphStore should retry on lock errors."""

    def test_retry_succeeds_on_second_attempt(self, tmp_path):
        import kuzu
        from src.knowledge.graph_store import GraphStore

        call_count = [0]
        real_db_init = kuzu.Database.__init__

        def patched_init(self_db, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Could not set lock on file")
            real_db_init(self_db, *args, **kwargs)

        with patch.object(kuzu.Database, "__init__", patched_init):
            store = GraphStore.__new__(GraphStore)
            store._db_path = tmp_path / "retry.kuzu"
            store._db_path.parent.mkdir(parents=True, exist_ok=True)
            store._MAX_LOCK_RETRIES = 3
            store._LOCK_RETRY_DELAY = 0.01
            db = store._open_db_with_retry()
            assert call_count[0] == 2
            assert db is not None

    def test_retry_exhaustion_raises(self, tmp_path):
        import kuzu
        from src.knowledge.graph_store import GraphStore

        def always_fail(self_db, *args, **kwargs):
            raise RuntimeError("Could not set lock on file")

        with patch.object(kuzu.Database, "__init__", always_fail):
            store = GraphStore.__new__(GraphStore)
            store._db_path = tmp_path / "locked.kuzu"
            store._db_path.parent.mkdir(parents=True, exist_ok=True)
            store._MAX_LOCK_RETRIES = 2
            store._LOCK_RETRY_DELAY = 0.01
            with pytest.raises(RuntimeError, match="Could not open KuzuDB"):
                store._open_db_with_retry()


# ═══════════════════════════════════════════════════════════════════════════
# 6. .egp upload API endpoint
# ═══════════════════════════════════════════════════════════════════════════


class TestEGPUploadEndpoint:
    """Test the /agent/sas/upload-egp endpoint."""

    @pytest.fixture()
    def client(self, tmp_path):
        from unittest.mock import patch as _patch

        # Redirect SAS root to tmp_path
        with _patch("api.routers.agent._SAS_ROOT", tmp_path / "sas"):
            from api.main import app
            from fastapi.testclient import TestClient
            yield TestClient(app)

    def test_upload_egp_extracts_blocks(self, client, tmp_path):
        egp_bytes = _make_egp(n_tasks=5, lines_per_task=10)

        resp = client.post(
            "/agent/sas/upload-egp",
            files={"file": ("test_project.egp", egp_bytes, "application/octet-stream")},
            data={"version": "v3"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["blocks_extracted"] >= 5
        assert len(data["saved"]) >= 5
        assert data["version"] == "v3"

    def test_upload_non_egp_rejected(self, client):
        resp = client.post(
            "/agent/sas/upload-egp",
            files={"file": ("test.sas", b"DATA x; RUN;", "text/plain")},
            data={"version": "v3"},
        )
        assert resp.status_code == 400

    def test_upload_sas_accepts_egp_too(self, client):
        """The general /sas/upload endpoint should now accept .egp files."""
        egp_bytes = _make_egp(n_tasks=3, lines_per_task=5)

        resp = client.post(
            "/agent/sas/upload",
            files={"files": ("project.egp", egp_bytes, "application/octet-stream")},
            data={"version": "v2"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["saved"]) >= 3


# ═══════════════════════════════════════════════════════════════════════════
# 7. Full pipeline integration (graph builder with mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════


class TestGraphBuilderScale:
    """Graph builder should handle many regulation files without crashing."""

    @pytest.fixture()
    def builder(self, tmp_path):
        from src.knowledge.graph_store import GraphStore
        from src.knowledge.graph_builder import GraphBuilder
        from unittest.mock import MagicMock

        store = GraphStore(db_path=tmp_path / "scale.kuzu")

        # Mock LLM that returns valid extraction JSON
        mock_llm = MagicMock()
        mock_llm.detect_backend.return_value = "stub"

        entity_counter = [0]

        def chat_json_side_effect(system, user, **kwargs):
            """Return plausible extraction results for any input."""
            entity_counter[0] += 1
            i = entity_counter[0]
            return {
                "entities": [
                    {"id": f"concept_{i}", "type": "Concept",
                     "label": f"Concept {i}", "props": {}},
                    {"id": f"reg_{i}", "type": "Regulation",
                     "label": f"Regulation {i}", "props": {}},
                ],
                "relationships": [
                    {"source": f"concept_{i}", "target": f"reg_{i}",
                     "type": "DEFINITION_IN", "confidence": 0.9},
                ],
                "links": [],  # For auto-link calls
            }

        mock_llm.chat_json.side_effect = chat_json_side_effect

        # Mock embedding service
        mock_embed = MagicMock()
        mock_embed.available = False

        return GraphBuilder(store, None, mock_embed, mock_llm), store

    def test_ingest_50_regulation_files(self, builder, tmp_path):
        graph_builder, store = builder

        # Create 50 regulation markdown files
        reg_dir = tmp_path / "regulations"
        reg_dir.mkdir()
        for i in range(50):
            (reg_dir / f"article_{i:03d}.md").write_text(
                f"# Article {i}\n\n"
                f"Paragraph {i}.1: The entity shall maintain a minimum "
                f"LGD floor of {10 + i}% for segment {i}.\n\n"
                f"Paragraph {i}.2: This applies to all IRB exposures "
                f"classified under risk weight category {i}.\n"
            )

        result = graph_builder.build_from_directory(
            regulation_dir=str(reg_dir),
            schema_file=None,
        )

        # Should have processed all 50 files
        assert result["nodes"] >= 100  # 2 per file minimum
        stats = store.stats()
        assert stats.get("node:Concept", 0) >= 50

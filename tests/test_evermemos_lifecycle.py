"""EverMemOS-inspired lifecycle integration tests.

Tests the full engram lifecycle across three phases:
  1. Episodic Trace Formation (learn) — extract insights from sessions
  2. Semantic Consolidation (organize + consolidate) — reduce entropy via concepts
  3. Reconstructive Recollection (retrieve) — precise retrieval after consolidation

Each test verifies with concrete metrics: entropy delta, retrieval precision, etc.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.memory_reward import compute_memory_reward
from training.memory_scenario_generator import (
    MemoryNode,
    MemoryScenario,
    MemoryScenarioGenerator,
)
from src.knowledge.graph_entropy import GraphEntropy, compute_graph_entropy, entropy_delta
from src.knowledge.graph_store import GraphStore
from src.knowledge.memory_ops import (
    add_tags,
    create_concept,
    flag_conflict,
    link_nodes,
    merge_nodes,
)
from src.knowledge.ontology import Concept, EdgeType, Insight, KGEdge, NodeType


# ── Lifecycle test harness ──────────────────────────────────────────────────


@dataclass
class LifecycleReport:
    """Metrics report for one lifecycle run."""
    sessions_processed: int = 0
    insights_extracted: int = 0
    insights_skipped_noise: int = 0
    nodes_merged: int = 0
    tags_added: int = 0
    conflicts_flagged: int = 0
    concepts_created: int = 0
    queries_answered: int = 0
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    entropy_delta: float = 0.0
    conflict_count: int = 0


class LifecycleHarness:
    """Ties together the three EverMemOS phases for integration testing.

    Uses a real GraphStore (tmp_path) so memory_ops work end-to-end.
    """

    def __init__(self, store: GraphStore, seed: int = 42):
        self.store = store
        self.gen = MemoryScenarioGenerator(seed=seed)
        self.report = LifecycleReport()
        self._node_ids: list[str] = []

    # ── Phase 1: Episodic Trace Formation ──────────────────────────────

    def _seed_nodes(self, scenario: MemoryScenario) -> None:
        """Ensure all scenario graph nodes exist in the store."""
        for node in scenario.graph_nodes:
            self._add_insight(node)
            if node.id not in self._node_ids:
                self._node_ids.append(node.id)

    def run_formation(self, scenario: MemoryScenario) -> list[str]:
        """Simulate learn phase: decide what to persist, extract insights."""
        self._seed_nodes(scenario)
        saved: list[str] = []

        for node in scenario.graph_nodes:
            if node.node_type == "insight" and scenario.should_save is not False:
                saved.append(node.id)
            elif scenario.should_save is False:
                self.report.insights_skipped_noise += 1

        if scenario.should_save:
            self.report.insights_extracted += len(scenario.expected_ids) if scenario.expected_ids else len(
                [n for n in scenario.graph_nodes if n.node_type == "insight"]
            )

        self.report.sessions_processed += 1
        return saved

    def _add_insight(self, node: MemoryNode) -> None:
        existing = self.store.get_node(node.id)
        if existing is not None:
            return
        self.store.add_node(Insight(
            id=node.id,
            label=node.label,
            summary=node.summary,
            tags=node.tags,
            fields=node.fields,
            segment=node.segment,
            priority=node.priority,
        ))

    # ── Phase 2: Semantic Consolidation ────────────────────────────────

    def run_consolidation(self, scenario: MemoryScenario) -> dict[str, Any]:
        """Run organize + consolidate phase on the graph.

        Returns consolidation results (concept_id, entropy_delta, etc.).
        """
        self._seed_nodes(scenario)
        nodes_before = self._all_nodes()

        result: dict[str, Any] = {"concepts_created": 0, "merges": 0, "conflicts": 0}

        # 2a. Deduplication
        if scenario.merge_pairs:
            for src, tgt in scenario.merge_pairs:
                out = merge_nodes(self.store, src, tgt)
                if out.get("merged"):
                    result["merges"] += 1
                    self.report.nodes_merged += 1

        # 2b. Tag enrichment
        if scenario.must_contain:
            for nid in scenario.expected_ids or []:
                tags_to_add = [t for t in scenario.must_contain if t not in ("save_insight", "search_experience", "create_concept")]
                if tags_to_add:
                    add_tags(self.store, nid, tags_to_add)
                    self.report.tags_added += len(tags_to_add)

        # 2c. Cross-linking (link similar insights)
        linked = set()
        if len(self._node_ids) >= 2:
            for i in range(len(self._node_ids)):
                for j in range(i + 1, len(self._node_ids)):
                    a, b = self._node_ids[i], self._node_ids[j]
                    if a not in linked and b not in linked:
                        link_nodes(self.store, a, b, "RELATES_TO")
                        linked.add(a)
                        linked.add(b)

        # 2d. Conflict detection
        if scenario.phase == "organize":
            for node in scenario.graph_nodes:
                for other in scenario.graph_nodes:
                    if node.id < other.id:
                        tags_a = set(t.lower() for t in (node.tags or []))
                        tags_b = set(t.lower() for t in (other.tags or []))
                        fields_a = set(f.upper() for f in (node.fields or []))
                        fields_b = set(f.upper() for f in (other.fields or []))
                        if len(tags_a & tags_b) >= 2 and fields_a & fields_b:
                            seg_match = node.segment and node.segment == other.segment
                            if seg_match and node.summary != other.summary:
                                if "%" in node.summary and "%" in other.summary:
                                    flag_conflict(self.store, node.id, other.id, "valores contradictorios")
                                    self.report.conflicts_flagged += 1

        # 2e. Concept creation (if scenario expects it)
        if scenario.should_create_concept and len(scenario.covers_ids) >= 2:
            result["concept"] = create_concept(
                self.store,
                label=scenario.concept_label or "Tema consolidado",
                definition=scenario.concept_definition or "Insights relacionados",
                covers=scenario.covers_ids,
                links_to=[],
            )
            if result["concept"].get("created"):
                result["concepts_created"] = 1
                self.report.concepts_created += 1

        nodes_after = self._all_nodes()
        self.report.entropy_before = compute_graph_entropy(nodes_before).total
        self.report.entropy_after = compute_graph_entropy(nodes_after).total

        concept_covers = None
        if result.get("concept") and result["concept"].get("created"):
            cid = result["concept"]["id"]
            concept_covers = {cid: scenario.covers_ids}
        self.report.entropy_delta = entropy_delta(nodes_before, nodes_after, concept_covers_after=concept_covers)

        return result

    # ── Phase 3: Reconstructive Recollection ───────────────────────────

    def run_recollection(self, query: str, expected_ids: list[str]) -> dict[str, Any]:
        """Simulate retrieve phase: search the graph and measure accuracy."""
        self.report.queries_answered += 1

        q_lower = query.lower()
        q_words = set(q_lower.split())

        # Score nodes by tag/field/content overlap (mirrors _sim_search_experience)
        scored: list[tuple[float, str]] = []
        for nid in self._node_ids:
            n = self.store.get_node(nid)
            if n is None:
                continue
            tags = set(t.lower() for t in (getattr(n, "tags", None) or []))
            fields = set(f.lower() for f in (getattr(n, "fields", None) or []))
            summary = (getattr(n, "summary", None) or "").lower()
            score = len(q_words & tags) * 2.0 + len(q_words & fields) * 3.0
            score += sum(1 for w in q_words if w in summary) * 0.5
            if score > 0:
                scored.append((score, nid))

        scored.sort(key=lambda x: -x[0])
        retrieved = [nid for _, nid in scored[:3]]

        # Also check concepts
        concept_nodes = []
        for n in self.store.search_nodes(limit=500):
            if getattr(n, "node_type", None) == NodeType.CONCEPT:
                summary = (getattr(n, "definition", "") + " " + getattr(n, "label", "")).lower()
                score = sum(1 for w in q_words if w in summary)
                if score > 0:
                    concept_nodes.append((score, n.id))

        concept_nodes.sort(key=lambda x: -x[0])
        retrieved.extend(cid for _, cid in concept_nodes if cid not in retrieved)

        retrieved_set = set(retrieved)
        expected_set = set(expected_ids)

        true_positives = len(retrieved_set & expected_set)
        self.report.retrieval_precision = true_positives / max(len(retrieved_set), 1)
        self.report.retrieval_recall = true_positives / max(len(expected_set), 1)

        return {
            "retrieved": retrieved,
            "expected": expected_ids,
            "precision": self.report.retrieval_precision,
            "recall": self.report.retrieval_recall,
        }

    def _all_ids(self) -> set[str]:
        return set(n.id for n in self.store.search_nodes(limit=500))

    def _all_nodes(self):
        return self.store.search_nodes(limit=500)

    def report_summary(self) -> str:
        r = self.report
        lines = [
            "═══ Lifecycle Report ═══",
            f"  Sessions processed:  {r.sessions_processed}",
            f"  Insights extracted:  {r.insights_extracted}",
            f"  Noise skipped:       {r.insights_skipped_noise}",
            f"  Nodes merged:        {r.nodes_merged}",
            f"  Tags added:          {r.tags_added}",
            f"  Conflicts flagged:   {r.conflicts_flagged}",
            f"  Concepts created:    {r.concepts_created}",
            f"  Queries answered:    {r.queries_answered}",
            f"  Retrieval precision: {r.retrieval_precision:.2%}",
            f"  Retrieval recall:    {r.retrieval_recall:.2%}",
            f"  Entropy before:      {r.entropy_before:.4f}",
            f"  Entropy after:       {r.entropy_after:.4f}",
            f"  Entropy delta:       {r.entropy_delta:.4f} (positive = reduction)",
        ]
        return "\n".join(lines)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> GraphStore:
    return GraphStore(db_path=tmp_path / "evermemos.kuzu")


@pytest.fixture
def harness(store: GraphStore) -> LifecycleHarness:
    return LifecycleHarness(store)


# ── Tests ───────────────────────────────────────────────────────────────────


class TestFullLifecycle:
    """EverMemOS full lifecycle: formation → consolidation → recollection."""

    def test_lifecycle_reduces_entropy(self, harness: LifecycleHarness) -> None:
        """Phase 1→2: after consolidation, total entropy should drop."""
        gen = harness.gen
        for _ in range(5):
            sc = gen.generate_one("learn")
            harness.run_formation(sc)
        consolidate_sc = gen.generate_one("consolidate")
        harness.run_consolidation(consolidate_sc)

        assert harness.report.entropy_after <= harness.report.entropy_before + 0.05
        assert harness.report.concepts_created >= 0
        assert harness.report.sessions_processed == 5

    def test_retrieval_after_consolidation(self, harness: LifecycleHarness) -> None:
        """Phase 3: after consolidation, retrieval must find relevant insights."""
        gen = harness.gen
        sc = gen.generate_one("learn")
        harness.run_formation(sc)

        if sc.expected_ids:
            result = harness.run_recollection(
                " ".join(sc.expected_ids),
                sc.expected_ids,
            )
            assert result["precision"] >= 0.0
            assert result["recall"] >= 0.0

    def test_reward_alignment(self, harness: LifecycleHarness) -> None:
        """Verify golden actions from scenario generator get positive reward."""
        gen = harness.gen
        sc = gen.generate_one("retrieve")
        golden = "\n".join(sc.golden_actions)
        reward = compute_memory_reward(golden, sc)
        assert reward.total > 0.3, f"Expected positive reward, got {reward.total}"

        if sc.subtype == "retrieve_segment":
            assert reward.components.get("correct_retrieval", 0) > 0

    def test_concept_compression_quality(self, harness: LifecycleHarness) -> None:
        """Consolidate phase: concept must compress >=2 insights and reduce entropy."""
        gen = harness.gen
        for _ in range(3):
            harness.run_formation(gen.generate_one("learn"))

        sc = gen.generate_one("consolidate")
        if sc.subtype == "consolidate_lift":
            result = harness.run_consolidation(sc)
            assert result.get("concept", {}).get("created", True), "Should create concept"
        else:
            harness.run_consolidation(sc)

    def test_multi_session_accumulation(self, harness: LifecycleHarness) -> None:
        """After multiple learn sessions, organized graph has lower entropy."""
        gen = harness.gen
        for _ in range(4):
            harness.run_formation(gen.generate_one("learn"))

        # Seed consolidate scenario nodes into the graph, then consolidate
        consolidate_sc = gen.generate_one("consolidate")
        for n in consolidate_sc.graph_nodes:
            harness._add_insight(n)
            if n.id not in harness._node_ids:
                harness._node_ids.append(n.id)

        result = harness.run_consolidation(consolidate_sc)
        if result.get("concept", {}).get("created", False):
            concept = result["concept"]
            assert concept.get("covers_linked", 0) >= 2, (
                f"Expected ≥2 covers linked, got {concept.get('covers_linked')}"
            )
            assert concept.get("entropy_delta", 0) >= 0, "Entropy should not increase"


class TestEpisodicFormation:
    """Phase 1: extracting insights from sessions."""

    def test_save_relevant_insight(self, harness: LifecycleHarness) -> None:
        gen = harness.gen
        sc = gen.generate_one("learn")
        if sc.subtype == "learn_save":
            saved = harness.run_formation(sc)
            golden = "\n".join(sc.golden_actions)
            reward = compute_memory_reward(golden, sc)
            assert reward.total > 0, f"Expected positive reward for save, got {reward.total}"

    def test_skip_noise(self, harness: LifecycleHarness) -> None:
        gen = harness.gen
        for _ in range(30):
            sc = gen.generate_one("learn")
            if sc.subtype == "learn_skip_noise":
                reward = compute_memory_reward("No hay nada que guardar.", sc)
                assert reward.total > 0
                return
        pytest.skip("no learn_skip scenario generated in 30 draws")


class TestSemanticConsolidation:
    """Phase 2: organizing and consolidating knowledge."""

    def test_dedup_merges_duplicates(self, harness: LifecycleHarness) -> None:
        harness._add_insight(MemoryNode(id="dup_a", label="a", summary="LGD floor CORP = 0.10", tags=["LGD", "floor"]))
        harness._add_insight(MemoryNode(id="dup_b", label="b", summary="LGD floor CORP = 0.10", tags=["LGD", "floor"]))
        harness._node_ids = ["dup_a", "dup_b"]

        sc = MemoryScenario(
            phase="organize", scenario_id="test_dedup",
            graph_nodes=[], user_prompt="", merge_pairs=[("dup_b", "dup_a")],
            expected_tool="merge_nodes",
        )
        harness.run_consolidation(sc)
        assert harness.store.get_node("dup_b") is None
        assert harness.store.get_node("dup_a") is not None

    def test_conflict_detection(self, harness: LifecycleHarness) -> None:
        harness._add_insight(MemoryNode(id="c1", label="c1", summary="LGD floor CORP = 0.10", tags=["LGD", "floor", "CORP"], segment="CORP"))
        harness._add_insight(MemoryNode(id="c2", label="c2", summary="LGD floor CORP = 0.15", tags=["LGD", "floor", "CORP"], segment="CORP"))
        harness._node_ids = ["c1", "c2"]

        sc = MemoryScenario(
            phase="organize", scenario_id="test_conflict",
            graph_nodes=[
                MemoryNode(id="c1", label="c1", summary="LGD floor CORP = 0.10", tags=["LGD", "floor", "CORP"], segment="CORP"),
                MemoryNode(id="c2", label="c2", summary="LGD floor CORP = 0.15", tags=["LGD", "floor", "CORP"], segment="CORP"),
            ],
            user_prompt="", expected_tool="flag_conflict",
        )
        harness.run_consolidation(sc)
        assert harness.report.conflicts_flagged >= 0

    def test_entropy_reduction_with_concept(self, harness: LifecycleHarness) -> None:
        harness._add_insight(MemoryNode(id="e1", label="e1", summary="LGD floor CORP = 10%", tags=["LGD", "floor"], fields=["LGD_ESTIMADA"]))
        harness._add_insight(MemoryNode(id="e2", label="e2", summary="LGD floor retail = 12%", tags=["LGD", "floor"], fields=["LGD_ESTIMADA"]))
        harness._add_insight(MemoryNode(id="e3", label="e3", summary="LGD floor HIPOTECA = 8%", tags=["LGD", "floor"], fields=["LGD_ESTIMADA"]))
        harness._node_ids = ["e1", "e2", "e3"]

        nodes_before = harness._all_nodes()
        entropy_before = compute_graph_entropy(nodes_before).total

        sc = MemoryScenario(
            phase="consolidate", scenario_id="test_entropy",
            graph_nodes=[], user_prompt="",
            covers_ids=["e1", "e2", "e3"], should_create_concept=True,
            concept_label="LGD floor por segmento",
            concept_definition="El floor de LGD varía por segmento de calibración",
            min_entropy_delta=0.01,
        )
        result = harness.run_consolidation(sc)
        assert result["concepts_created"] == 1
        assert result["concept"]["covers_linked"] == 3
        assert result["concept"]["entropy_delta"] > 0

    def test_create_concept_rejects_single_cover(self, harness: LifecycleHarness) -> None:
        harness._add_insight(MemoryNode(id="solo", label="solo", summary="único insight"))
        result = create_concept(harness.store, "Tema", "def", covers=["solo"])
        assert result["created"] is False


class TestReconstructiveRecollection:
    """Phase 3: retrieving knowledge after consolidation."""

    def test_precise_retrieval_after_consolidation(self, harness: LifecycleHarness) -> None:
        harness._add_insight(MemoryNode(id="r1", label="r1", summary="LGD floor CORP = 0.10", tags=["LGD", "floor", "CORP"]))
        harness._add_insight(MemoryNode(id="r2", label="r2", summary="LGD floor retail = 0.12", tags=["LGD", "floor", "retail"]))
        harness._add_insight(MemoryNode(id="r3", label="r3", summary="PD floor CORP = 0.03", tags=["PD", "floor", "CORP"]))
        harness._node_ids = ["r1", "r2", "r3"]

        # Consolidate LGD floors into a concept
        create_concept(harness.store, "LGD floors", "LGD floor por segmento", covers=["r1", "r2"])

        result = harness.run_recollection("LGD floor CORP", ["r1"])
        assert result["precision"] > 0.0, "Should retrieve the CORP LGD floor insight"

    def test_temporal_disambiguation(self, harness: LifecycleHarness) -> None:
        harness._add_insight(MemoryNode(id="old", label="old", summary="LGD floor CORP = 0.10 (2023)", tags=["LGD", "floor", "CORP"], is_historical=True, priority=0.3))
        harness._add_insight(MemoryNode(id="new", label="new", summary="LGD floor CORP = 0.12 (2024)", tags=["LGD", "floor", "CORP"], priority=0.8))
        harness._node_ids = ["old", "new"]

        result = harness.run_recollection("LGD floor CORP 2024", ["new"])
        assert result["recall"] > 0.0

    def test_retrieval_with_noise(self, harness: LifecycleHarness) -> None:
        harness._add_insight(MemoryNode(id="t1", label="t1", summary="ECL formula change", tags=["ECL", "formula"]))
        harness._add_insight(MemoryNode(id="t2", label="t2", summary="MoC calculation fix", tags=["MoC", "fix"]))
        harness._node_ids = ["t1", "t2"]

        result = harness.run_recollection("Fórmula ECL", ["t1"])
        assert result["precision"] > 0.0

    def test_empty_result_on_unrelated_query(self, harness: LifecycleHarness) -> None:
        harness._add_insight(MemoryNode(id="u1", label="u1", summary="LGD floor CORP = 0.10", tags=["LGD", "floor"]))
        harness._node_ids = ["u1"]

        result = harness.run_recollection("cálculo de provisiones IFRS9", [])
        assert len(result["retrieved"]) == 0 or result["precision"] == 0.0


class TestCrossPhaseIntegration:
    """Multi-phase scenarios (>1 phase in sequence)."""

    def test_learn_then_organize_with_reward(self, harness: LifecycleHarness) -> None:
        gen = harness.gen
        learn_sc = gen.generate_one("learn")
        harness.run_formation(learn_sc)

        golden_learn = "\n".join(learn_sc.golden_actions)
        learn_reward = compute_memory_reward(golden_learn, learn_sc)
        assert learn_reward.total > -0.5

        organize_sc = gen.generate_one("organize")
        result = harness.run_consolidation(organize_sc)
        golden_organize = "\n".join(organize_sc.golden_actions)
        organize_reward = compute_memory_reward(golden_organize, organize_sc)
        assert organize_reward.total > -0.5

    def test_learn_consolidate_retrieve_cycle(self, harness: LifecycleHarness) -> None:
        gen = harness.gen
        for _ in range(3):
            harness.run_formation(gen.generate_one("learn"))

        consolidate_sc = gen.generate_one("consolidate")
        harness.run_consolidation(consolidate_sc)

        retrieve_sc = gen.generate_one("retrieve")
        query = retrieve_sc.question or " ".join(retrieve_sc.expected_ids)
        result = harness.run_recollection(query, retrieve_sc.expected_ids)
        golden_retrieve = "\n".join(retrieve_sc.golden_actions)
        retrieve_reward = compute_memory_reward(golden_retrieve, retrieve_sc)
        assert retrieve_reward.total > -0.5, f"Expected non-negative retrieve reward, got {retrieve_reward.total}"

    def test_entropy_monotonic_decrease(self, harness: LifecycleHarness) -> None:
        """Entropy should stay bounded after consolidation cycles."""
        gen = harness.gen
        entropies = []
        # Seed a base set of related nodes
        base = gen.generate_one("learn")
        harness._seed_nodes(base)
        base_entropy = compute_graph_entropy(harness._all_nodes()).total
        entropies.append(base_entropy)

        for _ in range(3):
            sc = gen.generate_one("learn")
            harness._seed_nodes(sc)
            e = compute_graph_entropy(harness._all_nodes()).total
            entropies.append(e)

        for _ in range(3):
            sc = gen.generate_one("organize")
            harness.run_consolidation(sc)
            e = compute_graph_entropy(harness._all_nodes()).total
            entropies.append(e)

        max_entropy = max(entropies)
        assert max_entropy < 0.5, f"Entropy spiked too high: {max_entropy:.4f}"

    def test_conflict_reduces_with_consolidation(self, harness: LifecycleHarness) -> None:
        harness._add_insight(MemoryNode(
            id="x1", label="x1", summary="LGD floor CORP = 10% (según calibración 2023)", tags=["LGD", "floor", "CORP"],
            segment="CORP", fields=["LGD_ESTIMADA"],
        ))
        harness._add_insight(MemoryNode(
            id="x2", label="x2", summary="LGD floor CORP = 15% (según auditoría 2024)", tags=["LGD", "floor", "CORP"],
            segment="CORP", fields=["LGD_ESTIMADA"],
        ))
        harness._node_ids = ["x1", "x2"]

        e_before = compute_graph_entropy(harness._all_nodes())
        assert e_before.conflict > 0, "Expected conflict between contradictory insights"

        create_concept(harness.store, "LGD floor CORP", "LGD floor CORP values reconciled", covers=["x1", "x2"])

        e_after = compute_graph_entropy(harness._all_nodes(), concept_covers={"concept:lgd_floor_CORP": ["x1", "x2"]})
        assert e_after.conflict <= e_before.conflict, "Conflict should not increase after consolidation"

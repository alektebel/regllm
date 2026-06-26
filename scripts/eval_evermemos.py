#!/usr/bin/env python3
"""EverMemOS-style lifecycle benchmark for the RegLLM knowledge graph.

Evaluates the full engram lifecycle (formation → consolidation → recollection)
with concrete metrics and report generation. Runs N episodes and aggregates.

Usage:
    python scripts/eval_evermemos.py --episodes 5 --seed 42
    python scripts/eval_evermemos.py --episodes 10 --report-only  # re-report last run

Output: scripts/eval_evermemos_report.json (+ terminal summary)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.memory_reward import compute_memory_reward
from training.memory_scenario_generator import (
    MemoryNode,
    MemoryScenarioGenerator,
)
from src.knowledge.graph_entropy import compute_graph_entropy, GraphEntropy
from src.knowledge.graph_store import GraphStore
from src.knowledge.memory_ops import (
    add_tags, create_concept, flag_conflict, link_nodes, merge_nodes,
)
from src.knowledge.ontology import Concept, EdgeType, Insight, NodeType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "scripts" / "eval_evermemos_report.json"


# ── Lifecycle runner ────────────────────────────────────────────────────────

class LifecycleRunner:
    """Runs a single lifecycle episode: learn → organize → consolidate → retrieve.

    Uses a temporary graph store (SQLite-backed via GraphStore with tmp_path).
    """

    def __init__(self, store_dir: Path, seed: int = 42):
        self.store = GraphStore(db_path=store_dir / "eval_evermemos.kuzu")
        self.gen = MemoryScenarioGenerator(seed=seed)
        self.node_ids: list[str] = []

    def add_insight(self, node: MemoryNode) -> None:
        if self.store.get_node(node.id) is not None:
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
        if node.id not in self.node_ids:
            self.node_ids.append(node.id)

    def run_learn(self, scenario) -> int:
        saved = 0
        for node in scenario.graph_nodes:
            if node.node_type == "insight" and scenario.should_save is not False:
                self.add_insight(node)
                saved += 1
        return saved

    def run_organize(self, scenario) -> dict:
        result: dict[str, Any] = {"merges": 0, "tags": 0, "conflicts": 0}
        if scenario.merge_pairs:
            for src, tgt in scenario.merge_pairs:
                out = merge_nodes(self.store, src, tgt)
                if out.get("merged"):
                    result["merges"] += 1
        if scenario.expected_ids:
            for nid in scenario.expected_ids:
                tags = [t for t in scenario.must_contain if t not in ("save_insight", "search_experience", "create_concept", "merge_nodes", "add_tags")]
                if tags:
                    add_tags(self.store, nid, tags)
                    result["tags"] += len(tags)
        return result

    def run_consolidate(self, scenario) -> dict:
        nodes_before = self.store.search_nodes(limit=500)
        result: dict[str, Any] = {"concept_created": False, "concept": None}
        if scenario.should_create_concept and len(scenario.covers_ids) >= 2:
            concept = create_concept(
                self.store,
                label=scenario.concept_label or "Tema consolidado",
                definition=scenario.concept_definition or "Insights relacionados",
                covers=scenario.covers_ids,
                links_to=[],
            )
            result["concept"] = concept
            result["concept_created"] = concept.get("created", False)
        nodes_after = self.store.search_nodes(limit=500)
        result["entropy_before"] = compute_graph_entropy(nodes_before).total
        result["entropy_after"] = compute_graph_entropy(nodes_after).total
        return result

    def run_retrieve(self, query: str, expected_ids: list[str]) -> dict:
        q_words = set(query.lower().split())
        scored: list[tuple[float, str]] = []
        for nid in self.node_ids:
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
        retrieved = [nid for _, nid in scored[:5]]

        concept_nodes = []
        for n in self.store.search_nodes(limit=500):
            if getattr(n, "node_type", None) == NodeType.CONCEPT:
                text = (getattr(n, "definition", "") + " " + getattr(n, "label", "")).lower()
                score = sum(1 for w in q_words if w in text)
                if score > 0:
                    concept_nodes.append((score, n.id))
        concept_nodes.sort(key=lambda x: -x[0])
        retrieved.extend(cid for _, cid in concept_nodes if cid not in retrieved)

        retrieved_set = set(retrieved)
        expected_set = set(expected_ids)
        tp = len(retrieved_set & expected_set)
        return {
            "retrieved": retrieved,
            "expected": expected_ids,
            "precision": tp / max(len(retrieved_set), 1),
            "recall": tp / max(len(expected_set), 1) if expected_set else 1.0,
        }


# ── Episode definition ──────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    episode: int = 0
    seed: int = 0
    formation_saved: int = 0
    consolidation_created: bool = False
    consolidation_entropy_before: float = 0.0
    consolidation_entropy_after: float = 0.0
    consolidation_entropy_delta: float = 0.0
    retrieval_query: str = ""
    retrieval_expected: list[str] = field(default_factory=list)
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    reward: float = 0.0
    duration_ms: float = 0.0


@dataclass
class BenchmarkReport:
    episodes: list[EpisodeResult] = field(default_factory=list)
    timestamp: str = ""
    total_episodes: int = 0
    avg_formation_saved: float = 0.0
    avg_entropy_delta: float = 0.0
    avg_retrieval_precision: float = 0.0
    avg_retrieval_recall: float = 0.0
    avg_reward: float = 0.0
    total_concepts_created: int = 0
    concepts_created_rate: float = 0.0


def run_episode(episode: int, seed: int, tmp_dir: Path) -> EpisodeResult:
    """Run one full lifecycle and return metrics."""
    result = EpisodeResult(episode=episode, seed=seed)
    t0 = time.perf_counter()

    runner = LifecycleRunner(tmp_dir, seed=seed)

    # ── Phase 1: Learn — seed all scenario nodes into graph ─────────
    learn_sc = runner.gen.generate_one("learn")
    for node in learn_sc.graph_nodes:
        runner.add_insight(node)
    result.formation_saved = len([n for n in learn_sc.graph_nodes if n.node_type == "insight"])

    # ── Phase 2: Consolidate — also seed its nodes first ────────────
    consolidate_sc = runner.gen.generate_one("consolidate")
    for node in consolidate_sc.graph_nodes:
        runner.add_insight(node)
    consolidate_result = runner.run_consolidate(consolidate_sc)
    result.consolidation_created = consolidate_result.get("concept_created", False)
    result.consolidation_entropy_before = consolidate_result.get("entropy_before", 0.0)
    result.consolidation_entropy_after = consolidate_result.get("entropy_after", 0.0)
    result.consolidation_entropy_delta = result.consolidation_entropy_before - result.consolidation_entropy_after

    # ── Phase 3: Retrieve — seed nodes, then search ──────────────────
    retrieve_sc = runner.gen.generate_one("retrieve")
    for node in retrieve_sc.graph_nodes:
        runner.add_insight(node)
    query = retrieve_sc.question or " ".join(retrieve_sc.expected_ids[:3])
    result.retrieval_query = query[:100]
    result.retrieval_expected = retrieve_sc.expected_ids
    retrieval_result = runner.run_retrieve(query, retrieve_sc.expected_ids)
    result.retrieval_precision = retrieval_result["precision"]
    result.retrieval_recall = retrieval_result["recall"]

    # ── Golden reward ────────────────────────────────────────────────
    golden = "\n".join(retrieve_sc.golden_actions)
    reward = compute_memory_reward(golden, retrieve_sc)
    result.reward = reward.total

    result.duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result


def run_benchmark(episodes: int, seed: int) -> BenchmarkReport:
    """Run N lifecycle episodes and aggregate metrics."""
    import tempfile

    results: list[EpisodeResult] = []
    for ep in range(episodes):
        ep_seed = seed + ep * 7
        with tempfile.TemporaryDirectory() as td:
            result = run_episode(ep, ep_seed, Path(td))
            results.append(result)

        sys.stdout.write(f"  [{ep + 1}/{episodes}] "
                         f"Δentropy={result.consolidation_entropy_delta:+.4f} "
                         f"P@{result.retrieval_precision:.0%} "
                         f"R@{result.retrieval_recall:.0%} "
                         f"R={result.reward:+.3f}\n")
        sys.stdout.flush()

    created_count = sum(1 for r in results if r.consolidation_created)
    report = BenchmarkReport(
        episodes=results,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        total_episodes=episodes,
        avg_formation_saved=sum(r.formation_saved for r in results) / max(episodes, 1),
        avg_entropy_delta=sum(r.consolidation_entropy_delta for r in results) / max(episodes, 1),
        avg_retrieval_precision=sum(r.retrieval_precision for r in results) / max(episodes, 1),
        avg_retrieval_recall=sum(r.retrieval_recall for r in results) / max(episodes, 1),
        avg_reward=sum(r.reward for r in results) / max(episodes, 1),
        total_concepts_created=created_count,
        concepts_created_rate=created_count / max(episodes, 1),
    )
    return report


def print_report(report: BenchmarkReport) -> None:
    print()
    print("═══ EverMemOS Lifecycle Benchmark Report ═══")
    print(f"  Episodes:              {report.total_episodes}")
    print(f"  Timestamp:             {report.timestamp}")
    print(f"  Avg formation saved:   {report.avg_formation_saved:.1f}")
    print(f"  Concepts created:      {report.total_concepts_created} / {report.total_episodes} "
          f"({report.concepts_created_rate:.0%})")
    print(f"  Avg entropy delta:     {report.avg_entropy_delta:+.4f} "
          f"{'(reduction)' if report.avg_entropy_delta > 0 else '(increase!)'}")
    print(f"  Avg retrieval prec:    {report.avg_retrieval_precision:.2%}")
    print(f"  Avg retrieval recall:  {report.avg_retrieval_recall:.2%}")
    print(f"  Avg golden reward:     {report.avg_reward:+.3f}")
    print()

    # Highlight concerns
    if report.avg_entropy_delta <= 0:
        print("  ⚠ Consolidation is NOT reducing entropy on average.")
    if report.avg_retrieval_precision < 0.5:
        print("  ⚠ Retrieval precision is low (<50%). Consider better disambiguation.")
    if report.avg_retrieval_recall < 0.5:
        print("  ⚠ Retrieval recall is low (<50%). Consider broader indexing.")
    if report.concepts_created_rate < 0.3:
        print("  ℹ Low concept creation rate. Consolidation scenarios may be sparse.")


def main() -> None:
    parser = argparse.ArgumentParser(description="EverMemOS lifecycle benchmark")
    parser.add_argument("--episodes", type=int, default=5, help="Number of lifecycle episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report-only", action="store_true", help="Re-print last report")
    args = parser.parse_args()

    if args.report_only:
        if REPORT_PATH.exists():
            data = json.loads(REPORT_PATH.read_text())
            report = BenchmarkReport(**data)
            report.episodes = [EpisodeResult(**e) for e in data["episodes"]]
            print_report(report)
        else:
            print(f"No previous report found at {REPORT_PATH}")
        return

    print(f"Running EverMemOS benchmark: {args.episodes} episodes (seed={args.seed})...")
    report = run_benchmark(args.episodes, args.seed)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print(f"Report saved → {REPORT_PATH}")

    print_report(report)


if __name__ == "__main__":
    main()

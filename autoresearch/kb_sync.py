"""Knowledge base synchronizer — grows the KG from research outcomes.

After each experiment iteration, extracts insights, quirks, and new
connections from the evaluation results and stores them as nodes/edges
in the knowledge graph. This is the "self-organizing" part of the KB.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(_PROJECT_ROOT))

from src.knowledge.graph_store import GraphStore
from src.knowledge.ontology import (
    EdgeType,
    Experience,
    Insight,
    KGEdge,
    NodeType,
    Quirk,
)

logger = logging.getLogger(__name__)


class KBSynchronizer:
    """Synchronizes research experiment results into the knowledge graph.

    Each experiment's evaluation results become Experience nodes.
    Insights (patterns, successes, failures) become Insight nodes linked
    to the experience. Quirks (unexpected model behaviors) become Quirk
    nodes linked to relevant fields.
    """

    def __init__(self, graph_db: str | Path = "data/knowledge/regllm.kuzu") -> None:
        self._store = GraphStore(str(graph_db))

    def sync_experiment(
        self,
        iteration: int,
        eval_summary: dict[str, Any],
        synthetic_stats: dict[str, int],
        training_params: dict[str, Any],
        insights: list[dict[str, Any]] | None = None,
    ) -> dict[str, int]:
        """Record an experiment as an Experience node + linked Insights.

        Returns counts of {nodes, edges} created.
        """
        counts = {"nodes": 0, "edges": 0}

        ts = datetime.now(timezone.utc).isoformat()
        exp_id = f"exp:research_iter_{iteration:03d}"

        summary_text = json.dumps({
            "iteration": iteration,
            "eval_accuracy": eval_summary.get("accuracy", 0.0),
            "synthetic_examples": sum(synthetic_stats.values()),
            "training_params": training_params,
            "per_tool_accuracy": eval_summary.get("per_tool", {}),
        }, ensure_ascii=False)

        self._store.add_node(Experience(
            id=exp_id,
            label=f"[research] Iteration {iteration}: "
                  f"acc={eval_summary.get('accuracy', 0.0):.2%}",
            session_id=f"research_{iteration}",
            question=f"Research iteration {iteration}",
            answer_summary=summary_text,
            timestamp=ts,
        ))
        counts["nodes"] += 1

        # Create insight nodes from evaluation patterns
        for ins in (insights or []):
            ins_id = self._store_insight(exp_id, ins, counts)

        # Create tool-specific insight nodes
        per_tool = eval_summary.get("per_tool", {})
        for tool_name, tool_stats in per_tool.items():
            if isinstance(tool_stats, dict):
                acc = tool_stats.get("accuracy", 0.0)
                total = int(tool_stats.get("total", 0))
                if total < 3:
                    continue
                if acc < 0.5:
                    insight = {
                        "summary": f"Tool {tool_name} has low accuracy "
                                   f"({acc:.0%}) after iteration {iteration}. "
                                   f"Need more training examples for this tool.",
                        "related_fields": [],
                        "tags": [tool_name, "low_accuracy"],
                    }
                    self._store_insight(exp_id, insight, counts)
                elif acc >= 0.9:
                    insight = {
                        "summary": f"Tool {tool_name} achieved high accuracy "
                                   f"({acc:.0%}) at iteration {iteration}.",
                        "related_fields": [],
                        "tags": [tool_name, "high_accuracy"],
                    }
                    self._store_insight(exp_id, insight, counts)

        logger.info(
            "KB sync for iter %d: %d nodes, %d edges",
            iteration, counts["nodes"], counts["edges"],
        )
        return counts

    def _store_insight(
        self, exp_id: str, ins: dict[str, Any], counts: dict[str, int],
    ) -> str:
        summary = ins.get("summary", "")
        if not summary:
            return ""
        ins_id = f"insight:research_{hashlib.sha256(summary.encode()).hexdigest()[:12]}"

        self._store.add_node(Insight(
            id=ins_id,
            label=summary[:100],
            summary=summary,
            tags=ins.get("tags", []),
        ))
        counts["nodes"] += 1

        self._store.add_edge(KGEdge(
            src_id=ins_id, dst_id=exp_id,
            edge_type=EdgeType.DISCOVERED_IN,
        ))
        counts["edges"] += 1

        # Link to related columns
        for field in ins.get("related_fields", []):
            col_id = f"col:{field.upper()}"
            concept_id = f"concept:{field.upper()}"
            target = col_id if self._store.get_node(col_id) else concept_id
            if self._store.get_node(target):
                self._store.add_edge(KGEdge(
                    src_id=ins_id, dst_id=target,
                    edge_type=EdgeType.RELATES_TO,
                ))
                counts["edges"] += 1

        return ins_id

    def get_research_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve past research experiments from the KG."""
        nodes = self._store.search_nodes(
            node_types=[NodeType.EXPERIENCE],
            label_contains="[research]",
            limit=limit,
        )
        return [n.model_dump() for n in nodes]

    def store_model_artifact(
        self,
        iteration: int,
        artifact_path: str,
        artifact_type: str,
        metrics: dict[str, float],
    ) -> None:
        """Record a trained model artifact as an Insight node."""
        ins_id = f"insight:model_iter_{iteration:03d}"
        self._store.add_node(Insight(
            id=ins_id,
            label=f"[model] Iteration {iteration}: {artifact_type}",
            summary=json.dumps({"path": artifact_path, "type": artifact_type, "metrics": metrics}),
            tags=["model", artifact_type, f"iter_{iteration}"],
            priority=0.8,
        ))
        logger.info("Model artifact stored: %s → %s", artifact_type, ins_id)

    @property
    def store(self) -> GraphStore:
        return self._store

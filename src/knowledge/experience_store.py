"""Experience store — persists what the agent learns from validations.

Each validation session can produce ``Experience`` and ``Insight`` nodes
that are linked into the knowledge graph. The store enables the agent
to get better over time by recalling past learnings.

Usage::

    es = ExperienceStore(graph_store)
    es.record_experience(
        session_id="abc123",
        question="Why is LGD different?",
        answer_summary="LGD floor changed from 0.45 to 0.50",
        related_fields=["LGD_ESTIMADA", "LGD_FLOOR_APLICADO"],
        related_articles=["art_15_dotaciones_minimas"],
        insights=["CORP LGD floor increased by 5pp in Circular 4/2022"],
    )
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any

from .graph_store import GraphStore
from .ontology import (
    EdgeType,
    Experience,
    Insight,
    KGEdge,
    KGNode,
    NodeType,
)

logger = logging.getLogger(__name__)


class ExperienceStore:
    """Manages experience and insight nodes in the knowledge graph."""

    def __init__(self, store: GraphStore) -> None:
        self._store = store

    def record_experience(
        self,
        session_id: str,
        question: str,
        answer_summary: str,
        related_fields: list[str] | None = None,
        related_articles: list[str] | None = None,
        insights: list[str] | None = None,
    ) -> str:
        """Record a validation experience and return its node ID."""
        ts = datetime.now().isoformat()
        exp_id = f"exp:{hashlib.sha256(f'{session_id}:{ts}'.encode()).hexdigest()[:12]}"

        self._store.add_node(Experience(
            id=exp_id,
            label=question[:100],
            session_id=session_id,
            question=question,
            answer_summary=answer_summary,
            timestamp=ts,
        ))

        # Link to related columns/concepts
        for field in (related_fields or []):
            self._link_to_field(exp_id, field)

        # Link to related regulations
        for article in (related_articles or []):
            self._link_to_regulation(exp_id, article)

        # Create insight nodes
        for insight_text in (insights or []):
            self._create_insight(exp_id, insight_text, related_fields or [])

        return exp_id

    def search(self, query: str, k: int = 5) -> list[KGNode]:
        """Search experiences and insights by label."""
        return self._store.search_nodes(
            node_types=[NodeType.EXPERIENCE, NodeType.INSIGHT],
            label_contains=query,
            limit=k,
        )

    def get_related(self, node_id: str, hops: int = 2) -> dict[str, Any]:
        """Get the graph neighborhood of an experience/insight."""
        return self._store.subgraph(node_id, hops=hops)

    def _link_to_field(self, exp_id: str, field_name: str) -> None:
        upper = field_name.upper()
        for prefix in ("col:", "concept:"):
            target_id = f"{prefix}{upper}"
            if self._store.get_node(target_id):
                self._store.add_edge(KGEdge(
                    src_id=exp_id,
                    dst_id=target_id,
                    edge_type=EdgeType.RELATES_TO,
                ))
                return

    def _link_to_regulation(self, exp_id: str, article: str) -> None:
        reg_nodes = self._store.search_nodes(
            node_types=[NodeType.REGULATION],
            label_contains=article,
            limit=1,
        )
        if reg_nodes:
            self._store.add_edge(KGEdge(
                src_id=exp_id,
                dst_id=reg_nodes[0].id,
                edge_type=EdgeType.RELATES_TO,
            ))

    def _create_insight(
        self,
        exp_id: str,
        insight_text: str,
        related_fields: list[str],
    ) -> str:
        ts = datetime.now().isoformat()
        ins_id = f"insight:{hashlib.sha256(f'{insight_text}:{ts}'.encode()).hexdigest()[:12]}"

        self._store.add_node(Insight(
            id=ins_id,
            label=insight_text[:100],
            summary=insight_text,
        ))

        # Link insight to experience
        self._store.add_edge(KGEdge(
            src_id=ins_id,
            dst_id=exp_id,
            edge_type=EdgeType.DISCOVERED_IN,
        ))

        # Link insight to fields
        for field in related_fields:
            self._link_to_field(ins_id, field)

        return ins_id

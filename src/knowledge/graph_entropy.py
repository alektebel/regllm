"""Graph entropy metrics for experience memory consolidation.

Higher entropy = more redundant, fragmented, or conflicting knowledge.
Concept formation should reduce total entropy by compressing related insights.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence


class _NodeLike(Protocol):
    id: str
    summary: str
    tags: list[str]
    fields: list[str]
    segment: str
    node_type: str
    priority: float


@dataclass(frozen=True)
class GraphEntropy:
    """Decomposed entropy (all components in [0, 1]; total is weighted sum)."""
    total: float
    structure: float
    redundancy: float
    conflict: float

    @property
    def delta_ready(self) -> float:
        """Single scalar for before/after comparisons."""
        return self.total


_PCT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _node_tags(n: Any) -> set[str]:
    tags = getattr(n, "tags", None) or (n.get("tags") if isinstance(n, dict) else [])
    return {t.lower() for t in tags}


def _node_fields(n: Any) -> set[str]:
    fields = getattr(n, "fields", None) or (n.get("fields") if isinstance(n, dict) else [])
    return {f.upper() for f in fields}


def _node_id(n: Any) -> str:
    return str(getattr(n, "id", None) or n.get("id", ""))


def _node_summary(n: Any) -> str:
    return str(getattr(n, "summary", None) or n.get("summary", ""))


def _node_segment(n: Any) -> str:
    seg = getattr(n, "segment", None)
    if not seg and isinstance(n, dict):
        seg = n.get("segment", "")
    return str(seg or "").upper()


def _node_type(n: Any) -> str:
    return str(getattr(n, "node_type", None) or n.get("type", n.get("node_type", "insight"))).lower()


def _covered_by_same_concept(
    a_id: str,
    b_id: str,
    concept_covers: dict[str, list[str]] | None,
) -> bool:
    if not concept_covers:
        return False
    for covers in concept_covers.values():
        cover_set = set(covers)
        if a_id in cover_set and b_id in cover_set:
            return True
    return False


def compute_graph_entropy(
    nodes: Sequence[Any],
    *,
    concept_covers: dict[str, list[str]] | None = None,
    weights: tuple[float, float, float] = (0.25, 0.45, 0.30),
) -> GraphEntropy:
    """Compute H(G) for a memory subgraph.

    ``concept_covers`` maps concept_id → insight ids already abstracted.
    Pairs covered by the same concept do not contribute to redundancy.
    """
    insights = [n for n in nodes if _node_type(n) not in ("concept", "regulation")]
    n = len(insights)
    if n == 0:
        return GraphEntropy(total=0.0, structure=0.0, redundancy=0.0, conflict=0.0)

    # Structural complexity grows with uncollapsed insight count
    structure = min(1.0, math.log2(max(n, 1)) / math.log2(max(n + 5, 6)))

    redundant_pairs = 0
    conflict_pairs = 0
    comparable_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            a, b = insights[i], insights[j]
            a_id, b_id = _node_id(a), _node_id(b)
            if _covered_by_same_concept(a_id, b_id, concept_covers):
                continue

            tags_a, tags_b = _node_tags(a), _node_tags(b)
            fields_a, fields_b = _node_fields(a), _node_fields(b)
            tag_sim = _jaccard(tags_a, tags_b)
            field_overlap = bool(fields_a & fields_b)

            if tag_sim >= 0.4 and field_overlap:
                comparable_pairs += 1
                sum_a, sum_b = _node_summary(a), _node_summary(b)
                if sum_a != sum_b and (tag_sim >= 0.5 or field_overlap):
                    redundant_pairs += 1

                seg_a, seg_b = _node_segment(a), _node_segment(b)
                same_scope = (
                    (seg_a and seg_a == seg_b)
                    or (not seg_a and not seg_b and tag_sim >= 0.6)
                )
                if same_scope and field_overlap:
                    pct_a = _PCT_RE.findall(sum_a)
                    pct_b = _PCT_RE.findall(sum_b)
                    if pct_a and pct_b and pct_a != pct_b:
                        conflict_pairs += 1

    max_pairs = max(1, n * (n - 1) // 2)
    redundancy = redundant_pairs / max_pairs
    conflict = conflict_pairs / max_pairs

    w_s, w_r, w_c = weights
    total = min(1.0, w_s * structure + w_r * redundancy + w_c * conflict)
    return GraphEntropy(
        total=round(total, 4),
        structure=round(structure, 4),
        redundancy=round(redundancy, 4),
        conflict=round(conflict, 4),
    )


def entropy_delta(
    nodes_before: Sequence[Any],
    nodes_after: Sequence[Any],
    *,
    concept_covers_after: dict[str, list[str]] | None = None,
) -> float:
    """Positive delta means entropy decreased (good consolidation)."""
    before = compute_graph_entropy(nodes_before)
    after = compute_graph_entropy(nodes_after, concept_covers=concept_covers_after)
    return round(before.total - after.total, 4)

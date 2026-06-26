"""Tests for graph entropy metrics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.knowledge.graph_entropy import compute_graph_entropy, entropy_delta
from training.memory_scenario_generator import MemoryNode


def test_entropy_drops_after_concept_covers():
    nodes = [
        MemoryNode("a", "LGD retail", "LGD floor retail = 0.45%", tags=["LGD"], fields=["LGD_ESTIMADA"], segment="retail"),
        MemoryNode("b", "LGD CORP", "LGD floor CORP = 0.50%", tags=["LGD"], fields=["LGD_ESTIMADA"], segment="CORP"),
        MemoryNode("c", "LGD hip", "LGD floor HIPOTECA = 0.30%", tags=["LGD"], fields=["LGD_ESTIMADA"], segment="HIPOTECA"),
    ]
    before = compute_graph_entropy(nodes)
    after = compute_graph_entropy(nodes, concept_covers={"concept:theme": ["a", "b", "c"]})
    assert after.total <= before.total
    delta = entropy_delta(nodes, nodes, concept_covers_after={"concept:theme": ["a", "b", "c"]})
    assert delta >= 0


def test_single_node_low_entropy():
    nodes = [MemoryNode("x", "solo", "un insight aislado", tags=["PD"], fields=["PD_ESTIMADA"])]
    h = compute_graph_entropy(nodes)
    assert h.redundancy == 0.0

"""Tests for procedural memory scenario generator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.memory_reward import compute_memory_reward
from training.memory_scenario_generator import MemoryScenarioGenerator


def test_generate_batch_mix():
    gen = MemoryScenarioGenerator(seed=0)
    batch = gen.generate_batch(30)
    phases = {s.phase for s in batch}
    assert "learn" in phases and "retrieve" in phases
    assert len(batch) == 30


def test_golden_retrieve_reward():
    gen = MemoryScenarioGenerator(seed=1)
    sc = gen.generate_one("retrieve")
    golden = "\n".join(sc.golden_actions)
    r = compute_memory_reward(golden, sc)
    assert r.total > 0.3, f"expected positive reward, got {r.total}"


def test_learn_skip_no_save():
    gen = MemoryScenarioGenerator(seed=2)
    for _ in range(20):
        sc = gen.generate_one("learn")
        if sc.subtype == "learn_skip_noise":
            r = compute_memory_reward("No hay nada que guardar.", sc)
            assert r.total > 0
            return
    raise AssertionError("no learn_skip scenario in 20 draws")


def test_consolidate_lift_golden_reward():
    gen = MemoryScenarioGenerator(seed=3)
    for _ in range(30):
        sc = gen.generate_one("consolidate")
        if sc.subtype == "consolidate_lift":
            golden = "\n".join(sc.golden_actions)
            r = compute_memory_reward(golden, sc)
            assert r.total > 0.5, f"expected strong reward, got {r.total}"
            assert sc.covers_ids and len(sc.covers_ids) >= 2
            return
    raise AssertionError("no consolidate_lift scenario in 30 draws")


def test_consolidate_skip_no_concept():
    gen = MemoryScenarioGenerator(seed=4)
    sc = gen.generate_one("consolidate")
    if sc.subtype != "consolidate_skip":
        for _ in range(20):
            sc = gen.generate_one("consolidate")
            if sc.subtype == "consolidate_skip":
                break
    r = compute_memory_reward("# No crear concepto", sc)
    assert r.total > 0

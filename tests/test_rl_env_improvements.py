"""Tests for RL bug environment improvements."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.curriculum import STAGES
from training.rl_env import build_prompt, compute_reward, generate_training_batch
from training.tool_selection_tasks import generate_tool_selection_batch
from training.toy_lgd_levels import ToyLgdLevelCatalog, load_level_bug, LEVEL_SPECS


def test_curated_levels_load():
    cat = ToyLgdLevelCatalog()
    assert len(cat.levels) >= 3, f"expected >=3 levels, got {cat.level_ids}"


def test_level_01_varswap_diff():
    spec = next(s for s in LEVEL_SPECS if s.level_id == "01_easy_varswap")
    bug = load_level_bug(spec)
    assert bug is not None
    assert "LGD_ESTIMADA" in bug.changed_fields
    assert bug.correct_values["LGD_ESTIMADA"] != bug.buggy_values["LGD_ESTIMADA"]


def test_tool_selection_reward():
    stage = STAGES[0]
    bugs = generate_tool_selection_batch(6, seed=0)
    reg = next(b for b in bugs if b.task_kind == "regulation")
    good = '<tool_call>{"name": "search_regulation", "arguments": {"query": "LGD hipoteca"}}</tool_call>'
    bad = '<tool_call>{"name": "trace_dependencies", "arguments": {"target": "ECL"}}</tool_call>'
    r_good = compute_reward(good, reg, stage).total
    r_bad = compute_reward(bad, reg, stage).total
    assert r_good > r_bad


def test_stage0_batch_mix():
    batch = generate_training_batch(20, STAGES[0], seed=1)
    assert len(batch) == 20
    sources = {b.source for b, _ in batch}
    assert "tool_task" in sources


def test_build_prompt_tool_task():
    bug = generate_tool_selection_batch(1, seed=2)[0]
    msgs = build_prompt(bug, STAGES[0])
    assert msgs[1]["role"] == "user"
    assert len(msgs[1]["content"]) > 10

"""Tests for gold trace and trace_path reward."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.bug_generator import make_toy_generator
from training.curriculum import STAGES
from training.gold_trace import attach_gold_trace, compute_gold_trace, oracle_trace_completion
from training.rl_env import build_prompt, compute_reward


def test_gold_trace_on_procedural_bug():
    gen = make_toy_generator(seed=42, max_depth=2)
    bug = None
    for _ in range(30):
        b = gen.generate()
        if b and b.depth >= 1:
            bug = b
            break
    assert bug is not None, "could not generate depth>=1 bug"
    assert bug.gold_trace.get("symptom_field")
    assert bug.gold_trace.get("path_fields")
    assert bug.mutation.step_name in bug.gold_trace.get("path_steps", []) or bug.gold_trace.get("root_step")


def test_oracle_beats_no_tools_on_multi_depth():
    gen = make_toy_generator(seed=7, max_depth=3)
    bug = None
    for _ in range(40):
        b = gen.generate()
        if b and b.depth >= 1:
            bug = attach_gold_trace(b)
            break
    assert bug is not None
    stage = STAGES[2]  # multi_depth
    oracle = oracle_trace_completion(bug)
    bad = "PASO: work.final\nCAUSA: no sé."
    r_o = compute_reward(oracle, bug, stage)
    r_b = compute_reward(bad, bug, stage)
    assert r_o.components.get("trace_path", 0) > r_b.components.get("trace_path", 0)
    assert r_o.total > r_b.total


def test_depth_hides_sas_snippet_in_prompt():
    gen = make_toy_generator(seed=99, max_depth=3)
    bug = None
    for _ in range(40):
        b = gen.generate()
        if b and b.depth >= 1:
            bug = attach_gold_trace(b)
            break
    assert bug is not None
    msgs = build_prompt(bug, STAGES[2])
    assert "```sas" not in msgs[1]["content"]
    assert bug.gold_trace.get("symptom_field", "") in msgs[1]["content"] or bug.changed_fields[0] in msgs[1]["content"]


def test_single_depth_shows_code_at_depth_zero():
    gen = make_toy_generator(seed=11, max_depth=0)
    bugs = gen.generate_batch(5)
    assert bugs
    bug = attach_gold_trace(bugs[0])
    msgs = build_prompt(bug, STAGES[1])
    assert "```sas" in msgs[1]["content"]

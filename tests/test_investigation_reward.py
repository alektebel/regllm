"""Tests for Phase 1 investigation reward."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_one_scenario():
    from training.investigation_scenarios import load_scenarios_from_eval
    scenarios = load_scenarios_from_eval()
    assert scenarios
    return scenarios[0]


def test_oracle_beats_hallucination():
    from training.investigation_env import oracle_completion, compute_investigation_reward

    sc = _load_one_scenario()
    oracle = oracle_completion(sc)
    bad = "OR_EADIL según mi conocimiento causa el missing sin trazar nada."
    r_good = compute_investigation_reward(oracle, sc)
    r_bad = compute_investigation_reward(bad, sc)
    assert r_good.total > r_bad.total
    assert r_good.components["tool_order"] >= r_bad.components["tool_order"]


def test_experience_only_penalized_on_bug_type():
    from training.investigation_env import compute_investigation_reward
    from training.investigation_scenarios import load_scenarios_from_eval

    bug_scenarios = [s for s in load_scenarios_from_eval() if s.scenario_type == "missing_value"]
    if not bug_scenarios:
        return
    sc = bug_scenarios[0]
    completion = '<tool_call>{"name": "search_experience", "arguments": {"query": "x"}}</tool_call>\nAlgo falla.'
    r = compute_investigation_reward(completion, sc)
    assert r.components["tool_order"] < 0.5


def test_grounding_prefers_fields_present_in_tool_context():
    from training.investigation_env import compute_investigation_reward
    from training.investigation_scenarios import load_scenarios_from_eval

    sc = next(s for s in load_scenarios_from_eval() if s.scenario_id == "eval_020")
    tools = "\n".join(
        f'<tool_call>{{"name": "{t}", "arguments": {json.dumps(a)}}}</tool_call>'
        for t, a in sc.tool_steps[:2]
    )
    in_ctx = "Causa raíz: LGD_ESTIMADA missing en la cadena → MoC y ECL missing. proj_03.sas:33."
    not_in_ctx = "Causa: SW_FUSION=1 según hipótesis sin evidencia en tools."
    r_good = compute_investigation_reward(tools + "\n" + in_ctx, sc)
    r_bad = compute_investigation_reward(tools + "\n" + not_in_ctx, sc)
    assert r_good.components["grounding"] > r_bad.components["grounding"]


def test_grpo_advantage_spread():
    from training.simulate_investigation_grpo import _grpo_advantages

    rewards = [0.9, 0.5, -0.2, -0.8]
    adv = _grpo_advantages(rewards)
    assert max(adv) > 0 and min(adv) < 0
    assert abs(sum(adv)) < 0.01  # roughly zero-sum

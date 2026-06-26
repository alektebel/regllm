"""Verifiable rewards for procedural memory scenarios (no LLM judge)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from training.memory_scenario_generator import MemoryScenario


@dataclass
class MemoryRewardBreakdown:
    total: float
    components: dict[str, float]
    tool_calls: list[str]
    retrieved_ids: list[str]


def extract_tool_calls(text: str) -> list[dict]:
    """Parse tool calls from model completion (reuse rl_env patterns)."""
    calls: list[dict] = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        try:
            calls.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
    if not calls:
        for m in re.finditer(
            r'\{"name"\s*:\s*"(\w+)"[^}]*"arguments"\s*:\s*(\{[^}]*\})',
            text,
        ):
            try:
                calls.append({"name": m.group(1), "arguments": json.loads(m.group(2))})
            except json.JSONDecodeError:
                calls.append({"name": m.group(1), "arguments": {}})
    if not calls:
        for tool in (
            "save_insight", "save_quirk", "save_feedback", "merge_nodes",
            "add_tags", "link_nodes", "flag_conflict", "delete_node",
            "create_concept", "link_to_concept",
            "search_experience",
        ):
            if tool in text:
                calls.append({"name": tool, "arguments": {}})
    return calls


def _text_lower(text: str) -> str:
    return text.lower()


def _covers_in_text(text: str, covers: list[str]) -> float:
    if not covers:
        return 0.0
    hits = sum(1 for cid in covers if cid.lower() in text.lower())
    return hits / len(covers)


def compute_memory_reward(completion: str, scenario: MemoryScenario) -> MemoryRewardBreakdown:
    """Rule-based reward in [-1, 1] aligned with scenario ground truth."""
    calls = extract_tool_calls(completion)
    tool_names = [c.get("name", "") for c in calls]
    text = _text_lower(completion)
    components: dict[str, float] = {}

    if scenario.phase == "learn":
        save_called = any(t in tool_names for t in ("save_insight", "save_quirk", "save_feedback"))
        if scenario.should_save:
            components["save_decision"] = 1.0 if save_called else -0.5
            if save_called and scenario.must_contain:
                hits = sum(1 for k in scenario.must_contain if k.lower() in text)
                components["extraction"] = hits / len(scenario.must_contain)
            elif save_called:
                components["extraction"] = 0.5
            else:
                components["extraction"] = 0.0
        else:
            components["save_decision"] = -0.8 if save_called else 1.0
            components["extraction"] = 1.0 if not save_called else 0.0

    elif scenario.phase == "organize":
        org_tools = {"merge_nodes", "add_tags", "link_nodes", "flag_conflict", "delete_node"}
        components["organize_action"] = 1.0 if org_tools & set(tool_names) else 0.0
        if scenario.merge_pairs:
            merged = 0
            for a, b in scenario.merge_pairs:
                if a.lower() in text and b.lower() in text:
                    merged += 1
            components["merge_correct"] = merged / len(scenario.merge_pairs)
        if scenario.expected_ids:
            components["target_node"] = (
                1.0 if any(i.lower() in text for i in scenario.expected_ids) else 0.0
            )

    elif scenario.phase == "consolidate":
        create_called = "create_concept" in tool_names or "create_concept" in text
        if scenario.should_create_concept:
            components["create_decision"] = 1.0 if create_called else -0.5
            if create_called and scenario.covers_ids:
                components["coverage"] = _covers_in_text(completion, scenario.covers_ids)
            if create_called and scenario.concept_label:
                label_hit = scenario.concept_label.lower()[:20] in text
                components["concept_label"] = 1.0 if label_hit else 0.3
            if create_called and scenario.must_contain:
                hits = sum(1 for k in scenario.must_contain if k.lower() in text)
                components["semantic_content"] = hits / len(scenario.must_contain)
            if create_called and len(scenario.covers_ids or []) < 2:
                components["over_abstraction"] = -1.0
        else:
            components["skip_decision"] = -0.8 if create_called else 1.0

    elif scenario.phase == "retrieve":
        search = "search_experience" in tool_names or "search_experience" in text
        components["search_called"] = 1.0 if search else 0.0
        retrieved: list[str] = []
        for nid in scenario.expected_ids + scenario.forbidden_ids:
            if nid.lower() in text:
                retrieved.append(nid)
        if scenario.expected_ids:
            components["correct_retrieval"] = (
                1.0 if any(i in retrieved for i in scenario.expected_ids) else 0.0
            )
        if scenario.forbidden_ids:
            wrong = sum(1 for i in scenario.forbidden_ids if i in retrieved)
            components["no_confusion"] = 1.0 if wrong == 0 else max(0.0, 1.0 - wrong * 0.5)
        if scenario.must_contain:
            hits = sum(1 for k in scenario.must_contain if k.lower() in text)
            components["content_accuracy"] = hits / len(scenario.must_contain)

    # Penalize forbidden content
    if scenario.must_not_contain:
        bad = any(k.lower() in text for k in scenario.must_not_contain if k.startswith("save"))
        if bad and scenario.should_save is False:
            components["no_false_save"] = -1.0
        if any(k.lower() in text for k in scenario.must_not_contain if k == "create_concept"):
            if scenario.should_create_concept is False:
                components["no_false_concept"] = -1.0

    if not components:
        total = 0.0
    else:
        total = sum(components.values()) / len(components)
        total = max(-1.0, min(1.0, total))

    return MemoryRewardBreakdown(
        total=total,
        components=components,
        tool_calls=tool_names,
        retrieved_ids=[],
    )


def eval_memory_batch(scenarios: list[MemoryScenario], golden_completions: list[str]) -> dict:
    """Eval with golden actions — sanity check generator."""
    rewards = [
        compute_memory_reward(g, s).total
        for s, g in zip(scenarios, golden_completions)
    ]
    return {
        "n": len(rewards),
        "mean_reward": round(sum(rewards) / len(rewards), 3) if rewards else 0,
        "min": min(rewards) if rewards else 0,
        "max": max(rewards) if rewards else 0,
    }

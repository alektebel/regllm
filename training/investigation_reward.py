"""Grounding-focused rewards for Phase 1 investigation (no LLM judge)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from training.investigation_scenarios import InvestigationScenario
from training.rl_env import extract_tool_calls

# Weights for Phase 1 single-stage reward
WEIGHTS = {
    "tool_order": 0.30,
    "grounding": 0.40,
    "structure": 0.20,
    "citations": 0.10,
}

BANNED_SOLO_TOOLS = {"search_experience"}
LINEAGE_TOOLS = {
    "trace_dependencies", "inspect_lineage", "trace_causal_chain", "get_sas_formula",
}
DATA_TOOLS = {"query_cycle_data", "trace_causal_chain"}


@dataclass
class InvestigationRewardBreakdown:
    total: float  # [0, 1]
    grpo_reward: float  # [-1, 1] for TRL
    components: dict[str, float]
    tool_calls_found: list[str]
    ungrounded_claims: list[str]


def _analysis_text(completion: str) -> str:
    """Strip tool_call blocks; score prose analysis."""
    text = re.sub(r"<tool_call>.*?</tool_call>", "", completion, flags=re.DOTALL)
    text = re.sub(r'\{"name"\s*:\s*"\w+".*?\}', "", text, flags=re.DOTALL)
    return text.strip()


def _tool_order_score(called: list[str], required: list[str], scenario_type: str) -> float:
    if not required:
        return 1.0 if called else 0.0
    if not called:
        return 0.0

    # Experience-only on bug/missing scenarios is wrong
    if scenario_type in ("missing_value", "wrong_value", "lineage", "cross_reference"):
        if set(called) <= BANNED_SOLO_TOOLS:
            return -0.5
        if not (set(called) & LINEAGE_TOOLS or set(called) & DATA_TOOLS):
            return 0.1

    # Subsequence match: required tools in order
    req_i = 0
    hits = 0
    for tool in called:
        if req_i < len(required) and tool == required[req_i]:
            hits += 1
            req_i += 1
    order_frac = hits / len(required)
    # Partial credit if lineage tools present but wrong order
    if order_frac < 0.5 and (set(called) & LINEAGE_TOOLS):
        order_frac = max(order_frac, 0.4)
    return min(1.0, max(-0.5, order_frac))


def _extract_claim_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for m in re.finditer(r"\b[A-Z][A-Z0-9_]{2,}\b", text):
        tokens.add(m.group(0))
    for m in re.finditer(r"\b\d+(?:\.\d+)?%?\b", text):
        tokens.add(m.group(0).rstrip("%"))
    return tokens


def _grounding_score(analysis: str, scenario: InvestigationScenario) -> tuple[float, list[str]]:
    ctx = scenario.tool_context.lower()
    ctx_raw = scenario.tool_context
    gt = scenario.ground_truth
    ungrounded: list[str] = []

    # Required fields must appear in analysis AND in tool context
    fields = gt.get("must_mention_fields") or []
    field_hits = 0
    for f in fields:
        fu = f.upper()
        if fu in analysis.upper():
            if fu.lower() in ctx or fu in ctx_raw:
                field_hits += 1
            else:
                ungrounded.append(f"field:{fu}")
    field_score = field_hits / len(fields) if fields else 0.5

    # Key phrases from gold answer
    phrases = gt.get("must_contain_phrases") or []
    phrase_hits = sum(1 for p in phrases if p.lower() in analysis.lower())
    phrase_score = phrase_hits / len(phrases) if phrases else 0.5

    # Penalize invented tokens not in context
    claim_tokens = _extract_claim_tokens(analysis)
    grounded_tokens = set(gt.get("grounded_tokens") or [])
    for tok in claim_tokens:
        if tok.isdigit() or "." in tok:
            if tok not in ctx_raw and tok not in grounded_tokens:
                # allow common small integers
                if len(tok) > 1:
                    ungrounded.append(f"num:{tok}")
        elif len(tok) > 4 and tok not in grounded_tokens and tok.lower() not in ctx:
            if tok not in ("ECL", "EAD", "PD", "LGD", "MOC", "RWA", "SAS", "EBA", "IFRS"):
                ungrounded.append(f"token:{tok}")

    inv_penalty = min(0.4, 0.08 * len(ungrounded))
    for bad in scenario.forbidden_phrases:
        if bad.lower() in analysis.lower() and bad.lower() not in ctx:
            ungrounded.append(f"forbidden:{bad}")
            inv_penalty += 0.25

    raw = 0.5 * field_score + 0.5 * phrase_score - inv_penalty
    return max(0.0, min(1.0, raw)), ungrounded


def _structure_score(analysis: str, scenario_type: str) -> float:
    text = analysis.lower()
    markers = ["causa", "paso", "campo", "por qué", "porque", "evidencia", "root", "bug", "missing"]
    hits = sum(1 for m in markers if m in text)
    base = min(1.0, hits / 4)
    if scenario_type in ("missing_value", "wrong_value") and "missing" not in text and "causa" not in text:
        base *= 0.7
    return base


def _citation_score(analysis: str) -> float:
    if "```json" in analysis and "citation" in analysis.lower():
        return 1.0
    if re.search(r"\b(proj_\d+|\.sas:\d+)", analysis, re.I):
        return 0.8
    if re.search(r"\b(art\.|artículo|eba|gl/)\b", analysis, re.I):
        return 0.6
    return 0.2 if len(analysis) > 120 else 0.0


def compute_investigation_reward(
    completion: str,
    scenario: InvestigationScenario,
) -> InvestigationRewardBreakdown:
    """Score completion for Phase 1 investigation. Returns total in [0,1]."""
    calls = extract_tool_calls(completion)
    tool_names = [c.get("name", "") for c in calls]
    analysis = _analysis_text(completion)

    components = {
        "tool_order": _tool_order_score(tool_names, scenario.required_tools, scenario.scenario_type),
        "grounding": 0.0,
        "structure": _structure_score(analysis, scenario.scenario_type),
        "citations": _citation_score(analysis),
    }
    components["grounding"], ungrounded = _grounding_score(analysis, scenario)

    total = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    total = max(0.0, min(1.0, total))
    grpo = 2.0 * total - 1.0

    return InvestigationRewardBreakdown(
        total=round(total, 4),
        grpo_reward=round(grpo, 4),
        components={k: round(v, 4) for k, v in components.items()},
        tool_calls_found=tool_names,
        ungrounded_claims=ungrounded,
    )

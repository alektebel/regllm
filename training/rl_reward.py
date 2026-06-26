"""Verifiable reward function for toy LGD bug-finding RL.

Two modes:
  1. Level-based: keyword matching against hand-crafted solution criteria
     (for the 9 fixed toy_lgd levels).
  2. Functional: AST-mutation-based verification — runs the proposed fix
     through the SAS evaluator and checks if values match the correct
     baseline (for procedurally generated bugs).

No LLM judge needed in either mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOY_LGD = Path(__file__).resolve().parent.parent / "data" / "sas" / "toy_lgd"

# ── Solution criteria per level ──────────────────────────────────────

@dataclass
class BugCriterion:
    """One atomic check for the reward function."""
    name: str
    keywords: list[str]           # ANY of these must appear (case-insensitive)
    weight: float = 1.0
    # If set, ALL must appear (stricter)
    require_all: list[str] = field(default_factory=list)


@dataclass
class LevelSolution:
    level: str
    criteria: list[BugCriterion]
    required_tools: list[str] = field(default_factory=list)


# Hand-coded from solution.md files — these are the ground-truth checks.
SOLUTIONS: dict[str, LevelSolution] = {
    "01_easy_varswap": LevelSolution(
        level="01_easy_varswap",
        criteria=[
            BugCriterion("wrong_variable", ["EAD < 0.30", "EAD instead of LGD", "variable swap", "EAD en lugar de LGD"], weight=2.0),
            BugCriterion("correct_line", ["HIPOTECA", "floor"], weight=1.0),
            BugCriterion("correct_fix", ["LGD_ESTIMADA < 0.30"], weight=1.0),
        ],
        required_tools=["trace_dependencies", "inspect_lineage"],
    ),
    "02_easy_missing": LevelSolution(
        level="02_easy_missing",
        criteria=[
            BugCriterion("missing_coalesce", ["COALESCE", "missing", "default", "valor faltante"], weight=2.0),
            BugCriterion("fusion_context", ["fusion", "SW_FUSION", "absorbida"], weight=1.0),
            BugCriterion("propagation", ["MoC", "ECL", "propaga"], weight=1.0),
        ],
        required_tools=["trace_dependencies", "inspect_lineage"],
    ),
    "03_easy_boundary": LevelSolution(
        level="03_easy_boundary",
        criteria=[
            BugCriterion("off_by_one", ["> 30", ">= 30", "off-by-one", "boundary", "frontera"], weight=2.0),
            BugCriterion("dpds_context", ["DPDS", "stage", "IFRS"], weight=1.0),
            BugCriterion("correct_fix", [">= 30", "DPDS >= 30"], weight=1.0),
        ],
        required_tools=["trace_dependencies", "inspect_lineage"],
    ),
    "04_medium_filter": LevelSolution(
        level="04_medium_filter",
        criteria=[
            BugCriterion("wrong_threshold", ["> 12", ">= 9", "12 instead", "umbral incorrecto"], weight=2.0),
            BugCriterion("provision_period", ["PROVISION_PERIOD_MONTHS", "periodo de provision"], weight=1.0),
            BugCriterion("regulatory_ref", ["Circular", "art", "9 meses"], weight=1.0),
        ],
        required_tools=["trace_dependencies", "inspect_lineage"],
    ),
    "05_medium_fusion_dup": LevelSolution(
        level="05_medium_fusion_dup",
        criteria=[
            BugCriterion("non_unique_join", ["non-unique", "duplica", "ID_FUSION_FINAL", "multiple rows"], weight=2.0),
            BugCriterion("wrong_agg", ["SUM", "MAX", "agregaci"], weight=1.5),
            BugCriterion("or_ead_inflated", ["OR_EAD", "inflat", "200000"], weight=1.0),
        ],
        required_tools=["trace_dependencies", "inspect_lineage"],
    ),
    "06_hard_agg_level": LevelSolution(
        level="06_hard_agg_level",
        criteria=[
            BugCriterion("wrong_group_by", ["COLATERAL_TIPO", "SEGMENTO", "GROUP BY"], weight=2.0,
                         require_all=["COLATERAL_TIPO", "SEGMENTO"]),
            BugCriterion("moc_deviation", ["MoC", "desviaci", "LGD_MEDIA"], weight=1.0),
            BugCriterion("merge_key", ["MERGE", "BY"], weight=0.5),
        ],
        required_tools=["trace_dependencies", "inspect_lineage"],
    ),
    "07_harder_type_coerce": LevelSolution(
        level="07_harder_type_coerce",
        criteria=[
            BugCriterion("case_sensitive", ["case", "mayusc", "minusc", "UPCASE", "hipoteca", "Hipoteca"], weight=2.0),
            BugCriterion("hipoteca_variants", ["hipoteca", "Hipoteca", "HIPOTECA"], weight=1.0,
                         require_all=["hipoteca", "HIPOTECA"]),
            BugCriterion("correct_fix", ["UPCASE", "upcase", "LOWCASE"], weight=1.0),
        ],
        required_tools=["trace_dependencies", "inspect_lineage"],
    ),
    "08_hardest_compound": LevelSolution(
        level="08_hardest_compound",
        criteria=[
            BugCriterion("bug1_join_dup", ["non-unique", "duplica", "ID_FUSION_FINAL", "JOIN"], weight=2.0),
            BugCriterion("bug2_sum_lgd", ["SUM(LGD_ESTIMADA)", "SUM", "MAX", "LGD_ESTIMADA"], weight=2.0,
                         require_all=["SUM", "LGD_ESTIMADA"]),
            BugCriterion("interaction", ["interact", "compound", "compuest", "ambos"], weight=1.5),
            BugCriterion("impossible_lgd", ["1.10", "> 1", "imposible", "fuera de rango"], weight=1.0),
        ],
        required_tools=["trace_dependencies", "inspect_lineage"],
    ),
}


# ── Reward computation ───────────────────────────────────────────────

def _check_criterion(text: str, crit: BugCriterion) -> float:
    """Return 0.0–1.0 for how well the text satisfies a criterion."""
    text_lower = text.lower()

    if crit.require_all:
        matched = sum(1 for kw in crit.require_all if kw.lower() in text_lower)
        return matched / len(crit.require_all)

    for kw in crit.keywords:
        if kw.lower() in text_lower:
            return 1.0
    return 0.0


def _check_tools(text: str, required: list[str]) -> float:
    """Check if the completion contains the right tool calls."""
    if not required:
        return 1.0
    found = 0
    for tool in required:
        if tool in text:
            found += 1
    return found / len(required)


def _check_hallucination(text: str) -> float:
    """Penalty for common hallucination patterns.

    Returns 0.0 (no hallucination) to 1.0 (severe hallucination).
    """
    penalties = 0.0
    markers = 0

    # Fabricated values not grounded in tool results
    if re.search(r"(?:he encontrado|puedo ver|según mi análisis).*?(?:\d+\.\d{4,})", text, re.I):
        penalties += 0.3
        markers += 1

    # Inventing regulation articles that don't exist in the context
    fake_articles = re.findall(r"art(?:ículo|\.)\s*\d{3,}", text, re.I)
    if len(fake_articles) > 2:
        penalties += 0.2 * min(len(fake_articles) - 2, 3)
        markers += 1

    # Claiming to have executed queries (the model can't run SAS)
    if re.search(r"(?:ejecut[ée]|corr[ií]|lance)\s+(?:la|el|una)\s+(?:consulta|query|proc)", text, re.I):
        penalties += 0.2
        markers += 1

    return min(penalties, 1.0)


def score_completion(text: str, level: str) -> dict:
    """Score a model completion against a level's solution.

    Returns dict with:
      - reward: float in [-1, 1] (normalized)
      - criteria_scores: per-criterion breakdown
      - tool_score: tool selection quality
      - hallucination_penalty: hallucination penalty
    """
    sol = SOLUTIONS.get(level)
    if not sol:
        return {"reward": 0.0, "error": f"Unknown level: {level}"}

    # Criterion scores
    criteria_scores = {}
    total_weight = 0.0
    weighted_score = 0.0
    for crit in sol.criteria:
        score = _check_criterion(text, crit)
        criteria_scores[crit.name] = score
        weighted_score += score * crit.weight
        total_weight += crit.weight

    criteria_reward = weighted_score / total_weight if total_weight > 0 else 0.0

    # Tool selection
    tool_score = _check_tools(text, sol.required_tools)

    # Hallucination penalty
    halluc = _check_hallucination(text)

    # Composite reward: 60% diagnosis, 25% tools, 15% no-hallucination
    reward = 0.60 * criteria_reward + 0.25 * tool_score + 0.15 * (1.0 - halluc)

    # Shift to [-1, 1] range for GRPO
    reward = 2.0 * reward - 1.0

    return {
        "reward": round(reward, 4),
        "criteria_scores": criteria_scores,
        "criteria_reward": round(criteria_reward, 4),
        "tool_score": round(tool_score, 4),
        "hallucination_penalty": round(halluc, 4),
    }


def load_level_prompt(level: str) -> dict:
    """Load a level's SAS code and question for the agent."""
    level_dir = TOY_LGD / "levels" / level
    code = (level_dir / "lgd.sas").read_text(encoding="utf-8")
    level_md = (level_dir / "level.md").read_text(encoding="utf-8")

    # Extract the expected vs actual from level.md
    question = (
        f"Analiza el siguiente código SAS LGD. Hay un bug que produce resultados incorrectos. "
        f"Usa las herramientas trace_dependencies e inspect_lineage para investigar. "
        f"Identifica la línea con el error, explica por qué falla, y propón la corrección.\n\n"
        f"```sas\n{code}\n```\n\n"
        f"Contexto:\n{level_md}"
    )

    return {
        "level": level,
        "code": code,
        "question": question,
    }


# ── Functional verification (for procedural bugs) ─────────────────────

def score_functional(
    completion: str,
    mutated_bug: Any,  # training.bug_generator.MutatedBug
) -> dict:
    """Score a completion using functional verification against a MutatedBug.

    This is the VibeThinker-style reward: run the agent's proposed fix
    through the SAS evaluator and compare to the correct baseline.
    Returns reward in [-1, 1].
    """
    from training.rl_env import compute_reward, STAGES
    from training.curriculum import StageConfig

    # Determine which stage to use based on bug depth
    if mutated_bug.depth == 0:
        stage = STAGES[1]  # single_depth
    else:
        stage = STAGES[2]  # multi_depth

    breakdown = compute_reward(completion, mutated_bug, stage)

    return {
        "reward": breakdown.total,
        "components": breakdown.components,
        "tool_calls": breakdown.tool_calls_found,
        "correct_step": breakdown.correct_step_found,
        "correct_var": breakdown.correct_var_found,
        "fix_restores": breakdown.fix_restores_values,
    }

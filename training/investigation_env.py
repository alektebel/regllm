"""GRPO-ready investigation environment for Phase 1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.investigation_reward import InvestigationRewardBreakdown, compute_investigation_reward
from training.investigation_scenarios import InvestigationScenario, INVESTIGATION_SYSTEM, load_scenarios_from_eval

MANIFEST = PROJECT_ROOT / "training/data/investigation_eval_manifest.jsonl"


def load_investigation_manifest(path: Path | str | None = None) -> list[InvestigationScenario]:
    p = Path(path or MANIFEST)
    if not p.exists():
        raise FileNotFoundError(f"Run build_investigation_eval_manifest.py first: {p}")
    out: list[InvestigationScenario] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(InvestigationScenario(
            scenario_id=row["id"],
            scenario_type=row["scenario_type"],
            question=row["question"],
            required_tools=row["required_tools"],
            tool_steps=[(s["tool"], s["args"]) for s in row["tool_steps"]],
            tool_context=row["tool_context"],
            ground_truth=row["ground_truth"],
            messages=row["messages"],
            source_category=row.get("source_category", ""),
            forbidden_phrases=row.get("forbidden_phrases") or [],
        ))
    return out


def build_prompt(scenario: InvestigationScenario) -> list[dict]:
    return scenario.messages


def oracle_completion(scenario: InvestigationScenario) -> str:
    """Gold completion: tool calls + grounded answer."""
    parts = []
    for tool, args in scenario.tool_steps:
        parts.append(
            f'<tool_call>{{"name": "{tool}", "arguments": {json.dumps(args, ensure_ascii=False)}}}</tool_call>'
        )
    answer = scenario.ground_truth.get("root_answer_snippet", "")
    fields = scenario.ground_truth.get("must_mention_fields") or []
    if fields:
        answer += f"\n\nCampos clave: {', '.join(fields[:4])}."
    parts.append(answer)
    return "\n".join(parts)


def generate_investigation_batch(
    n: int,
    *,
    seed: int = 42,
    manifest_path: Path | str | None = None,
) -> list[tuple[InvestigationScenario, list[dict]]]:
    import random
    scenarios = load_investigation_manifest(manifest_path) if Path(manifest_path or MANIFEST).exists() else load_scenarios_from_eval()
    rng = random.Random(seed)
    if len(scenarios) >= n:
        picked = rng.sample(scenarios, n)
    else:
        picked = [rng.choice(scenarios) for _ in range(n)]
    return [(sc, build_prompt(sc)) for sc in picked]


__all__ = [
    "InvestigationScenario",
    "InvestigationRewardBreakdown",
    "build_prompt",
    "compute_investigation_reward",
    "generate_investigation_batch",
    "load_investigation_manifest",
    "oracle_completion",
]

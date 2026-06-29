"""Investigation scenario types for Phase 1 — grounded DB analysis with RAG."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SESSION = "debug_lgd"

CATEGORY_TO_TYPE: dict[str, str] = {
    "bug_detection": "missing_value",
    "data_quality": "wrong_value",
    "lineage": "lineage",
    "regulation": "regulatory",
    "field_definition": "schema_docs",
    "cross_reference": "cross_reference",
}

INVESTIGATION_SYSTEM = """\
Eres un investigador de validación regulatoria bancaria (COREP/FINREP IRB).
DEBES llamar herramientas ANTES de concluir. Basa números y fórmulas SOLO en tool results.

## Herramientas

- **trace_dependencies** / **inspect_lineage**: lineage SAS (obligatorio en bugs/missing).
- **trace_causal_chain** / **query_cycle_data**: valores reales en la BD.
- **search_docs** / **get_field_definition**: documentación.
- **search_regulation**: normativa EBA/BdE.
- **search_experience**: experiencia previa — nunca como única herramienta en bugs.

## Respuesta final

Estructura: qué ocurre, dónde (paso/campo), por qué (causa raíz), evidencia citada.
Responde en español.
"""


@dataclass
class InvestigationScenario:
    scenario_id: str
    scenario_type: str
    question: str
    required_tools: list[str]
    tool_steps: list[tuple[str, dict]]
    tool_context: str
    ground_truth: dict[str, Any]
    messages: list[dict]
    source_category: str = ""
    forbidden_phrases: list[str] = field(default_factory=lambda: ["OR_EADIL", "según mi conocimiento"])

    def prompt_messages(self) -> list[dict]:
        """Chat messages for the user turn (system + user)."""
        return self.messages[:2]


def _flatten_tool_context(steps: list[tuple[str, dict]], results: list[Any]) -> str:
    chunks = []
    for (tool, args), result in zip(steps, results):
        chunks.append(json.dumps({"tool": tool, "args": args, "result": result}, ensure_ascii=False, default=str))
    return "\n".join(chunks)


def _parse_steps_from_eval(ex: dict) -> list[tuple[str, dict]]:
    from training.build_investigation_sft import _parse_steps
    return _parse_steps(ex)


def _required_tools(ex: dict, steps: list[tuple[str, dict]]) -> list[str]:
    from training.build_tool_sft import TOOL_ALIASES
    if steps:
        return [t for t, _ in steps]
    out = []
    for t in ex.get("expected_tools", []):
        t = TOOL_ALIASES.get(t, t)
        if t not in out:
            out.append(t)
    return out


def scenario_from_eval(ex: dict, *, dispatch_tool) -> InvestigationScenario | None:
    """Build one investigation scenario from eval_dataset entry."""
    cat = ex.get("category", "")
    if cat not in CATEGORY_TO_TYPE:
        return None
    steps = _parse_steps_from_eval(ex)
    if len(steps) < 1:
        return None
    required = _required_tools(ex, steps)
    results = [dispatch_tool(tool, args) for tool, args in steps]
    tool_context = _flatten_tool_context(steps, results)

    answer = ex.get("solution", {}).get("final_answer") or ex.get("gold_answer_summary", "")
    gt: dict[str, Any] = {
        "must_mention_fields": ex.get("expected_fields_mentioned") or [],
        "must_contain_phrases": ex.get("expected_answer_contains") or [],
        "root_answer_snippet": answer[:400],
        "key_sources": ex.get("solution", {}).get("key_sources") or [],
    }
    # Values/numerics from tool results for grounding checks
    nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", tool_context))
    gt["grounded_tokens"] = sorted(nums | {f.upper() for f in gt["must_mention_fields"]})

    messages = [
        {"role": "system", "content": INVESTIGATION_SYSTEM},
        {"role": "user", "content": ex["question"]},
    ]

    return InvestigationScenario(
        scenario_id=ex["id"],
        scenario_type=CATEGORY_TO_TYPE[cat],
        question=ex["question"],
        required_tools=required,
        tool_steps=steps,
        tool_context=tool_context,
        ground_truth=gt,
        messages=messages,
        source_category=cat,
    )


def load_scenarios_from_eval(
    path: Path | str | None = None,
    *,
    categories: set[str] | None = None,
) -> list[InvestigationScenario]:
    from src.agent.tools import dispatch_tool

    p = Path(path or PROJECT_ROOT / "data/eval_dataset.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    cats = categories or set(CATEGORY_TO_TYPE.keys())
    out: list[InvestigationScenario] = []
    for ex in data:
        if ex.get("category") not in cats:
            continue
        sc = scenario_from_eval(ex, dispatch_tool=dispatch_tool)
        if sc:
            out.append(sc)
    return out

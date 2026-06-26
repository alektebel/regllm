"""Procedural tool-selection tasks for curriculum stage 0."""

from __future__ import annotations

import random
from dataclasses import dataclass

from training.bug_generator import MutationMeta, MutatedBug

LINEAGE_TOOLS = {"trace_dependencies", "inspect_lineage", "trace_causal_chain", "get_sas_formula"}
REG_TOOLS = {"search_regulation", "query_regulation", "find_governing_rules"}
DOC_TOOLS = {"search_docs", "get_field_definition", "search_changelog"}
SCHEMA_TOOLS = {"get_schema_context", "enrich_fields", "query_table"}


@dataclass(frozen=True)
class ToolTaskTemplate:
    task_kind: str
    question: str
    expected_tools: frozenset[str]
    field: str = ""
    context: str = ""


_TEMPLATES: list[ToolTaskTemplate] = [
    ToolTaskTemplate(
        "lineage",
        "¿De qué campos depende {field}? Necesito el grafo de dependencias upstream.",
        frozenset(LINEAGE_TOOLS),
        "ECL",
    ),
    ToolTaskTemplate(
        "lineage",
        "Traza el linaje de {field} en el pipeline LGD hasta sus inputs.",
        frozenset(LINEAGE_TOOLS),
        "LGD_ESTIMADA",
    ),
    ToolTaskTemplate(
        "regulation",
        "¿Qué normativa regula el suelo de LGD para hipotecas (CRR Art. 154)?",
        frozenset(REG_TOOLS),
    ),
    ToolTaskTemplate(
        "regulation",
        "Busca en la regulación el artículo sobre dotaciones mínimas de LGD.",
        frozenset(REG_TOOLS),
    ),
    ToolTaskTemplate(
        "docs",
        "¿Qué documentación interna describe el cálculo de {field}?",
        frozenset(DOC_TOOLS),
        "MoC",
    ),
    ToolTaskTemplate(
        "schema",
        "¿En qué tabla está la columna {field} y qué tipo tiene?",
        frozenset(SCHEMA_TOOLS),
        "PD_ESTIMADA",
    ),
]


def make_tool_selection_bug(template: ToolTaskTemplate, *, task_id: str = "") -> MutatedBug:
    """Synthetic MutatedBug carrying tool-selection ground truth."""
    field = template.field or "ECL"
    question = template.question.format(field=field)
    return MutatedBug(
        original_code="",
        mutated_code="",
        mutation=MutationMeta(
            mutation_type="tool_selection",
            step_name="",
            node_index=0,
            original_expr="",
            mutated_expr="",
            target_var=field,
            description=question,
        ),
        correct_values={},
        buggy_values={},
        changed_fields=[field] if field else [],
        depth=0,
        expected_tools=sorted(template.expected_tools),
        task_kind=template.task_kind,
        source="tool_task",
        level_id=task_id,
    )


def generate_tool_selection_batch(n: int, seed: int = 42) -> list[MutatedBug]:
    rng = random.Random(seed)
    bugs: list[MutatedBug] = []
    for i in range(n):
        tmpl = rng.choice(_TEMPLATES)
        bugs.append(make_tool_selection_bug(tmpl, task_id=f"tool_{tmpl.task_kind}_{i:03d}"))
    return bugs

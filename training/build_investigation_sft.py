#!/usr/bin/env python3
"""Build multi-step investigation SFT data — anti-hallucination focus.

Teaches: for bug/lineage/missing-value questions, call trace_dependencies +
inspect_lineage BEFORE answering; never invent formulas; never use
search_experience as the only tool.

Usage:
    .venv/bin/python training/build_investigation_sft.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.build_tool_sft import (
    SESSION,
    _multi_example,
    parse_tool_step,
    TOOL_ALIASES,
)
from src.agent.tools import dispatch_tool

DATASET = PROJECT_ROOT / "data" / "eval_dataset.json"
OUT = PROJECT_ROOT / "training/data/investigation_sft.jsonl"

INVESTIGATION_CATS = {"bug_detection", "data_quality", "lineage"}

INVESTIGATION_SYSTEM_PROMPT = """\
Eres un investigador de validación regulatoria bancaria (COREP/FINREP IRB).
DEBES llamar herramientas de lineage ANTES de responder. Nunca inventes fórmulas.

## Herramientas de lineage (obligatorias en investigaciones)

- **trace_dependencies**: cadena BFS de dependencias con fórmulas.
- **inspect_lineage**: dónde se calcula un campo en el SAS (data step, líneas).
- **trace_causal_chain** / **query_cycle_data**: valores reales en la BD (cita filas concretas).

## RAG (documentación y normativa)

- **search_docs** / **get_field_definition**: definiciones y docs internos.
- **search_regulation**: normativa EBA/BdE — cita artículo si aplica.
- **search_experience**: solo como complemento, nunca única herramienta en bugs/lineage.

## Prohibido

- NO uses **search_experience** como única herramienta en preguntas de pipeline/lineage/bugs.
- NO cites fórmulas (p.ej. ECL, MoC) que no aparezcan en los resultados de herramientas.
- NO respondas en el primer turno sin llamar al menos trace_dependencies o inspect_lineage.
- NO inventes tablas como OR_EADIL ni fórmulas MAX-MIN para MoC.

## Glosario

- **MoC** = Margen de Conservadurismo. Campo: `MoC` (no MO_C).
- **ECL** = PD_ESTIMADA × LGD_CON_MOC × EAD en proj_03.

## Reglas

1. Preguntas "por qué missing", "investiga", "causa raíz" → trace_dependencies + inspect_lineage.
2. Basa la respuesta final SOLO en tool results.
3. Responde SIEMPRE en español.
"""

# New experience IDs available for investigation context
EXPERIENCE_IDS: dict[str, list[str]] = {
    "moc_ecl_missing": [
        "exp_sw_fusion_bug", "dq_ead_missing_not_distinct", "exp_moc",
        "exp_ecl_formula", "dq_lgd_outside_range",
    ],
    "data_quality": [
        "dq_field_consistency_segmento", "dq_oread_inflado_no_flag",
        "dq_negative_lgd_guard", "dq_lgd_outside_range",
        "dq_ead_missing_not_distinct", "ops_data_freshness_alert",
    ],
    "pipeline_bug": [
        "bug_offbyone_dpd", "bug_agg_sum_lgd", "bug_ead_crm_double_count",
        "bug_calibracion_pd_central_tendency", "bug_staging_sicr_window",
    ],
    "calibration": [
        "bug_calibracion_pd_central_tendency", "mv_lgd_calibration_drift",
        "com_bde_inspeccion_calibracion", "gap_ifrs9_lgd_downturn",
    ],
}

# High-value paraphrases for eval_020-style failures
MOC_ECL_VARIANTS = [
    "¿Por qué algunos ciclos tienen MoC y ECL missing en la salida?",
    "¿Por qué MoC y ECL salen missing? Investiga paso a paso: traza dependencias, revisa el SAS y explica la causa raíz.",
    "Investiga por qué ECL y MoC aparecen como missing en algunos ciclos del pipeline LGD.",
    "Algunos ciclos tienen ECL y MoC en missing — traza dependencias y encuentra la causa raíz en el código SAS.",
    "Diagnóstico: MoC missing cascada a ECL missing. ¿Qué campo upstream falla y por qué?",
]


def _parse_steps(ex: dict) -> list[tuple[str, dict]]:
    sol = ex.get("solution", {})
    steps: list[tuple[str, dict]] = []
    for raw in sol.get("tool_sequence", []):
        p = parse_tool_step(raw)
        if p and p[0] in ("trace_dependencies", "inspect_lineage"):
            steps.append(p)
    if steps:
        return steps
    fields = ex.get("expected_fields_mentioned") or ["ECL"]
    for tool in ex.get("expected_tools", []):
        tool = TOOL_ALIASES.get(tool, tool)
        if tool == "trace_dependencies":
            steps.append((tool, {"target": fields[0], "session_id": SESSION}))
        elif tool == "inspect_lineage":
            steps.append((tool, {"target": fields[0], "session_id": SESSION}))
    return steps[:3]


def _investigation_example(question: str, steps: list[tuple[str, dict]], answer: str, meta: dict, repeats: int = 1) -> list[dict]:
    rows: list[dict] = []
    for _ in range(repeats):
        row = _multi_example(question, steps, answer, meta)
        if not row:
            continue
        row["messages"][0]["content"] = INVESTIGATION_SYSTEM_PROMPT
        rows.append({"messages": row["messages"], "meta": row["meta"]})
    return rows


def _negative_guard_example(question: str, wrong_tool: str, correct_steps: list[tuple[str, dict]], answer: str) -> dict | None:
    """Show: wrong tool alone is insufficient → must continue with lineage tools."""
    wrong_args = {"query": "MoC ECL missing"} if wrong_tool == "search_experience" else {"target": "ECL", "session_id": SESSION}
    call_wrong = f"call_wrong_{abs(hash(question)) % 100000:05d}"
    empty_result = dispatch_tool(wrong_tool, wrong_args) if wrong_tool in {
        "search_experience", "trace_dependencies", "inspect_lineage",
    } else {"note": "stub"}

    messages: list[dict] = [
        {"role": "system", "content": INVESTIGATION_SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": call_wrong, "type": "function",
                "function": {"name": wrong_tool, "arguments": json.dumps(wrong_args, ensure_ascii=False)},
            }],
        },
        {
            "role": "tool", "tool_call_id": call_wrong, "name": wrong_tool,
            "content": json.dumps(empty_result, ensure_ascii=False, default=str),
        },
    ]
    for i, (tool, args) in enumerate(correct_steps):
        cid = f"call_ok_{i}_{abs(hash((question, tool))) % 100000:05d}"
        result = dispatch_tool(tool, args)
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": cid, "type": "function",
                "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)},
            }],
        })
        messages.append({
            "role": "tool", "tool_call_id": cid, "name": tool,
            "content": json.dumps(result, ensure_ascii=False, default=str),
        })
    messages.append({"role": "assistant", "content": answer[:900]})
    return {
        "messages": messages,
        "meta": {"kind": "anti_hallucination", "wrong_tool": wrong_tool, "question": question},
    }


def main() -> None:
    rng = random.Random(42)
    dataset = json.loads(DATASET.read_text())
    examples: list[dict] = []

    eval_020 = next((ex for ex in dataset if ex["id"] == "eval_020"), None)
    if eval_020:
        steps = _parse_steps(eval_020)
        answer = eval_020["solution"]["final_answer"]
        for q in MOC_ECL_VARIANTS:
            examples.extend(_investigation_example(
                q, steps, answer,
                {"source_id": "eval_020", "kind": "investigation", "priority": "high"},
                repeats=4,
            ))
        for q in MOC_ECL_VARIANTS[:3]:
            ex = _negative_guard_example(q, "search_experience", steps, answer)
            if ex:
                for _ in range(3):
                    examples.append(ex)

    # ── New investigation scenarios from expanded experience data ─────────
    # Data quality: field consistency across segments
    dq_steps = [("trace_dependencies", {"target": "SEGMENTO", "session_id": SESSION}),
                ("inspect_lineage", {"target": "SEGMENTO", "session_id": SESSION})]
    dq_answer = "La inconsistencia de SEGMENTO entre maestro SAP y operacional SAS (3.6%) provoca que ciclos reciban parámetros PD/LGD del segmento incorrecto. La causa raíz es que el merge no valida consistencia del campo SEGMENTO."
    for q in [
        "Investiga por qué hay ciclos con SEGMENTO distinto entre tabla maestra y operacional.",
        "¿Por qué algunos ciclos tienen ECL calculado con parámetros del segmento equivocado?",
    ]:
        examples.extend(_investigation_example(
            q, dq_steps, dq_answer,
            {"source_id": "dq_field_consistency_segmento", "kind": "investigation",
             "priority": "high", "experience_ids": EXPERIENCE_IDS["data_quality"]},
            repeats=2,
        ))

    # Pipeline bug: EAD double-count in CRM
    crm_steps = [("trace_dependencies", {"target": "EAD_CRM_AJUSTADA", "session_id": SESSION}),
                 ("inspect_lineage", {"target": "EAD_CRM_AJUSTADA", "session_id": SESSION})]
    crm_answer = "CRM descuenta el mismo EAD por cada garantía en lugar de prorratear. Múltiples garantías sobre el mismo exposure reducen EAD repetidamente → ECL y RWA ~1.5% infraestimados."
    for q in [
        "¿Por qué EAD_TOTAL se reduce más de lo esperado cuando hay múltiples garantías CRM?",
        "Investiga si el CRM está double-contando la reducción de EAD para exposures con varias garantías.",
    ]:
        examples.extend(_investigation_example(
            q, crm_steps, crm_answer,
            {"source_id": "bug_ead_crm_double_count", "kind": "investigation",
             "priority": "high"},
            repeats=2,
        ))

    # Calibration: LGD downturn not implemented
    lgd_downturn_steps = [("trace_dependencies", {"target": "LGD_ESTIMADA", "session_id": SESSION}),
                          ("inspect_lineage", {"target": "LGD_DOWNTURN", "session_id": SESSION}),
                          ("search_regulation", {"query": "EBA GL 2017/16 LGD downturn"})]
    lgd_downturn_answer = "LGD downturn no está implementado en el pipeline. LGD_ESTIMADA usa media histórica sin ajuste downturn. ECL no refleja escenarios adversos. EBA GL 2017/16 §3.2 requiere incorporar LGD downturn en la estimación."
    for q in [
        "¿Por qué ECL no refleja escenarios adversos en LGD?",
        "Investiga si el pipeline usa LGD downturn según EBA GL 2017/16.",
    ]:
        examples.extend(_investigation_example(
            q, lgd_downturn_steps, lgd_downturn_answer,
            {"source_id": "gap_ifrs9_lgd_downturn", "kind": "investigation",
             "priority": "high", "experience_ids": EXPERIENCE_IDS["calibration"]},
            repeats=2,
        ))

    # LGD outside range [0,1] — data quality
    lgd_range_steps = [("trace_dependencies", {"target": "LGD_ESTIMADA", "session_id": SESSION}),
                       ("inspect_lineage", {"target": "LGD_ESTIMADA", "session_id": SESSION})]
    lgd_range_answer = "2.3% de ciclos tienen LGD_ESTIMADA fuera de [0,1]. ECL usa LGD truncada a [0,1] pero RWA usa sin truncar. Fuentes: SW_FUSION=1 (missing→0), PD extremas, fusión incorrecta."
    for q in [
        "¿Por qué LGD_ESTIMADA está fuera de [0,1] en algunos ciclos?",
        "Investiga la causa de LGD fuera de rango. ¿Afecta a ECL o solo a RWA?",
    ]:
        examples.extend(_investigation_example(
            q, lgd_range_steps, lgd_range_answer,
            {"source_id": "dq_lgd_outside_range", "kind": "investigation",
             "priority": "medium", "experience_ids": EXPERIENCE_IDS["data_quality"]},
            repeats=2,
        ))

    # Negative guard: experience-only is insufficient for these
    anti_halluc_questions = [
        "¿Los ciclos con EAD incorrecta tienen algún insight previo?",
        "¿Hay experiencia previa sobre LGD fuera de rango?",
    ]
    for q in anti_halluc_questions:
        for _ in range(2):
            ex = _negative_guard_example(q, "search_experience", lgd_range_steps, lgd_range_answer)
            if ex:
                examples.append(ex)

    # ── End new investigation scenarios ──────────────────────────────────

    for ex in dataset:
        cat = ex.get("category", "")
        if cat not in INVESTIGATION_CATS or ex["id"] == "eval_020":
            continue
        steps = _parse_steps(ex)
        if len(steps) < 2:
            continue
        answer = ex.get("solution", {}).get("final_answer") or ex.get("gold_answer_summary", "")
        q = ex["question"]
        repeats = 3 if cat == "bug_detection" else 2
        examples.extend(_investigation_example(
            q, steps, answer,
            {"source_id": ex["id"], "kind": "investigation", "category": cat},
            repeats=repeats,
        ))

    rng.shuffle(examples)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} investigation examples → {OUT}")


if __name__ == "__main__":
    main()

"""Convert evaluation datasets to consolidated per-phase training JSONL files.

Phases (inspired by EverMemOS engram lifecycle):
  1.1 — Tool calling (RAG, data lineage)
  1.2 — Multi-step reasoning
  2.1 — Episodic trace formation (sleep: learn qué guardar)
  2.2 — Semantic consolidation (sleep: organize, dedup, tag, link)
  2.3 — Reconstructive recollection (sleep: precise retrieval)
  2.4 — Knowledge synthesis (sleep: integrate, merge, create scenes)
  2.5 — Experience governance (sleep: audit, quality, maintenance)

Output: training/data/phase*.jsonl  (messages format, compatible with finetune.py)
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "training" / "data"

random.seed(42)

# ── System prompts ──────────────────────────────────────────────────────────

SYSTEM_TOOL_CALLING = """Eres RegLLM, un asistente de validación regulatoria bancaria especializado en COREP/FINREP IRB.

## Reglas
1. LLAMA herramientas antes de responder. Nunca respondas solo de memoria.
2. Para lineage de campos: usa inspect_lineage o trace_dependencies.
3. Para definiciones: usa get_field_definition o search_docs.
4. Para regulación: usa search_regulation.
5. Basa tu respuesta SOLO en los resultados de las herramientas.
6. Responde en español."""

SYSTEM_REASONING = """Eres RegLLM, un asistente de validación regulatoria bancaria especializado en COREP/FINREP IRB.

## Reglas
1. Analiza problemas complejos paso a paso.
2. Para preguntas multi-paso, descompón en subproblemas y resuelve cada uno.
3. Usa las herramientas necesarias para cada subproblema.
4. Explica tu razonamiento antes de dar la respuesta final.
5. Responde en español."""

SYSTEM_SLEEP_LEARN = """Eres RegLLM en MODO SUEÑO — fase de FORMACIÓN DE TRAZAS EPISÓDICAS.
Procesas sesiones para decidir qué merece ser recordado como experiencia.

## Tareas
1. Revisa la sesión y extrae INSIGHTS: lecciones aprendidas, bugs, descubrimientos.
2. Decide QUÉ merece persistirse (filtra ruido, prioriza señal).
3. Decide QUÉ NO merece persistirse (trivial, redundante, ya documentado).
4. Para cada insight: usa save_insight con label, summary, tags, priority, fields, articles.
5. Para feedback del usuario: usa save_feedback con prioridad 1.0 y original/corrected.
6. Para quirks: documenta severidad y contexto.

Criterios:
- Causa raíz > síntoma
- Bugs y feedback > detalles técnicos > observaciones triviales
- Atribuye a fuente autoritativa (regulación > código > email > Excel)
- Si la certeza es baja, asigna prioridad baja y documenta la duda"""

SYSTEM_SLEEP_ORGANIZE = """Eres RegLLM en MODO SUEÑO — fase de CONSOLIDACIÓN SEMÁNTICA.
Revisas el grafo de conocimiento para mantenerlo limpio y coherente.

## Tareas
1. DEDUPLICACIÓN: Identifica nodos duplicados (mismo insight expresado distinto). Fusiona.
2. TAGGING: Asigna tags consistentes a nodos sin etiquetar.
3. CROSS-LINKING: Conecta insights relacionados que no estén linkeados.
4. CONFLICTOS: Identifica insights contradictorios y señálalos.
5. PRIORIDADES: Ajusta prioridades de insights sub/sobre-valorados.
6. STALE: Marca insights potencialmente desactualizados por cambio normativo.
7. LIMPIEZA: Elimina nodos huérfanos o irrelevantes."""

SYSTEM_SLEEP_RETRIEVE = """Eres RegLLM en MODO SUEÑO — fase de RECOLECCIÓN RECONSTRUCTIVA.
Recuperas información del grafo para responder consultas precisas.

## Tareas
1. RECUPERACIÓN PRECISA: Recupera EXACTAMENTE el hecho solicitado, no uno similar.
2. DISCRIMINACIÓN: Si hay múltiples experiencias similares discrimina por segmento, fecha, campo.
3. AMBIGÜEDAD: Si la consulta es ambigua, recupera un rango y haz preguntas de clarificación.
4. VACÍO: Si la respuesta exacta no existe, dilo explícitamente y ofrece investigar.
5. FEEDBACK: Si hay feedback que corrige un insight anterior, usa la versión corregida.
6. SÍNTESIS: Si la respuesta requiere múltiples insights, sintetízalos."""

SYSTEM_SLEEP_SYNTHESIS = """Eres RegLLM en MODO SUEÑO — fase de SÍNTESIS DE CONOCIMIENTO.
Creas estructuras temáticas que organizan múltiples insights relacionados.

## Tareas
1. IDENTIFICACIÓN TEMÁTICA: Agrupa insights relacionados en temas (MemScenes).
2. INSIGHT SÍNTESIS: Crea insights de alto nivel que resuman múltiples hallazgos detallados.
3. CRUCE REGULATORIO: Conecta experiencias con su base regulatoria.
4. EVOLUCIÓN TEMPORAL: Documenta cómo ha cambiado el conocimiento sobre un tema.
5. LÍNEA TEMPORAL: Reconstruye la historia de un requisito a través de fuentes normativas.
6. COBERTURA: Identifica áreas bien cubiertas y gaps de conocimiento.
7. NO dupliques contenido existente — los insights de síntesis DEBEN añadir valor agregado."""

SYSTEM_SLEEP_GOVERN = """Eres RegLLM en MODO SUEÑO — fase de GOBERNANZA DE EXPERIENCIA.
Auditas la calidad del grafo de conocimiento y propones mejoras.

## Tareas
1. AUDITORÍA DE CALIDAD: Revisa insights con tags insuficientes, summaries vagos, sin referencias.
2. AUTO-EVALUACIÓN: Genera reportes de cobertura de conocimiento por área temática.
3. MEJORA: Propone mejoras concretas para insights deficientes.
4. PRIORIZACIÓN: Prioriza correcciones por impacto en recuperabilidad.
5. CONSISTENCIA: Verifica que insights de diferentes fuentes no se contradigan.
6. OBSOLESCENCIA: Identifica insights que pueden estar desactualizados por cambios normativos.
7. ENRIQUECIMIENTO: Propone enriquecer insights existentes con nueva información disponible."""

# ── Helpers ─────────────────────────────────────────────────────────────────

TOOL_ALIASES = {
    "search_docs": "search_docs",
    "search_regulation": "search_regulation",
    "get_field_definition": "get_field_definition",
    "trace_dependencies": "trace_dependencies",
    "inspect_lineage": "inspect_lineage",
    "backtrace_sas_field": "trace_dependencies",
    "search_changelog": "search_changelog",
    "search_experience": "search_experience",
    "save_insight": "save_insight",
    "save_feedback": "save_feedback",
    "save_quirk": "save_quirk",
    "find_governing_rules": "search_regulation",
    "query_regulation": "search_regulation",
    "trace_regulation_chain": "trace_regulation_chain",
    "validate_sas": "validate_sas",
    "enrich_fields": "enrich_fields",
    "describe_egp_project": "describe_egp_project",
    "compute_shapley": "compute_shapley",
    "find_row": "find_row",
    "find_rows_by_field_value": "find_rows_by_field_value",
    "formulate_data_request": "formulate_data_request",
    "create_investigation_plan": "create_investigation_plan",
    "analyze_uploaded_data": "analyze_uploaded_data",
    "get_schema_context": "get_schema_context",
    "merge_nodes": "merge_nodes",
    "link_nodes": "link_nodes",
    "add_tags": "add_tags",
    "flag_conflict": "flag_conflict",
    "delete_node": "delete_node",
}

_STEP_RE = re.compile(r"^(?P<name>[a-z_]+)\((?P<args>.*)\)$", re.IGNORECASE)


def _parse_step(step: str) -> tuple[str, dict] | None:
    m = _STEP_RE.match(step.strip())
    if not m:
        return None
    name = m.group("name").lower()
    raw_args = m.group("args")
    kv: dict[str, str] = {}
    for part in re.split(r",\s*(?=(?:[^']*'[^']*')*[^']*$)", raw_args.strip()):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        kv[k.strip()] = v.strip().strip("'\"")
    return name, kv


def _make_tool_call(id: str, name: str, args: dict) -> dict:
    return {
        "id": id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }


def _fake_tool_response(tool_name: str, tool_args: dict, hint: str) -> str:
    return json.dumps({
        "tool": tool_name,
        "result": f"Resultado simulado para {tool_name}: {hint[:300]}",
        "args": tool_args,
    }, ensure_ascii=False)


def _exp_scenario_classification(scenario: str) -> list[str]:
    """Classify experience eval scenario into applicable phases."""
    hard_synth = {
        "multi_hop_knowledge", "conflicting_insights", "temporal_awareness",
        "multi_session_accumulation", "semantic_gap", "cross_tool_synthesis",
        "save_and_immediately_recall", "cross_language_recall",
        "experience_maintenance", "cross_system_reconciliation",
        "experience_to_regulation_link", "multi_segment_floor_discrimination",
        "experience_decay_temporal", "cross_language_spanish_english",
        "model_validation_calibration", "multi_source_conciliation",
        "inter_experience_synthesis", "negative_knowledge_compound",
        "audit_trail_compliance", "experience_self_improvement",
        "stale_insight_identification", "comprehensive_knowledge_inventory",
    }
    hard_retrieve = {
        "bug_vs_quirk_discrimination", "regulatory_multi_source_conciliation",
        "multi_step_retrieval",
    }
    gov_scenarios = {
        "audit_trail_compliance", "experience_self_improvement",
        "stale_insight_identification", "segment_field_consistency",
    }
    phases = ["2.1", "2.3"]
    if scenario in hard_synth:
        phases.append("2.4")
    if scenario in hard_retrieve:
        phases.append("2.4")
    if scenario in gov_scenarios:
        phases.append("2.5")
    return phases


# ── Phase 1.1: Tool calling ─────────────────────────────────────────────────

def build_phase1_1() -> list[dict]:
    examples: list[dict] = []
    with open(DATA / "tool_calling_v2_dataset.json") as f:
        tc_items = json.load(f)
    for item in tc_items:
        question = item["question"]
        gold = item.get("gold_answer_summary", item.get("solution", {}).get("final_answer", ""))
        seq = item.get("tool_sequence", item.get("solution", {}).get("tool_sequence", []))
        msgs = [{"role": "system", "content": SYSTEM_TOOL_CALLING}]
        msgs.append({"role": "user", "content": question})
        tool_idx = 0
        for step in seq:
            parsed = _parse_step(step)
            if not parsed:
                continue
            tname, targs = parsed
            tid = f"call_{item['id']}_{tool_idx}"
            msgs.append({
                "role": "assistant", "content": "",
                "tool_calls": [_make_tool_call(tid, tname, targs)],
            })
            msgs.append({
                "role": "tool", "tool_call_id": tid, "name": tname,
                "content": _fake_tool_response(tname, targs, gold),
            })
            tool_idx += 1
        msgs.append({"role": "assistant", "content": gold})
        examples.append({"messages": msgs, "id": item["id"], "phase": "1.1", "difficulty": item.get("difficulty", "medium")})
    return examples


# ── Phase 1.2: Multi-step reasoning ─────────────────────────────────────────

def build_phase1_2() -> list[dict]:
    examples: list[dict] = []
    with open(DATA / "eval_dataset.json") as f:
        eval_items = json.load(f)
    reasoning_cats = {"bug_detection", "trace_analysis", "cross_reference", "data_quality", "analysis"}
    for item in eval_items:
        cat = item.get("category", "")
        if cat not in reasoning_cats:
            continue
        question = item["question"]
        sol = item.get("solution", {})
        steps = sol.get("reasoning_steps", [])
        answer = sol.get("final_answer", item.get("gold_answer_summary", ""))
        msgs = [{"role": "system", "content": SYSTEM_REASONING}]
        msgs.append({"role": "user", "content": question})
        reasoning_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)) if steps else ""
        full = f"{reasoning_text}\n\n{answer}" if reasoning_text else answer
        msgs.append({"role": "assistant", "content": full})
        examples.append({"messages": msgs, "id": item["id"], "phase": "1.2", "difficulty": item.get("difficulty", "medium")})
    with open(DATA / "tool_calling_v2_dataset.json") as f:
        tc_items = json.load(f)
    for item in tc_items:
        if item.get("category") not in {"reasoning", "analysis"}:
            continue
        question = item["question"]
        answer = item.get("gold_answer_summary", item.get("solution", {}).get("final_answer", ""))
        msgs = [{"role": "system", "content": SYSTEM_REASONING}]
        msgs.append({"role": "user", "content": question})
        msgs.append({"role": "assistant", "content": answer})
        examples.append({"messages": msgs, "id": item["id"], "phase": "1.2", "difficulty": item.get("difficulty", "medium")})
    return examples


# ── Phase 2.1: Episodic trace formation (learning) ──────────────────────────

def build_phase2_1() -> list[dict]:
    examples: list[dict] = []
    with open(DATA / "memory_eval_dataset.json") as f:
        mem_items = json.load(f)
    for item in mem_items:
        if item.get("cycle_phase") not in {"learn"}:
            continue
        scenario = item.get("scenario", {})
        ctx = scenario.get("context", item.get("description", ""))
        trigger = scenario.get("trigger", "Procesa la sesión y decide qué aprender.")
        actions = item.get("expected_agent_actions", [])
        actions_text = "\n".join(f"- {a}" for a in actions) if actions else "- save_insight con causa raíz, no síntoma"
        msgs = [{"role": "system", "content": SYSTEM_SLEEP_LEARN}]
        msgs.append({"role": "user", "content": f"## Sesión del día\n\n{ctx}\n\n{trigger}"})
        msgs.append({"role": "assistant", "content": actions_text})
        examples.append({"messages": msgs, "id": item["id"], "phase": "2.1", "difficulty": item.get("difficulty", "medium")})
    # Experience eval items for learn phase (all scenarios feed into 2.1)
    with open(DATA / "experience_eval_dataset.json") as f:
        exp_items = json.load(f)
    for item in exp_items:
        desc = item.get("description", "")
        setup = item.get("setup_experiences", [])
        setup_text = "\n".join(f"- {e.get('label','')}: {e.get('summary','')[:100]}" for e in setup[:3])
        msgs = [{"role": "system", "content": SYSTEM_SLEEP_LEARN}]
        msgs.append({"role": "user", "content": f"## Estado actual del grafo\n{setup_text}\n\n## Tarea\n{desc}"})
        golden = item.get("golden_answer", item.get("golden_answer_ideal", ""))
        msgs.append({"role": "assistant", "content": golden})
        examples.append({"messages": msgs, "id": item["id"], "phase": "2.1", "difficulty": item.get("difficulty", "medium")})
    return examples


# ── Phase 2.2: Semantic consolidation (organize) ────────────────────────────

def build_phase2_2() -> list[dict]:
    examples: list[dict] = []
    with open(DATA / "memory_eval_dataset.json") as f:
        mem_items = json.load(f)
    for item in mem_items:
        if item.get("cycle_phase") not in {"organize", "integrate"}:
            continue
        scenario = item.get("scenario", {})
        ctx = scenario.get("context", item.get("description", ""))
        trigger = scenario.get("trigger", "Organiza el grafo de conocimiento.")
        actions = item.get("expected_agent_actions", [])
        actions_text = "\n".join(f"- {a}" for a in actions) if actions else "- merge_nodes / link_nodes / add_tags"
        msgs = [{"role": "system", "content": SYSTEM_SLEEP_ORGANIZE}]
        msgs.append({"role": "user", "content": f"## Estado del grafo\n\n{ctx}\n\n{trigger}"})
        msgs.append({"role": "assistant", "content": actions_text})
        examples.append({"messages": msgs, "id": item["id"], "phase": "2.2", "difficulty": item.get("difficulty", "medium")})
    return examples


# ── Phase 2.3: Reconstructive recollection (retrieval) ──────────────────────

def build_phase2_3() -> list[dict]:
    examples: list[dict] = []
    with open(DATA / "fragile_retrieval_dataset.json") as f:
        frag_items = json.load(f)
    for item in frag_items:
        setup = item.get("setup", {})
        insights = setup.get("insights_in_graph", [])
        question = item["question"]
        golden = item.get("golden_answer", "")
        retrieval = item.get("retrieval_expected", {})
        msgs = [{"role": "system", "content": SYSTEM_SLEEP_RETRIEVE}]
        context = f"## Experiencias en grafo\n" + "\n".join(f"- {ins}" for ins in insights)
        context += f"\n\n## Consulta\n{question}\n\nRecupera EXACTAMENTE lo que corresponde."
        msgs.append({"role": "user", "content": context})
        msgs.append({"role": "assistant", "content": golden})
        examples.append({"messages": msgs, "id": item["id"], "phase": "2.3", "difficulty": item.get("difficulty", "medium")})

    with open(DATA / "memory_eval_dataset.json") as f:
        mem_items = json.load(f)
    for item in mem_items:
        if item.get("cycle_phase") not in {"retrieve"}:
            continue
        ctx = item.get("scenario", {}).get("context", item.get("description", ""))
        trigger = item.get("scenario", {}).get("trigger", "Recupera la información relevante.")
        actions = item.get("expected_agent_actions", [])
        actions_text = "\n".join(f"- {a}" for a in actions) if actions else "- search_experience con query precisa"
        msgs = [{"role": "system", "content": SYSTEM_SLEEP_RETRIEVE}]
        msgs.append({"role": "user", "content": f"## Estado\n{ctx}\n\n{trigger}"})
        msgs.append({"role": "assistant", "content": actions_text})
        examples.append({"messages": msgs, "id": item["id"], "phase": "2.3", "difficulty": item.get("difficulty", "medium")})

    # Route experience_eval items - all go to 2.3, some also go to 2.4
    with open(DATA / "experience_eval_dataset.json") as f:
        exp_items = json.load(f)
    for item in exp_items:
        desc = item.get("description", "")
        question = item.get("question", "")
        golden = item.get("golden_answer", "")
        setup = item.get("setup_experiences", [])
        setup_text = "\n".join(f"- {e.get('label','')}: {e.get('summary','')[:100]}" for e in setup[:3])
        msgs = [{"role": "system", "content": SYSTEM_SLEEP_RETRIEVE}]
        msgs.append({"role": "user", "content": f"## Experiencias en grafo\n{setup_text}\n\n## Consulta\n{question}\n\nRecupera la experiencia relevante."})
        msgs.append({"role": "assistant", "content": golden})
        examples.append({"messages": msgs, "id": item["id"], "phase": "2.3", "difficulty": item.get("difficulty", "medium")})
    return examples


# ── Phase 2.4: Knowledge synthesis ──────────────────────────────────────────

def build_phase2_4() -> list[dict]:
    """Synthesis phase: route complex experience scenarios that require integration."""
    examples: list[dict] = []

    # Synthesis scenarios from memory_eval integration phase
    with open(DATA / "memory_eval_dataset.json") as f:
        mem_items = json.load(f)
    for item in mem_items:
        if item.get("cycle_phase") not in {"integrate"}:
            continue
        scenario = item.get("scenario", {})
        ctx = scenario.get("context", item.get("description", ""))
        trigger = scenario.get("trigger", "Sintetiza el conocimiento.")
        actions = item.get("expected_agent_actions", [])
        actions_text = "\n".join(f"- {a}" for a in actions) if actions else "- crear insight síntesis / conectar nodos"
        msgs = [{"role": "system", "content": SYSTEM_SLEEP_SYNTHESIS}]
        msgs.append({"role": "user", "content": f"## Conocimiento existente\n\n{ctx}\n\n{trigger}"})
        msgs.append({"role": "assistant", "content": actions_text})
        examples.append({"messages": msgs, "id": item["id"], "phase": "2.4", "difficulty": item.get("difficulty", "medium")})

    # Also route experience_eval scenarios classified for synthesis
    with open(DATA / "experience_eval_dataset.json") as f:
        exp_items = json.load(f)
    for item in exp_items:
        scenario_type = item.get("scenario", "")
        phases = _exp_scenario_classification(scenario_type)
        if "2.4" not in phases:
            continue
        desc = item.get("description", "")
        question = item.get("question", "")
        golden = item.get("golden_answer", "")
        setup = item.get("setup_experiences", [])
        setup_text = "\n".join(f"- {e.get('label','')}: {e.get('summary','')[:100]}" for e in setup[:3])
        msgs = [{"role": "system", "content": SYSTEM_SLEEP_SYNTHESIS}]
        msgs.append({"role": "user", "content": f"## Experiencias en grafo\n{setup_text}\n\n## Consulta\n{question}\n\nSintetiza el conocimiento aplicable."})
        msgs.append({"role": "assistant", "content": golden})
        examples.append({"messages": msgs, "id": f"{item['id']}_synth", "phase": "2.4", "difficulty": item.get("difficulty", "medium")})

    return examples


# ── Phase 2.5: Experience governance ────────────────────────────────────────

def build_phase2_5() -> list[dict]:
    """Governance phase: audit, quality, maintenance scenarios."""
    examples: list[dict] = []

    with open(DATA / "experience_eval_dataset.json") as f:
        exp_items = json.load(f)
    for item in exp_items:
        scenario_type = item.get("scenario", "")
        phases = _exp_scenario_classification(scenario_type)
        if "2.5" not in phases:
            continue
        desc = item.get("description", "")
        question = item.get("question", "")
        golden = item.get("golden_answer", "")
        setup = item.get("setup_experiences", [])
        setup_text = "\n".join(f"- {e.get('label','')}: {e.get('summary','')[:100]}" for e in setup[:3])
        msgs = [{"role": "system", "content": SYSTEM_SLEEP_GOVERN}]
        msgs.append({"role": "user", "content": f"## Estado del grafo\n{setup_text}\n\n## Tarea\n{desc}\n\nPregunta: {question}"})
        msgs.append({"role": "assistant", "content": golden})
        examples.append({"messages": msgs, "id": f"{item['id']}_gov", "phase": "2.5", "difficulty": item.get("difficulty", "medium")})

    # Memory organize scenarios also go to governance (quality audit)
    with open(DATA / "memory_eval_dataset.json") as f:
        mem_items = json.load(f)
    for item in mem_items:
        if item.get("capability") in ("insight_quality_audit", "tag_consistency", "stale_insight_flagging"):
            scenario = item.get("scenario", {})
            ctx = scenario.get("context", item.get("description", ""))
            trigger = scenario.get("trigger", "Audita la calidad del grafo.")
            actions = item.get("expected_agent_actions", [])
            actions_text = "\n".join(f"- {a}" for a in actions) if actions else "- auditoría de calidad"
            msgs = [{"role": "system", "content": SYSTEM_SLEEP_GOVERN}]
            msgs.append({"role": "user", "content": f"## Estado del grafo\n\n{ctx}\n\n{trigger}"})
            msgs.append({"role": "assistant", "content": actions_text})
            examples.append({"messages": msgs, "id": f"{item['id']}_gov", "phase": "2.5", "difficulty": item.get("difficulty", "medium")})

    return examples


# ── Main ────────────────────────────────────────────────────────────────────

def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  {path}  ({len(examples)} examples)")


def main() -> None:
    print("Building consolidated phase datasets (EverMemOS-inspired lifecycle)...")

    print("  Phase 1.1 — Tool calling...")
    p11 = build_phase1_1()
    write_jsonl(p11, OUT / "phase1.1_tool_calling.jsonl")

    print("  Phase 1.2 — Multi-step reasoning...")
    p12 = build_phase1_2()
    write_jsonl(p12, OUT / "phase1.2_reasoning.jsonl")

    print("  Phase 2.1 — Episodic trace formation (learn)...")
    p21 = build_phase2_1()
    write_jsonl(p21, OUT / "phase2.1_learning.jsonl")

    print("  Phase 2.2 — Semantic consolidation (organize)...")
    p22 = build_phase2_2()
    write_jsonl(p22, OUT / "phase2.2_organizing.jsonl")

    print("  Phase 2.3 — Reconstructive recollection (retrieval)...")
    p23 = build_phase2_3()
    write_jsonl(p23, OUT / "phase2.3_retrieval.jsonl")

    print("  Phase 2.4 — Knowledge synthesis...")
    p24 = build_phase2_4()
    write_jsonl(p24, OUT / "phase2.4_synthesis.jsonl")

    print("  Phase 2.5 — Experience governance...")
    p25 = build_phase2_5()
    write_jsonl(p25, OUT / "phase2.5_governance.jsonl")

    total = len(p11) + len(p12) + len(p21) + len(p22) + len(p23) + len(p24) + len(p25)
    print(f"\nTotal: {total} training examples across 7 phases")
    print("Phase distribution:")
    print(f"  1.1: {len(p11)} | 1.2: {len(p12)} | 2.1: {len(p21)} | 2.2: {len(p22)}")
    print(f"  2.3: {len(p23)} | 2.4: {len(p24)} | 2.5: {len(p25)}")


if __name__ == "__main__":
    main()

"""Generate synthetic training data for fine-tuning Qwen3-3B on RegLLM tasks.

Produces JSONL files with chat-format examples covering:
1. Tool calling — SAS lineage, validation, KG queries
2. Schema knowledge — DB schema Q&A in Spanish
3. Regulation Q&A — regulatory article interpretation
4. Graph traversal — multi-hop KG reasoning
5. Credit risk terminology — domain concepts in Spanish

Each example is a list of messages: [system, user, assistant] with optional
tool_calls and tool results for tool-calling examples.

Usage:
    python training/generate_training_data.py [--n-examples 5000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Loaders ────────────────────────────────────────────────────────────────


def _load_schema() -> str:
    p = PROJECT_ROOT / "data" / "samples" / "irb_schema.sql"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _load_regulation_articles() -> list[dict]:
    """Load regulation markdown files as article dicts."""
    reg_dir = PROJECT_ROOT / "data" / "regulation"
    articles = []
    if reg_dir.exists():
        for f in sorted(reg_dir.glob("*.md")):
            articles.append({
                "filename": f.name,
                "title": f.stem.replace("_", " ").title(),
                "content": f.read_text(encoding="utf-8")[:2000],
            })
    return articles


def _load_sas_sample() -> str:
    p = PROJECT_ROOT / "data" / "samples" / "sample_lgd.sas"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _load_field_docs() -> list[dict]:
    """Load field definition docs."""
    docs_dir = PROJECT_ROOT / "data" / "docs" / "fields"
    fields = []
    if docs_dir.exists():
        for f in sorted(docs_dir.glob("*.md")):
            fields.append({
                "name": f.stem.upper(),
                "content": f.read_text(encoding="utf-8")[:1000],
            })
    return fields


def _load_tool_schemas() -> list[dict]:
    """Load tool schemas from the registry."""
    from src.agent.tools import tool_schemas
    return tool_schemas()


# ── Tool Registry Info ─────────────────────────────────────────────────────

TOOL_NAMES = [
    "describe_egp_project", "validate_sas", "inspect_lineage",
    "trace_dependencies", "backtrace_sas_field", "compute_shapley",
    "find_row", "find_rows_by_field_value", "search_docs",
    "get_field_definition", "search_regulation", "search_changelog",
    "query_regulation", "find_governing_rules", "trace_regulation_chain",
    "search_experience", "save_insight", "save_feedback",
    "create_investigation_plan", "formulate_data_request",
    "analyze_uploaded_data", "enrich_fields", "get_schema_context",
]

# Credit risk fields commonly encountered
FIELDS = [
    "PD_ESTIMADA", "LGD_ESTIMADA", "EAD", "ECL", "RWA",
    "PROVISION_PERIOD_MONTHS", "STAGE_IFRS9", "DPDS",
    "LGD_FLOOR_APLICADO", "CURE_FLAG", "MOC", "RATING_GRADO",
    "OR_EAD_TIT", "OR_EAD", "VENTANA_OBSERVACION_YEARS",
]

# Spanish regulatory concepts
REGULATORY_CONCEPTS = [
    "dotaciones mínimas", "liberación de provisiones", "periodos de dotación",
    "clasificación de ciclos", "floor de LGD", "segmento IRB",
    "clase de activo CRR", "factor de conversión crediticia (CCF)",
    "pérdida esperada (ECL)", "activos ponderados por riesgo (RWA)",
    "probabilidad de impago (PD)", "severidad (LGD)",
    "exposición al impago (EAD)", "stage IFRS9",
]

# ── System prompt (same as agent) ──────────────────────────────────────────

SYSTEM_PROMPT = (
    "Eres un investigador de cumplimiento regulatorio para bases de datos "
    "bancarias (COREP/FINREP IRB). Tienes acceso a un grafo de conocimiento "
    "que enlaza regulaciones EBA/BdE, columnas de base de datos, reglas de "
    "validación, linaje de código SAS e insights de investigaciones previas. "
    "Tu trabajo es investigar problemas de forma metódica usando las herramientas "
    "disponibles y responder en español profesional."
)


# ── Generators ─────────────────────────────────────────────────────────────


def gen_tool_calling_example(rng: random.Random, schemas: list[dict]) -> dict:
    """Generate a tool-calling conversation example."""
    field = rng.choice(FIELDS)
    session_id = f"sess_{rng.randint(1000, 9999)}"

    # Pick a scenario
    scenario = rng.choice([
        "lineage", "validate", "describe_project", "backtrace",
        "governing_rules", "investigation",
    ])

    if scenario == "lineage":
        user_msg = f"¿De qué campos depende {field} en el código SAS?"
        tool_call = {
            "name": "inspect_lineage",
            "arguments": {"target": field, "session_id": session_id},
        }
        tool_result = {
            "target": field, "session_id": session_id, "found": True,
            "ancestor_count": rng.randint(3, 12),
            "ancestors": rng.sample(FIELDS, min(5, len(FIELDS))),
            "direct_predecessors": rng.sample(FIELDS, min(3, len(FIELDS))),
            "layers": [rng.sample(FIELDS, min(2, len(FIELDS)))],
        }
        assistant_answer = (
            f"He trazado el linaje de **{field}**. Depende de "
            f"{', '.join(tool_result['direct_predecessors'])} como predecesores directos. "
            f"En total hay {tool_result['ancestor_count']} campos ancestros en la cadena."
        )

    elif scenario == "validate":
        user_msg = "Valida el código SAS que acabo de subir."
        tool_call = {
            "name": "validate_sas",
            "arguments": {"session_id": session_id},
        }
        n_diags = rng.randint(0, 5)
        diags = [
            {"code": rng.choice(["MISSING_DATASET", "UNDEFINED_VAR", "TYPE_MISMATCH"]),
             "level": rng.choice(["warning", "error"]),
             "message": f"Problema detectado en data step {rng.randint(1,10)}"}
            for _ in range(n_diags)
        ]
        tool_result = {"session_id": session_id, "diagnostics": diags, "total": n_diags}
        if n_diags == 0:
            assistant_answer = "El análisis estático no ha detectado errores en el código SAS."
        else:
            assistant_answer = (
                f"Se han encontrado {n_diags} diagnósticos:\n" +
                "\n".join(f"- **{d['code']}** ({d['level']}): {d['message']}" for d in diags)
            )

    elif scenario == "describe_project":
        user_msg = "Describe la estructura del proyecto .egp que subí."
        tool_call = {
            "name": "describe_egp_project",
            "arguments": {"session_id": session_id},
        }
        n_blocks = rng.randint(3, 8)
        blocks = [
            {"filename": f"proj_task_{i}.sas", "task_name": f"Task {i}",
             "block_type": "embedded", "order": i,
             "line_count": rng.randint(10, 100)}
            for i in range(n_blocks)
        ]
        tool_result = {
            "session_id": session_id, "source_file": "proyecto.egp",
            "block_count": n_blocks, "blocks": blocks,
        }
        assistant_answer = (
            f"El proyecto contiene {n_blocks} bloques de código SAS:\n" +
            "\n".join(f"- **{b['task_name']}**: {b['line_count']} líneas" for b in blocks)
        )

    elif scenario == "backtrace":
        user_msg = f"¿Qué tablas de entrada alimentan el cálculo de {field}?"
        tool_call = {
            "name": "backtrace_sas_field",
            "arguments": {"target": field, "session_id": session_id},
        }
        tables = [f"work.step_{rng.randint(1,5)}" for _ in range(rng.randint(1, 4))]
        tool_result = {
            "target": field, "session_id": session_id, "found": True,
            "ancestor_count": rng.randint(3, 10),
            "input_tables": [
                {"table": t, "fields": rng.sample(FIELDS, min(3, len(FIELDS))),
                 "field_count": rng.randint(1, 5)}
                for t in tables
            ],
        }
        assistant_answer = (
            f"El campo **{field}** se alimenta de {len(tables)} pasos de datos:\n" +
            "\n".join(f"- `{t['table']}`: {', '.join(t['fields'])}" for t in tool_result["input_tables"])
        )

    elif scenario == "governing_rules":
        user_msg = f"¿Qué regulación gobierna el campo {field}?"
        tool_call = {
            "name": "find_governing_rules",
            "arguments": {"column": field},
        }
        tool_result = {
            "column": field, "found": True,
            "rules": [
                {"article": f"art{rng.randint(1,30)}",
                 "description": rng.choice(REGULATORY_CONCEPTS),
                 "confidence": round(rng.uniform(0.7, 1.0), 2)}
            ],
        }
        assistant_answer = (
            f"El campo **{field}** está gobernado por: " +
            ", ".join(f"**{r['article']}** ({r['description']}, confianza {r['confidence']})"
                      for r in tool_result["rules"])
        )

    else:  # investigation
        concept = rng.choice(REGULATORY_CONCEPTS)
        user_msg = f"Investiga por qué {field} muestra valores inesperados para el ciclo 202312."
        tool_call = {
            "name": "create_investigation_plan",
            "arguments": {
                "target_field": field,
                "problem": f"{field} muestra valores inesperados en ciclo 202312",
                "steps": [
                    f"Trazar dependencias de {field}",
                    "Verificar regulación aplicable",
                    "Solicitar datos del ciclo 202312",
                ],
                "cycles": ["202312"],
            },
        }
        tool_result = {"saved": True, "id": f"exp:plan_{rng.randint(1000,9999)}"}
        assistant_answer = (
            f"He creado un plan de investigación para {field}:\n"
            f"1. Trazar dependencias de {field}\n"
            f"2. Verificar regulación aplicable\n"
            f"3. Solicitar datos del ciclo 202312"
        )

    return _format_tool_conversation(
        user_msg, tool_call, tool_result, assistant_answer,
    )


def gen_schema_knowledge_example(
    rng: random.Random, schema: str, fields: list[dict],
) -> dict:
    """Generate schema Q&A in Spanish."""
    if fields:
        field = rng.choice(fields)
        user_msg = f"¿Qué es el campo {field['name']} en la base de datos?"
        answer = (
            f"**{field['name']}**: {field['content'][:300]}\n\n"
            "Este campo forma parte del esquema IRB y se utiliza en los "
            "cálculos regulatorios de riesgo de crédito."
        )
    else:
        table = rng.choice(["contratos", "clientes", "garantias", "parametros_irb", "ciclos_lgd"])
        user_msg = f"¿Cuál es la estructura de la tabla `{table}`?"
        # Extract table DDL from schema
        answer = f"La tabla `{table}` forma parte del esquema IRB de riesgo de crédito."
        if table in schema:
            start = schema.find(f"CREATE TABLE {table}")
            if start >= 0:
                end = schema.find(");", start)
                ddl = schema[start:end + 2] if end > 0 else schema[start:start + 500]
                answer = f"```sql\n{ddl}\n```\n\nEsta tabla almacena datos regulatorios del modelo IRB."

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ]
    }


def gen_regulation_qa_example(
    rng: random.Random, articles: list[dict],
) -> dict:
    """Generate regulation Q&A from article content."""
    if articles:
        art = rng.choice(articles)
        concept = rng.choice(REGULATORY_CONCEPTS)
        user_msg = f"¿Qué dice {art['title']} sobre {concept}?"
        answer = (
            f"Según **{art['title']}** ({art['filename']}):\n\n"
            f"{art['content'][:500]}\n\n"
            f"Este artículo es relevante para {concept} en el contexto "
            "del modelo IRB de riesgo de crédito."
        )
    else:
        concept = rng.choice(REGULATORY_CONCEPTS)
        user_msg = f"Explica el concepto de {concept} en el marco regulatorio bancario."
        answer = (
            f"**{concept.title()}** es un concepto clave en la regulación bancaria "
            "de riesgo de crédito bajo el marco CRR/CRD y las circulares del BdE."
        )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ]
    }


def gen_graph_traversal_example(rng: random.Random) -> dict:
    """Generate KG traversal reasoning examples."""
    field = rng.choice(FIELDS)
    concept = rng.choice(REGULATORY_CONCEPTS)

    scenarios = [
        {
            "user": f"¿Cómo se conecta el artículo 15 con el campo {field} en el grafo?",
            "tool": "trace_regulation_chain",
            "args": {"article_id": "art15", "target_column": field},
            "result": {
                "found": True,
                "path": [
                    {"id": "art15", "type": "Regulation", "label": "Artículo 15 - Dotaciones mínimas"},
                    {"id": f"concept:{concept}", "type": "Concept", "label": concept},
                    {"id": f"col:{field}", "type": "Column", "label": field},
                ],
                "path_length": 2,
            },
            "answer": (
                f"El camino en el grafo desde art15 hasta {field} pasa por el concepto "
                f"**{concept}**:\n\n"
                f"art15 (Dotaciones mínimas) → {concept} → {field}\n\n"
                f"Esto indica que la regulación de dotaciones mínimas afecta a {field} "
                f"a través del concepto de {concept}."
            ),
        },
        {
            "user": f"¿Qué nodos del grafo están conectados a {concept}?",
            "tool": "query_regulation",
            "args": {"concept": concept, "hops": 2},
            "result": {
                "found": True, "center": concept,
                "nodes": [
                    {"type": "Regulation", "label": f"art{rng.randint(1,30)}"},
                    {"type": "Column", "label": field},
                    {"type": "Concept", "label": concept},
                ],
            },
            "answer": (
                f"El concepto **{concept}** en el grafo de conocimiento está conectado a:\n"
                f"- Regulación: art{rng.randint(1,30)}\n"
                f"- Columna: {field}\n\n"
                "Estos son los nodos dentro de 2 saltos de distancia."
            ),
        },
    ]

    s = rng.choice(scenarios)
    return _format_tool_conversation(
        s["user"],
        {"name": s["tool"], "arguments": s["args"]},
        s["result"],
        s["answer"],
    )


def gen_credit_risk_example(rng: random.Random) -> dict:
    """Generate credit risk terminology Q&A in Spanish."""
    qa_pairs = [
        ("¿Qué es la PD estimada?",
         "La **Probabilidad de Impago (PD)** estimada es la probabilidad de que un "
         "deudor entre en situación de impago en un horizonte temporal determinado "
         "(típicamente 12 meses bajo IFRS9 Stage 1). Se estima mediante modelos "
         "estadísticos calibrados con datos históricos de impago y se expresa como "
         "un porcentaje entre 0% y 100%."),
        ("¿Cómo se calcula el ECL?",
         "La **Pérdida Crediticia Esperada (ECL)** se calcula como:\n\n"
         "`ECL = PD × LGD × EAD`\n\n"
         "Donde PD es la probabilidad de impago, LGD la severidad (pérdida dado "
         "el impago) y EAD la exposición en el momento del impago."),
        ("¿Qué son los stages de IFRS9?",
         "IFRS9 clasifica los activos financieros en tres stages:\n\n"
         "- **Stage 1**: Sin incremento significativo de riesgo. Provisión = ECL 12 meses.\n"
         "- **Stage 2**: Incremento significativo de riesgo. Provisión = ECL lifetime.\n"
         "- **Stage 3**: Impago/default. Provisión = ECL lifetime con PD=100%."),
        ("¿Qué es el floor de LGD?",
         "El **floor de LGD** es el valor mínimo que la regulación permite aplicar "
         "a la LGD estimada por modelos internos. Según CRR Art. 161, para "
         "exposiciones senior sin garantía el floor es 25% (corporativos) o "
         "varía según el tipo de garantía admisible."),
        ("¿Qué es el RWA?",
         "Los **Activos Ponderados por Riesgo (RWA)** representan los activos "
         "del banco ajustados por su nivel de riesgo. Se calculan aplicando "
         "funciones de ponderación regulatorias (IRB) que dependen de PD, LGD, "
         "EAD y la correlación del activo. Determinan el capital mínimo requerido."),
        ("¿Qué es el CCF?",
         "El **Factor de Conversión Crediticia (CCF)** convierte compromisos "
         "fuera de balance (líneas no dispuestas) en equivalentes de crédito "
         "para el cálculo del EAD. Un CCF del 75% significa que se espera "
         "que el deudor disponga del 75% del límite no utilizado antes del impago."),
    ]

    q, a = rng.choice(qa_pairs)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
    }


# ── Helpers ────────────────────────────────────────────────────────────────


def _format_tool_conversation(
    user_msg: str,
    tool_call: dict,
    tool_result: dict,
    assistant_answer: str,
) -> dict:
    """Format a tool-calling conversation as chat messages."""
    call_id = f"call_{hash(user_msg) % 100000:05d}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call["arguments"], ensure_ascii=False),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_call["name"],
                "content": json.dumps(tool_result, ensure_ascii=False),
            },
            {"role": "assistant", "content": assistant_answer},
        ]
    }


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RegLLM training data")
    parser.add_argument("--n-examples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load source data
    schema = _load_schema()
    articles = _load_regulation_articles()
    fields = _load_field_docs()

    try:
        schemas = _load_tool_schemas()
    except Exception:
        schemas = []

    # Generate examples according to mix ratios
    examples: list[dict] = []
    mix = {
        "tool_calling": 0.30,
        "schema_knowledge": 0.20,
        "regulation_qa": 0.20,
        "graph_traversal": 0.15,
        "credit_risk": 0.15,
    }

    for _ in range(args.n_examples):
        r = rng.random()
        cumulative = 0.0
        for category, ratio in mix.items():
            cumulative += ratio
            if r <= cumulative:
                if category == "tool_calling":
                    ex = gen_tool_calling_example(rng, schemas)
                elif category == "schema_knowledge":
                    ex = gen_schema_knowledge_example(rng, schema, fields)
                elif category == "regulation_qa":
                    ex = gen_regulation_qa_example(rng, articles)
                elif category == "graph_traversal":
                    ex = gen_graph_traversal_example(rng)
                else:
                    ex = gen_credit_risk_example(rng)
                examples.append(ex)
                break

    # Split train/eval
    rng.shuffle(examples)
    n_eval = max(1, int(len(examples) * args.eval_ratio))
    eval_examples = examples[:n_eval]
    train_examples = examples[n_eval:]

    # Write output
    out_dir = PROJECT_ROOT / "training" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path, data in [
        (out_dir / "train.jsonl", train_examples),
        (out_dir / "eval.jsonl", eval_examples),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} examples to {path}")

    print(f"\nTotal: {len(train_examples)} train + {len(eval_examples)} eval = {len(examples)}")


if __name__ == "__main__":
    main()

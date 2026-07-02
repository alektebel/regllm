"""Prompt builder + batch generator for DQ check generation RL."""

from __future__ import annotations

import random

from training.dq.coherence_rules import CoherenceRule, RULES, TABLE_NAME, PK_COLUMN

SYSTEM_PROMPT = (
    "Eres ingeniero/a de datos del equipo de Riesgo IRB. Generas chequeos SQL "
    "de **coherencia de base de datos**: predicados que cruzan al menos 2 "
    "columnas para detectar filas internamente contradictorias.\n\n"
    "Reglas estrictas:\n"
    "- Emites UN único cheque como JSON dentro de etiquetas <check>…</check>.\n"
    "- Claves requeridas: name, description, severity, category, sql.\n"
    "- severity ∈ {HIGH, MED, LOW}; category ∈ {cross_field, cross_table, cardinality}.\n"
    "- `sql` debe ser SELECT portable — válido en SQLite y en SAS PROC SQL.\n"
    "- El WHERE debe combinar ≥ 2 columnas (referenciar la misma columna dos "
    "veces no cuenta). Nunca uses SELECT *.\n"
    "- La consulta debe devolver 0 filas sobre datos coherentes y capturar "
    "exactamente la violación descrita.\n"
    "- Responde únicamente el bloque <check>; sin texto adicional."
)

OUTPUT_TEMPLATE = (
    "Formato de salida obligatorio:\n"
    "<check>\n"
    '{"name": "nombre_corto_snake_case",\n'
    ' "description": "descripción humana de la incoherencia",\n'
    ' "severity": "HIGH",\n'
    ' "category": "cross_field",\n'
    ' "sql": "SELECT <pk_o_columnas> FROM <tabla> WHERE <predicado_que_combina_2+_columnas>"}\n'
    "</check>\n\n"
    "Ejemplo (para otra regla, sólo como referencia de forma):\n"
    "<check>\n"
    '{"name": "fecha_cierre_sin_estado_cerrado",\n'
    ' "description": "Contrato con fecha de cierre informada pero estado distinto de CERRADO/DEFAULT.",\n'
    ' "severity": "HIGH",\n'
    ' "category": "cross_field",\n'
    ' "sql": "SELECT contrato_id FROM contratos WHERE fecha_cierre IS NOT NULL '
    'AND estado NOT IN (' + "'CERRADO','DEFAULT'" + ')"}\n'
    "</check>"
)


def _ddl_excerpt(rule: CoherenceRule) -> str:
    cols = ", ".join(rule.columns)
    return (
        f"TABLA: {rule.table}\n"
        f"COLUMNAS RELEVANTES PARA ESTA REGLA: {cols}\n"
        f"PK: {PK_COLUMN} (texto, ej. '100001_201903').\n"
        f"Nota: STAGE_IFRS9, CURE_FLAG, ADJUDICACION_FLAG son TEXT ('0','1','2','3'); "
        f"DPDS, MES_CICLO son INTEGER; el resto numérico REAL."
    )


def build_prompt(rule: CoherenceRule) -> list[dict]:
    user = (
        f"Genera el chequeo SQL de coherencia para esta regla:\n\n"
        f">>> {rule.description}\n\n"
        f"Severidad esperada: {rule.severity}\n"
        f"Categoría esperada: {rule.category}\n"
        + (f"Referencia regulatoria: {rule.regulation_ref}\n" if rule.regulation_ref else "")
        + f"\n{_ddl_excerpt(rule)}\n\n{OUTPUT_TEMPLATE}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate_batch(
    n: int,
    seed: int = 42,
    include_decoys: bool = True,
) -> list[tuple[CoherenceRule, list[dict]]]:
    """Sample n rules uniformly (decoys included by default for variety)."""
    pool = [r for r in RULES if include_decoys or not r.rule_id.startswith("dq_decoy_")]
    rng = random.Random(seed)
    picks = rng.choices(pool, k=n)
    return [(r, build_prompt(r)) for r in picks]


def get_rule(rule_id: str) -> CoherenceRule:
    from training.dq.coherence_rules import RULES_BY_ID
    if rule_id not in RULES_BY_ID:
        raise KeyError(f"unknown rule_id: {rule_id!r}. available: {list(RULES_BY_ID)}")
    return RULES_BY_ID[rule_id]

"""Prompts for LLM-generated complete DQC catalogs."""

from __future__ import annotations

import json

from training.dq.catalog_schema import FieldUniverse, catalog_template, load_field_universe
from training.dq.coherence_rules import PK_COLUMN, TABLE_NAME

CATALOG_SYSTEM_PROMPT = """\
Eres un experto en calidad de datos para reporting regulatorio bancario IRB \
(COREP/FINREP). Generas un CATÁLOGO COMPLETO de controles DQC para la tabla \
de ciclos de recuperación.

Objetivos obligatorios:
1. **Cobertura 100%**: cada campo auditable del universo debe aparecer en \
   `campos_entrada` de al menos un DQC.
2. **Coherencia de campos**: los campos declarados deben corresponder a los \
   usados en `regla_sql`. Controles de tipo consistencia/formula/referencial \
   deben combinar ≥ 2 campos relacionados.
3. **IDs únicos**: cada `dqc_id` distinto, formato `DQC_<VARIABLE>_NNN`.
4. **SQL portable**: SELECT sobre `{table}` que devuelve `{pk}` de filas \
   violadoras. Válido en SQLite y SAS PROC SQL.

Prioriza relaciones cruzadas (no trivial NULL checks salvo completitud \
regulatoria justificada).

Responde SOLO con JSON válido: {{"dataset": "...", "table": "...", "dqcs": [...]}}.
"""


def build_compact_catalog_messages(universe: FieldUniverse | None = None) -> list[dict]:
    """Shorter prompt for local models — grouped coverage, fewer DQCs."""
    universe = universe or load_field_universe()
    system = CATALOG_SYSTEM_PROMPT.format(table=universe.table, pk=universe.pk_column)
    groups = [
        "SEGMENTO, CALIBRATION_SEGMENT, PRODUCTO, TERMINACION, ESTADO_CICLO, CAUSA_DEFAULT",
        "PD_ESTIMADA, LGD_ESTIMADA, LGD_REALIZADA, EAD, ECL, PROVISION, STAGE_IFRS9, CURE_FLAG, DPDS",
        "ADJUDICACION_FLAG, ADJUDICACION_TIPO, ADJUDICACION_VALOR",
        "RECUPERACION_ACUMULADA, COSTE_TOTAL_ACUMULADO, FLUJO_RECUPERADO, FLUJO_NETO, EAD",
        "COLATERAL_TIPO, VALOR_COLATERAL_INICIAL, VALOR_COLATERAL, LTV, OR_DISPTO",
        "ID_FUSION_FINAL, FECHA_DEFAULT, FECHA_REFERENCIA, MES_CICLO, RATING_GRADO, RWA, TASA_DESCUENTO",
        "SALDO_PENDIENTE, INTERESES_ACUMULADOS, TIPO_INTERES_ORIGINAL, TIPO_INTERES_ACTUAL, COSTE_RECUPERACION",
    ]
    user = (
        f"Genera un catálogo DQC JSON para `{universe.table}` con **exactamente {len(groups)} entradas**, "
        f"una por grupo. Cada entrada debe listar TODOS los campos de su grupo en `campos_entrada` "
        f"y verificar coherencia cruzada real (≥2 campos en el SQL WHERE).\n\n"
        "Grupos:\n" + "\n".join(f"{i+1}. {g}" for i, g in enumerate(groups)) + "\n\n"
        "Responde SOLO JSON: {\"dataset\":\"recuperatory_cycles\",\"table\":\"recuperatory_cycles\",\"dqcs\":[...]}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_catalog_messages(universe: FieldUniverse | None = None) -> list[dict]:
    universe = universe or load_field_universe()
    template = catalog_template(universe)
    fields_block = "\n".join(f"- {f}" for f in universe.auditable_fields)
    groups_block = "\n".join(
        f"- {', '.join(g)}" for g in universe.coherence_groups[:12]
    )

    user = (
        f"Genera el catálogo DQC completo para `{universe.table}`.\n\n"
        f"Campos auditable ({len(universe.auditable_fields)} — todos deben estar cubiertos):\n"
        f"{fields_block}\n\n"
        f"Grupos de coherencia conocidos (cada grupo debería tener ≥1 DQC):\n"
        f"{groups_block}\n\n"
        f"Plantilla de un elemento:\n"
        f"{json.dumps(template['dqcs'][0], indent=2, ensure_ascii=False)}\n\n"
        f"Responde con el catálogo completo en JSON."
    )

    system = CATALOG_SYSTEM_PROMPT.format(table=universe.table, pk=universe.pk_column)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_catalog_user_message(universe: FieldUniverse | None = None) -> str:
    return build_catalog_messages(universe)[1]["content"]

"""Generate experience .md files from real database cases and unstructured data scenarios.

Covers:
  - 15 new bug/regulatory-gap/anomaly experiences from SAS pipeline
  - 5 unstructured data scenarios (emails, Excel extracts, meeting notes)
  - Adds corresponding Phase 2.1 training examples (sleep learning)

Usage:
    python scripts/generate_real_cases.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "data" / "experience"
TRAINING_FILE = PROJECT_ROOT / "training" / "data" / "phase2.1_learning.jsonl"

# ── Schema ──────────────────────────────────────────────────────────────────

Experience = dict  # {id, type, priority, tags, fields, articles, source, feedback, body}


def _md(exp: Experience) -> str:
    front = {k: v for k, v in exp.items() if k != "body"}
    yaml_parts = []
    for k, v in front.items():
        if isinstance(v, bool):
            yaml_parts.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, list):
            yaml_parts.append(f"{k}: [{', '.join(str(x) for x in v)}]")
        elif isinstance(v, str):
            yaml_parts.append(f'{k}: "{v}"')
        elif isinstance(v, dict):
            yaml_parts.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
        else:
            yaml_parts.append(f"{k}: {v}")
    fm = "\n".join(yaml_parts)
    return f"---\n{fm}\n---\n\n{exp['body'].strip()}\n"


def _sleep_example(exp: Experience) -> dict:
    """Generate Phase 2.1 training example from an experience."""
    tags_str = ", ".join(exp.get("tags", []))
    fields_str = ", ".join(exp.get("fields", []))
    body_preview = exp["body"][:500].strip()

    user_msg = f"""## Sesión del día — datos de producción

Durante la revisión del pipeline SAS LGD se detectó el siguiente caso:

{body_preview}

### Contexto adicional
- Campos afectados: {fields_str}
- Tags: {tags_str}
- Fuente: {exp.get('source', 'producción')}

### Tarea
Procesa esta información durante el ciclo de sueño. Decide:
1. ¿Merece persistirse en el grafo de conocimiento? (criterio: ¿es información duradera y útil?)
2. ¿A qué campos/artículos debe linkearse?
3. ¿Qué prioridad tiene? (0-1)
4. ¿Es un insight, un quirk, o un feedback?

Si es irrelevante, redundante, o demasiado específico, indícalo explícitamente."""

    # Generate expected action
    exp_type = exp.get("type", "insight")
    if exp_type == "quirk":
        action = f"save_quirk(id={exp['id']}, label='{exp['body'].split(chr(10))[0].lstrip('#').strip()[:60]}', severity={exp.get('severity', 'medium')})"
    elif exp.get("feedback"):
        action = f"save_feedback(type={exp.get('feedback', {}).get('type', 'correction')}, original_claim='...', corrected_understanding='...')"
    else:
        action = f"save_insight(id={exp['id']}, priority={exp.get('priority', 0.5)}, tags={exp.get('tags', [])}, fields={exp.get('fields', [])})"
    link = f"link_nodes(src={exp['id']}, dst=col:{{field}}) for field in {exp.get('fields', [])}"

    assistant_msg = f"""Este caso merece persistencia. No es ruido — es un hallazgo concreto con impacto en cálculos downstream.

## Decisión: GUARDAR

{action}
{link}

## Razonamiento
- El problema tiene causa raíz identificable (no es un error puntual de datos)
- Afecta campos concretos del pipeline
- Tiene potencial de reaparecer en futuras sesiones
- No es redundante con experiencias existentes
- Prioridad {exp.get('priority', 0.5)}: {'crítico' if exp.get('priority', 0.5) >= 0.7 else 'importante' if exp.get('priority', 0.5) >= 0.5 else 'informativo'}"""

    return {
        "messages": [
            {"role": "system", "content": "Eres RegLLM en MODO SUEÑO. Procesas los hallazgos del día y decides qué merece persistirse en el grafo de conocimiento. No todo merece guardarse — filtra ruido, prioriza lo duradero."},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "id": exp["id"],
        "phase": "2.1",
        "difficulty": "medium",
    }


# ── Experiences ─────────────────────────────────────────────────────────────

BUGS = [
    {
        "id": "bug_agg_sum_lgd",
        "type": "insight",
        "priority": 0.7,
        "tags": ["bug", "LGD", "aggregation", "SUM", "MAX"],
        "fields": ["LGD_ESTIMADA", "MoC", "ECL"],
        "articles": [],
        "source": "toy_lgd/08_hardest_compound — revisión agregación LGD",
        "feedback": False,
        "body": """# SUM en lugar de MAX para LGD_ESTIMADA en agregación con fusiones

Cuando hay fusiones (SW_FUSION=1), el JOIN no único duplica filas y `SUM(LGD_ESTIMADA)` produce valores fuera de rango [0,1].

Caso real: CIC_006 con LGD=0.55, al duplicarse por fusión → SUM da 1.10 (imposible).

## Causa raíz
Agregación diseñada para 1-fila-por-ciclo, no actualizada cuando se introdujeron fusiones. Usar SUM en una tasa (LGD ∈ [0,1]) es semánticamente incorrecto.

## Fix
```sas
MAX(LGD_ESTIMADA) AS LGD_ESTIMADA  /* reemplaza SUM() */
```

## Impacto
LGD_ESTIMADA, MoC, LGD_CON_MOC y ECL pueden exceder el rango regulatorio. RWA sobrestimado.""",
    },
    {
        "id": "bug_varswap_ead_lgd",
        "type": "insight",
        "priority": 0.7,
        "tags": ["bug", "varswap", "LGD", "EAD", "floor", "hipoteca"],
        "fields": ["LGD_ESTIMADA", "EAD", "LGD_FLOOR"],
        "articles": ["CRR_154"],
        "source": "toy_lgd/01_easy_varswap — revisión condición floor hipoteca",
        "feedback": False,
        "body": """# Variable swap: EAD usado en lugar de LGD_ESTIMADA en condición de floor hipotecario

En `proj_03_suelos_lgd.sas` (o equivalente en versiones anteriores) la línea:
```sas
IF COLATERAL_TIPO = 'HIPOTECA' AND EAD < 0.30 THEN LGD_ESTIMADA = 0.30;
```
usa `EAD` (cientos de miles) en lugar de `LGD_ESTIMADA`. Como `EAD < 0.30` siempre es FALSO, el floor del 30% nunca se aplica.

## Causa raíz
Error de copy-paste o confusión de nombres (EAD y LGD son numéricos adyacentes en el esquema). SAS no genera advertencia.

## Fix
```sas
IF COLATERAL_TIPO = 'HIPOTECA' AND LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;
```

## Impacto
Hipotecas con LGD entre 0-30% no reciben floor regulatorio. Infracción CRR Art.154(3).""",
    },
    {
        "id": "bug_offbyone_dpd",
        "type": "insight",
        "priority": 0.7,
        "tags": ["bug", "DPD", "stage", "off-by-one", "IFRS9"],
        "fields": ["DPDS", "STAGE_IFRS9", "STAGE_RECLASIFICADO"],
        "articles": ["IFRS9_B5_5_12"],
        "source": "toy_lgd/03_easy_boundary — revisión backstop DPD",
        "feedback": False,
        "body": """# Off-by-one: `>` en lugar de `>=` en backstop DPD 30

```sas
IF DPDS > 30 AND STAGE_IFRS9 = 1 THEN DO;  /* BUG: debiera ser >= */
    STAGE_RECLASIFICADO = 2;
END;
```

DPDS=30 exactamente NO dispara reclasificación a Stage 2. IFRS 9 B5.5.12 requiere reclasificar a los 30 DPD.

## Causa raíz
Error clásico de fencepost. Probablemente copy-paste de un umbral exclusivo.

## Fix
```sas
IF DPDS >= 30 AND STAGE_IFRS9 = 1 THEN DO;
```

## Impacto
Ciclos con exactamente 30 DPD permanecen incorrectamente en Stage 1. Subestimación de provisiones.""",
    },
    {
        "id": "bug_filter_threshold",
        "type": "insight",
        "priority": 0.6,
        "tags": ["bug", "filter", "provision_period", "threshold"],
        "fields": ["PROVISION_PERIOD_MONTHS"],
        "articles": ["art_12_periodos_dotacion"],
        "source": "toy_lgd/04_medium_filter — revisión filtro WHERE",
        "feedback": False,
        "body": """# WHERE filter con umbral y operador incorrectos

```sas
WHERE PROVISION_PERIOD_MONTHS > 12;  /* BUG: debiera ser >= 9 */
```

## Problemas
1. Umbral incorrecto: 12 en vez de 9 (Circular 6/2016 Art.12 requiere mínimo 9 meses)
2. Operador incorrecto: `>` en vez de `>=` (excluye ciclos con exactamente 9, 10, 11 meses)

Caso real: CIC_004 con 9 meses de provision period fue excluido erróneamente.

## Causa raíz
Posible confusión con una versión anterior de la regulación. Umbral 12 podría venir de requerimiento CORP.

## Fix
```sas
WHERE PROVISION_PERIOD_MONTHS >= 9;
```

## Impacto
Pérdida silenciosa de filas válidas. Ciclos con 9-11 meses desaparecen del pipeline sin alerta.""",
    },
    {
        "id": "bug_wrong_bygroup_moc",
        "type": "insight",
        "priority": 0.6,
        "tags": ["bug", "MoC", "BY-group", "aggregation", "segmento"],
        "fields": ["MoC", "LGD_CON_MOC", "SEGMENTO", "COLATERAL_TIPO"],
        "articles": [],
        "source": "toy_lgd/06_hard_agg_level — revisión segmentación MoC",
        "feedback": False,
        "body": """# BY-group incorrecto: COLATERAL_TIPO en vez de SEGMENTO en cómputo de MoC

El MoC se calcula agrupando por `COLATERAL_TIPO` en lugar de `SEGMENTO`:
```sas
PROC MEANS DATA=...;
    VAR LGD_ESTIMADA;
    BY COLATERAL_TIPO;  /* BUG: debiera ser SEGMENTO */
    OUTPUT OUT=seg_means MEAN=SEG_LGD_MEAN;
RUN;
```

## Por qué es sutil
COLATERAL_TIPO y SEGMENTO están correlacionados (ej: HIPOTECA→MORTGAGE), por lo que la mayoría de filas obtienen el valor correcto por coincidencia. Pero un contrato CORP con colateral FINANCIERO se compara contra la media incorrecta.

## Fix
```sas
BY SEGMENTO;
```

## Impacto
MoC ligeramente incorrecto para combinaciones minoritarias (SEGMENTO×COLATERAL). ECL y RWA ligeramente desviados.""",
    },
    {
        "id": "bug_case_sensitive",
        "type": "insight",
        "priority": 0.6,
        "tags": ["bug", "case-sensitive", "colateral", "UPCASE"],
        "fields": ["COLATERAL_TIPO", "LGD_FLOOR", "LGD_ESTIMADA"],
        "articles": [],
        "source": "toy_lgd/07_harder_type_coerce — revisión comparación strings",
        "feedback": False,
        "body": """# Comparación case-sensitive en COLATERAL_TIPO

```sas
IF COLATERAL_TIPO = 'HIPOTECA' THEN LGD_FLOOR = 0.30;
```

SAS es case-sensitive. Datos fuente tienen casing inconsistente ('HIPOTECA', 'hipoteca', 'Hipoteca') por integración post-merger. Solo la variante mayúscula recibe floor.

## Causa raíz
Asunción de calidad de datos. Integración post-fusión introduce variaciones de casing. SAS no genera advertencia.

## Fix
```sas
IF UPCASE(COLATERAL_TIPO) = 'HIPOTECA' THEN LGD_FLOOR = 0.30;
```
O estandarizar en la capa de entrada.

## Impacto
Algunas hipotecas no reciben floor regulatorio. Inconsistencia silenciosa.""",
    },
    {
        "id": "bug_op_lgd_lookup",
        "type": "insight",
        "priority": 0.6,
        "tags": ["bug", "lookup", "OP_LGD", "cross-reference"],
        "fields": ["OP_LGD", "OP_LGD_FLAG", "OP_LGD_IS_SECURED", "ID_CONTR_CICLO_LGD"],
        "articles": [],
        "source": "toy_lgd/09_op_lgd_lookup — revisión función _LOOKUP",
        "feedback": False,
        "body": """# Wrong lookup key en OP_LGD cross-reference

```sas
/* BUG: pasa ID_CONTR_CICLO_LGD (ID del ciclo actual) en vez de OP_LGD (ID del contrato referenciado) */
_LOOKUP('SECURED_CYCLES', 'ID_CONTRATO', ID_CONTR_CICLO_LGD, 'SECURED');
```

Todo contrato no garantizado con OP_LGD referenciando a otro contrato se marca incorrectamente como problema, incluso cuando el contrato referenciado SÍ está garantizado.

## Causa raíz
Variable incorrecta pasada a la función _LOOKUP. Posible error de refactor (renombrar variable sin actualizar todas las referencias).

## Fix
```sas
_LOOKUP('SECURED_CYCLES', 'ID_CONTRATO', OP_LGD, 'SECURED');
```

## Impacto
Falsos positivos en validación de garantías cruzadas. Ruido en informes de excepciones.""",
    },
    {
        "id": "bug_rating_grado_0",
        "type": "insight",
        "priority": 0.5,
        "tags": ["bug", "rating", "PD", "recalibracion", "RATING_GRADO"],
        "fields": ["RATING_GRADO", "PD_ESTIMADA"],
        "articles": ["circular_4_2022"],
        "source": "tool_calling_v2_dataset tc_025 — revisión recalibración PD",
        "feedback": False,
        "body": """# RATING_GRADO=0 excluido de recalibración PD

```sas
IF RATING_GRADO in (1,2) THEN PD = PD * 1.15;
```

Ciclos con RATING_GRADO=0 (que existen en sistemas fuente) NO pasan por recalibración ×1.15.

## Causa raíz
Condición asume RATING_GRADO siempre ≥1. Código no documenta el caso RATING_GRADO=0 (posiblemente: nuevo sin rating, o rating externo no mapeado).

## Dilema
¿Es RATING_GRADO=0 un valor válido? Si sí, debe recalibrarse. Si no, debe filtrarse o mapearse antes.

## Impacto
PD_ESTIMADA puede estar infravalorada para ciclos con RATING_GRADO=0 que debieran recalibrarse.""",
    },
]

REGULATORY_GAPS = [
    {
        "id": "gap_stage3_pd_forced",
        "type": "insight",
        "priority": 0.6,
        "tags": ["gap", "Stage 3", "PD", "regulatorio", "CRR"],
        "fields": ["PD_ESTIMADA", "STAGE_IFRS9", "ECL"],
        "articles": ["circular_6_2016_art_15", "CRR_154"],
        "source": "eval_dataset eval_049 — revisión Stage 3 PD forced",
        "feedback": False,
        "body": """# Stage 3 PD no forzada a 1.0 en pipeline standalone

CRR requiere `PD_ESTIMADA = 1.0` para exposiciones Stage 3 (deterioradas). El pipeline standalone `proj_03_suelos_lgd.sas` NO implementa:
```sas
IF STAGE_IFRS9 = 3 THEN PD_ESTIMADA = 1.0;
```

La macro `lgd_macros.sas:498-501` SÍ lo hace, pero el script principal no la llama.

## Causa raíz
Inconsistencia entre la librería de macros y el pipeline standalone. El script principal se escribió antes de que existiera la macro.

## Impacto
Stage 3 con PD < 1.0 produce ECL artificialmente baja. Riesgo de infra-provisionamiento regulatorio.""",
    },
    {
        "id": "gap_segment_ventana_check",
        "type": "insight",
        "priority": 0.5,
        "tags": ["gap", "ventana_observacion", "segmento", "EBA"],
        "fields": ["VENTANA_OBSERVACION_YEARS", "NO_CONFORMES", "SEGMENTO"],
        "articles": ["eba_gl_2017_16"],
        "source": "proj_03_suelos_lgd.sas:78-81 — revisión no_conformes",
        "feedback": False,
        "body": """# VENTANA_OBSERVACION check agnóstico a segmento — gap regulatorio

```sas
IF VENTANA_OBSERVACION_YEARS < 5 THEN DO;
    FLAG_NC = 1;
    MOTIVO = 'Ventana observación < 5 años';
END;
```

EBA GL 2017/16 §6.3 requiere:
- CORP/RETAIL: ≥5 años ✓
- HIPOTECA: ≥7 años ✗ (el chequeo no discrimina)

Hipotecas con 5-6 años pasan sin ser marcadas como no-conformes.

## Fix
```sas
IF (SEGMENTO = 'HIPOTECA' AND VENTANA_OBSERVACION_YEARS < 7)
    OR VENTANA_OBSERVACION_YEARS < 5 THEN DO;
    FLAG_NC = 1;
END;
```""",
    },
]

DATA_QUALITY = [
    {
        "id": "dq_ead_missing_not_distinct",
        "type": "insight",
        "priority": 0.5,
        "tags": ["data quality", "EAD", "missing", "diagnóstico"],
        "fields": ["EAD", "ECL", "LGD_ESTIMADA", "PD_ESTIMADA"],
        "articles": [],
        "source": "eval_dataset eval_039 — análisis ECL missing",
        "feedback": False,
        "body": """# EAD missing no se distingue de LGD missing como causa de ECL missing

Cuando `ECL` sale missing, el pipeline siempre reporta "ECL missing (posible campo no informado)" sin distinguir si la causa es:
- EAD missing (problema upstream en source data)
- LGD_ESTIMADA missing (bug SW_FUSION)
- PD_ESTIMADA missing (caso raro)

## Problema
Sin diagnóstico de causa raíz, un analista pierde horas rastreando. El mensaje de error es genérico.

## Mejora propuesta
```sas
IF ECL = . THEN DO;
    IF EAD = . THEN PUT "ECL missing por EAD no informado";
    ELSE IF LGD_ESTIMADA = . THEN PUT "ECL missing por LGD missing (posible fusion)";
    ELSE IF PD_ESTIMADA = . THEN PUT "ECL missing por PD missing";
    ELSE PUT "ECL missing por causa desconocida";
END;
```""",
    },
    {
        "id": "dq_oread_inflado_no_flag",
        "type": "insight",
        "priority": 0.6,
        "tags": ["data quality", "OR_EAD", "inflado", "no_conformes", "flag"],
        "fields": ["OR_EAD", "NO_CONFORMES", "ECL"],
        "articles": [],
        "source": "eval_dataset eval_049 — revisión flags no_conformes",
        "feedback": False,
        "body": """# OR_EAD inflado no tiene flag en tabla no_conformes

El bug de JOIN no único (ID_FUSION_FINAL) infla OR_EAD silenciosamente. La tabla `no_conformes` no tiene ninguna validación que detecte esto.

Ciclos con OR_EAD artificialmente alto pasan sin alerta hasta que un humano revisa ECLs fuera de rango.

## Check propuesto
```sas
IF OR_EAD > (SELECT PERCENTILE(0.99, OR_EAD) FROM mylib.cycles_hist)
    THEN FLAG_NC = 1; MOTIVO = 'OR_EAD potencialmente inflado';
```
O usar ranges esperados por segmento: CORP max ~500k, RETAIL max ~100k, etc.""",
    },
    {
        "id": "dq_negative_lgd_guard",
        "type": "insight",
        "priority": 0.5,
        "tags": ["data quality", "LGD", "negative", "guard", "clamp"],
        "fields": ["LGD_ESTIMADA", "MoC", "ECL", "RWA"],
        "articles": [],
        "source": "eval_dataset eval_045 — revisión rangos LGD",
        "feedback": False,
        "body": """# Sin guardia contra LGD_ESTIMADA negativa

El pipeline no tiene ninguna protección contra valores negativos de LGD:
```sas
/* FALTA: LGD_ESTIMADA = MAX(0, LGD_ESTIMADA); */
```

Si LGD es negativa por error de datos: MoC negativo → LGD_CON_MOC negativo → ECL negativo → RWA negativo.

## Por qué importa
Aunque es raro en producción, un error upstream (ej: cálculo de recoveries mal etiquetado) puede producir LGD negativa. Sin clamp, el error se propaga a todos los downstream.

## Fix
```sas
LGD_ESTIMADA = MAX(0, MIN(1, LGD_ESTIMADA));  /* clamp a [0,1] */
```""",
    },
]

EDGE_CASES = [
    {
        "id": "edge_missing_auto_corrected",
        "type": "quirk",
        "priority": 0.5,
        "tags": ["quirk", "missing", "LGD", "auto-correction", "hipoteca"],
        "fields": ["LGD_ESTIMADA", "LGD_FLOOR", "COLATERAL_TIPO"],
        "articles": [],
        "severity": "medium",
        "source": "eval_dataset eval_023 — comportamiento serendípito SAS",
        "feedback": False,
        "body": """# Missing LGD auto-corregido por floor hipotecario (comportamiento serendípito)

En SAS, missing < cualquier número es TRUE:
```sas
IF LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;  /* también corrige missing! */
```

Cuando LGD_ESTIMADA es missing por el bug SW_FUSION, y COLATERAL_TIPO='HIPOTECA', el floor accidentalmente lo corrige a 0.30.

## Efecto colateral
Esto ENMASCARA el bug SW_FUSION para hipotecas. Un analista que solo revisa hipotecas nunca detecta el problema. Solo contratos NO hipotecarios con SW_FUSION=1 muestran LGD missing.

## Lección
Las correcciones automáticas pueden ocultar bugs río arriba. Todo default de missing debería loguearse explícitamente.""",
    },
    {
        "id": "edge_lgd_realizada_negativa",
        "type": "insight",
        "priority": 0.5,
        "tags": ["edge case", "LGD_REALIZADA", "negativo", "recovery"],
        "fields": ["LGD_REALIZADA", "OR_EAD", "EAD"],
        "articles": [],
        "source": "eval_dataset eval_047, eval_050 — revisión LGD realizada",
        "feedback": False,
        "body": """# LGD_REALIZADA puede ser negativa

```sas
LGD_REALIZADA = 1 - (EAD * 0.4) / OR_EAD;
```
La fórmula puede dar negativo cuando `OR_EAD < EAD * 0.4` (recuperaciones superan exposición). Esto es económicamente posible (ej: apreciación de colateral).

## Problemas
1. Tasa de recuperación fija 0.4 NO validada empíricamente
2. LGD_REALIZADA negativa es correcta pero sorprendente
3. No hay documentación sobre cómo interpretar valores negativos

## Recomendación
Validar la tasa 0.4 contra datos históricos reales. Documentar que LGD_REALIZADA negativa significa recuperaciones netas positivas (ganancia).""",
    },
    {
        "id": "edge_kirb_nonlinearity",
        "type": "insight",
        "priority": 0.4,
        "tags": ["edge case", "K_IRB", "SQRT", "non-linear", "capital"],
        "fields": ["K_IRB", "PD_ESTIMADA", "RWA", "CAPITAL_REGULATORIO"],
        "articles": ["CRR_154"],
        "source": "eval_dataset eval_042 — revisión fórmula IRB",
        "feedback": False,
        "body": """# K_IRB no lineal por SQRT(PD) — comportamiento no obvio

```sas
K_IRB = SQRT(PD) * 0.06 + PD * 0.5;
```
Cuando PD se duplica (por recalibración ×1.15 en ratings 1-2):
- PD=0.01 → K_IRB=0.011
- PD=0.02 → K_IRB=0.0185 (solo +68%, no +100%)

## Por qué es relevante
Un revisor podría pensar que el multiplicador de recalibración ×1.15 en PD produce ×1.15 en K_IRB. No es así por la raíz cuadrada. Esto NO es un bug, pero debe documentarse para evitar falsas alarmas en auditoría.

## Nota
Es una propiedad matemática de la fórmula IRB simplificada. La fórmula completa del CRR añade correlación R(PD) que agrava la no-linealidad.""",
    },
    {
        "id": "edge_v2_v3_version_aware",
        "type": "insight",
        "priority": 0.6,
        "tags": ["version", "V2", "V3", "thresholds", "migration"],
        "fields": ["PD_ESTIMADA", "LGD_FLOOR", "OR_EAD_TIT", "COLATERAL_FIN"],
        "articles": [],
        "source": "v2_to_v3_release_notes.md — análisis diferencias entre versiones",
        "feedback": False,
        "body": """# Diferencias V2 vs V3: thresholds version-sensitive

Múltiples umbrales cambiaron entre V2 y V3. Consultas DQC y validaciones deben ser parameterizadas por versión:

| Parámetro | V2 | V3 |
|---|---|---|
| PD floor | 0.03% | 0.05% |
| CORP LGD floor | 0.45 | 0.50 |
| OR_EAD_TIT | no existe | sí existe |
| FINANCIERO | COLATERAL_TIPO='FINANCIERO' | COLATERAL_FIN=1 |

## Riesgo
Aplicar umbrales de V3 a datos V2 produce falsos positivos (y viceversa). El pipeline debe recibir un parámetro de versión.

## Recomendación
Parametrizar todos los umbrales regulatorios por versión en un archivo de configuración, no hardcodearlos en SAS.""",
    },
    {
        "id": "edge_moc_case_mismatch",
        "type": "quirk",
        "priority": 0.3,
        "tags": ["quirk", "MoC", "case", "mapping", "CSV"],
        "fields": ["MoC", "MOC"],
        "articles": [],
        "severity": "low",
        "source": "audit/mapping.json — revisión naming columnas",
        "feedback": False,
        "body": """# MoC: nombre con mixed case inconsistente entre CSV y SAS

- En CSV de salida: `MoC` (mixed case)
- En código SAS: `MOC` (mayúscula)
- En mapping.json: referenciado como 'MoC' con nota "case mismatch"

## Impacto
Consultas DQC que usan `MOC` no encuentran la columna en el CSV. Queries deben hacer `UPPER(columna)` o el pipeline debe unificar el naming.

## Fix recomendado
Unificar a `MOC` (mayúscula, consistente con el resto de columnas SAS) y añadir alias en la exportación CSV.""",
    },
]

UNSTRUCTURED = [
    {
        "id": "email_business_cure_rate",
        "type": "insight",
        "priority": 0.7,
        "tags": ["email", "business", "cure", "LGD", "hipoteca"],
        "fields": ["LGD_ESTIMADA", "CURE_FLAG"],
        "articles": [],
        "source": "Email de equipo de negocio 2025-03-14 — cure rate CORP",
        "feedback": {
            "type": "correction",
            "original": "Cure rate ×0.95 aplica a todos los segmentos",
            "corrected": "Cure rate ×0.95 aplica SOLO a HIPOTECA. CORP y otros segmentos NO tienen ajuste por cure.",
        },
        "body": """# [EMAIL] Corrección: cure rate aplica solo a HIPOTECA, no a CORP

De: equipo.negocio@banco.es
Para: validacion.regulatoria@banco.es
Asunto: RE: Aplicación del cure rate LGD

Buenas,

Revisando la documentación del cure rate re-fitted 2018-2024, confirmamos que el ajuste ×0.95 se aplica EXCLUSIVAMENTE a contratos con COLATERAL_TIPO = 'HIPOTECA'.

La nota inicial que decía que aplicaba a todos los segmentos era incorrecta. Para CORP, NINGUNA, y otros segmentos no hay ajuste por cure rate en este momento. El floor del 30% sigue aplicando normalmente.

Quedo atenta a cualquier duda.

Saludos,
María García
Equipo de Negocio - Riesgos

## Acción requerida
Corregir insight automático que asumió aplicación global. El changelog 2025-q1-lgd-cure.md especifica HIPOTECA pero la redacción es ambigua.""",
    },
    {
        "id": "excel_dqc_outliers",
        "type": "insight",
        "priority": 0.6,
        "tags": ["excel", "DQC", "outliers", "data quality", "LGD"],
        "fields": ["LGD_ESTIMADA", "OR_EAD", "PD_ESTIMADA", "ECL"],
        "articles": [],
        "source": "Excel DQC mensual Q2-2025 — hoja outliers LGD",
        "feedback": False,
        "body": """# [EXCEL] Extracto DQC mensual — outliers detectados en pipeline LGD

Fuente: DQC_MENSUAL_Q2_2025.xlsx, hoja "OUTLIERS_LGD"

| Ciclo | LGD | OR_EAD | PD | ECL | Flag |
|---|---|---|---|---|---|
| CIC_042 | 1.10 | 250k | 0.02 | 550 | LGD>1 |
| CIC_071 | 0.00 | 180k | 0.01 | 0 | LGD=0 |
| CIC_089 | . | 95k | 0.03 | . | LGD missing |
| CIC_112 | 0.35 | 1.2M | 0.05 | 2100 | OR_EAD 3x mediana |

## Interpretación del agente
- CIC_042 LGD=1.10 → probable bug SUM vs MAX en agregación con fusiones (bug_agg_sum_lgd)
- CIC_071 LGD=0.00 → posible recovery >= exposure, verificar
- CIC_089 LGD missing → posible SW_FUSION=1
- CIC_112 OR_EAD=1.2M → posible JOIN no único inflando OR_EAD

## Patrón
3 de 4 outliers se explican por bugs conocidos del pipeline. Esto sugiere priorizar fixes antes de invertir en data quality upstream.""",
    },
    {
        "id": "meeting_minutes_floor_gap",
        "type": "insight",
        "priority": 0.7,
        "tags": ["meeting", "minutes", "floor", "gap", "CRR"],
        "fields": ["LGD_FLOOR", "LGD_FLOOR_APLICADO", "SEGMENTO", "COLATERAL_TIPO"],
        "articles": ["art_15_dotaciones_minimas", "CRR_154"],
        "source": "Acta reunión comité riesgos 2025-04-02 — floor gaps regulatorios",
        "feedback": {
            "type": "decision",
            "original": "Pendiente de implementar floors Art.15 completos",
            "corrected": "Decisión comité: implementar floors restantes en Q3-2025. Prioridad: CORP+FINANCIERO (0.35) y RETAIL+PERSONAL (0.35) por volumen de exposición.",
        },
        "body": """# [REUNIÓN] Acta comité de riesgos — gaps regulatorios LGD floors

Fecha: 2025-04-02
Asistentes: Dirección Riesgos, Validación, TI, Negocio

## Punto del día: Floors LGD Art.15 no implementados

Se presentó el análisis de gaps: de 6 combinaciones (SEGMENTO×COLATERAL) del Art.15, el pipeline solo implementa 2.

### Decisión
1. Implementar floors restantes en Q3-2025
2. Prioridad: CORP+FINANCIERO (0.35) y RETAIL+PERSONAL (0.35) por mayor volumen de exposición
3. RETAIL+NINGUNA (0.45) y MORTGAGE+NINGUNA (0.40) quedan para Q4-2025
4. Mientras tanto, los 4 gaps quedan documentados como excepción temporal aprobada por comité

### Acciones
- TI: estimar esfuerzo de implementación (2 semanas)
- Validación: actualizar pruebas para cubrir los 4 nuevos floors
- Riesgos: calcular impacto en ECL/RWA de implementar floors faltantes

## Relevancia para el agente
Esta decisión de comité es información vinculante. Si el agente encuentra un gap de floor, debe consultar si está cubierto por una excepción aprobada.""",
    },
    {
        "id": "email_auditor_pd_floor",
        "type": "insight",
        "priority": 0.8,
        "tags": ["email", "auditor", "PD", "floor", "retail", "circular"],
        "fields": ["PD_ESTIMADA"],
        "articles": ["circular_4_2022"],
        "source": "Email de auditoría externa 2025-05-20 — PD floor retail",
        "feedback": {
            "type": "finding",
            "original": "PD floor es 0.05% uniforme para todos los segmentos",
            "corrected": "Hallazgo de auditoría: PD floor retail debe ser 0.03% (Circular 4/2022), no 0.05%. Pipeline usa 0.05% incorrecto para retail.",
        },
        "body": """# [EMAIL] Hallazgo auditoría externa — PD floor retail incorrecto

De: auditor.externo@big4.com
Para: cumply.regulatorio@banco.es
CC: direccion.riesgos@banco.es
Asunto: HALLAZGO - PD floor retail Circular 4/2022

Estimados,

Como parte de la revisión independiente de cumplimiento regulatorio IRB correspondiente al ejercicio 2025, hemos identificado el siguiente hallazgo:

## Hallazgo AUD-2025-047: PD floor para retail no conforme con Circular 4/2022

**Evidencia:** En el pipeline SAS LGD, línea 42 de proj_03_suelos_lgd.sas:
```sas
IF PD_ESTIMADA < 0.0005 THEN PD_ESTIMADA = 0.0005;
```

**Requisito:** Circular 4/2022, modificación del Artículo X, establece:
- PD floor retail (PERSONAL, REVOLVING): **0.03% (0.0003)**
- PD floor corporate/sovereign: **0.05% (0.0005)**

El pipeline aplica 0.05% a TODOS los segmentos, sobrestimando PD_ESTIMADA para retail en 67%.

**Impacto:** ECL para cartera retail sobrestimado en ~0.02% × EAD. Para una cartera retail de 500M€, el exceso de provisiones es ~100k€.

**Severidad:** Media

**Plazo de corrección:** Próximo ciclo de reporting (Q3-2025)

Atentamente,
Equipo de Auditoría Regulatoria

## Acción
Corregir pipeline para usar 0.0003 en retail. Documentar como feedback.""",
    },
    {
        "id": "excel_recovery_rates_raw",
        "type": "insight",
        "priority": 0.6,
        "tags": ["excel", "recovery", "LGD", "tasa", "empírico"],
        "fields": ["LGD_REALIZADA", "OR_EAD", "EAD", "SEGMENTO"],
        "articles": [],
        "source": "Excel recuperaciones 2024 — recovery rates observados vs 0.4 fijo",
        "feedback": False,
        "body": """# [EXCEL] Recovery rates observados vs tasa fija 0.4 — análisis empírico

Fuente: RECUPERACIONES_2024.xlsx, hoja "RECOVERY_BY_SEGMENT"

Se comparó la tasa de recuperación real observada en 2024 contra la tasa fija 0.4 usada en el pipeline para LGD_REALIZADA.

| Segmento | N | Recovery medio | Recovery mediana | Desv | vs 0.4 |
|---|---|---|---|---|---|
| CORP | 1,247 | 0.38 | 0.35 | 0.22 | -0.02 |
| HIPOTECA | 3,891 | 0.52 | 0.48 | 0.31 | +0.12 |
| RETAIL | 2,563 | 0.29 | 0.25 | 0.18 | -0.11 |
| TOTAL | 7,701 | 0.43 | 0.40 | 0.28 | +0.03 |

## Conclusiones
1. La media global (0.43) está cerca de 0.4, pero hay gran dispersión por segmento
2. HIPOTECA tiene recovery significativamente mayor (+0.12 vs asunción)
3. RETAIL tiene recovery menor (-0.11 vs asunción)
4. Usar recovery fijo 0.4 infraestima LGD_REALIZADA para hipotecas y la sobrestima para retail

## Recomendación
Parametrizar recovery rate por segmento. Mínimo, usar mediana en vez de media para reducir impacto de outliers.""",
    },
]

ALL = BUGS + REGULATORY_GAPS + DATA_QUALITY + EDGE_CASES + UNSTRUCTURED


def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing training examples to append to
    existing = []
    if TRAINING_FILE.exists():
        with open(TRAINING_FILE) as f:
            existing = [json.loads(line) for line in f if line.strip()]

    new_training = existing[:]

    for exp in ALL:
        # Write .md file
        path = EXP_DIR / f"{exp['id']}.md"
        path.write_text(_md(exp), encoding="utf-8")
        print(f"  ✓ {path.name}")

        # Generate training example
        train_ex = _sleep_example(exp)
        new_training.append(train_ex)

    # Write updated training file
    with open(TRAINING_FILE, "w", encoding="utf-8") as f:
        for ex in new_training:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n{len(ALL)} new experiences created")
    print(f"Phase 2.1 training: {len(new_training)} total examples ({len(new_training) - len(existing)} new)")


if __name__ == "__main__":
    main()

"""Generate more real-case experiences — batch 2.

Focus: cross-system, performance, model validation, treasury,
regulatory communications, system alerts, and operational scenarios.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "data" / "experience"

# ── Helpers ─────────────────────────────────────────────────────────────────

def _md(exp: dict) -> str:
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


def _sleep_example(exp: dict) -> dict:
    tags_str = ", ".join(exp.get("tags", []))
    fields_str = ", ".join(exp.get("fields", []))
    body_preview = exp["body"][:500].strip()
    user_msg = f"""## Sesión del día — datos de producción

Durante la operativa se detectó el siguiente caso:

{body_preview}

### Contexto adicional
- Campos afectados: {fields_str}
- Tags: {tags_str}
- Fuente: {exp.get('source', 'producción')}

### Tarea
Procesa esta información durante el ciclo de sueño. Decide si merece persistirse."""
    exp_type = exp.get("type", "insight")
    if exp_type == "quirk":
        action = f"save_quirk(id={exp['id']}, severity={exp.get('severity', 'medium')})"
    elif exp.get("feedback"):
        action = "save_feedback(correction)"
    else:
        action = f"save_insight(id={exp['id']}, priority={exp.get('priority', 0.5)}, tags={exp.get('tags', [])})"
    assistant_msg = f"""## Decisión: GUARDAR

{action}

Prioridad {exp.get('priority', 0.5)}: {'crítico' if exp.get('priority', 0.5) >= 0.7 else 'importante' if exp.get('priority', 0.5) >= 0.5 else 'informativo'}"""
    return {
        "messages": [
            {"role": "system", "content": "Eres RegLLM en MODO SUEÑO. Procesas hallazgos del día y decides qué merece persistirse."},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "id": exp["id"],
        "phase": "2.1",
        "difficulty": "medium",
    }


# ── Experiences batch 2 ─────────────────────────────────────────────────────

CROSS_SYSTEM = [
    {
        "id": "xs_sap_sas_ead_mismatch",
        "type": "insight",
        "priority": 0.7,
        "tags": ["cross-system", "SAP", "SAS", "EAD", "reconciliation"],
        "fields": ["EAD", "EAD_TOTAL", "OR_EAD"],
        "articles": [],
        "source": "Concilación mensual SAP→SAS Q2-2025 — diferencias EAD",
        "feedback": False,
        "body": """# Diferencias sistemáticas EAD entre SAP (origen) y SAS pipeline

La conciliación mensual EAD entre SAP (sistema origen) y la tabla `mylib.ciclos_recuperacion` en SAS muestra una diferencia sistemática de ~2.3%:

| Mes | SAP SUM(EAD) | SAS SUM(EAD) | Dif | % |
|---|---|---|---|---|
| Ene-25 | 1,234M | 1,206M | -28M | -2.3% |
| Feb-25 | 1,241M | 1,213M | -28M | -2.3% |
| Mar-25 | 1,228M | 1,200M | -28M | -2.3% |

## Causa raíz
El extracto SAP exporta contratos con `STATUS ≠ 'CANCELED'`, pero el job de importación SAS filtra adicionalmente `WHERE FECHA_ALTA >= '2020-01-01'`. Contratos anteriores a 2020 (aprox. 28M€ en EAD) se pierden silenciosamente.

## Impacto
ECL y RWA infraestimados en ~2.3%. El gap es constante pero no se había detectado porque ambas cifras se movían juntas.

## Fix
Eliminar el filtro de fecha o justificarlo con documentación. Si es intencional, documentar la exclusión.""",
    },
    {
        "id": "xs_sas_hadoop_encoding",
        "type": "quirk",
        "priority": 0.5,
        "tags": ["cross-system", "encoding", "Hadoop", "SAS", "UTF-8"],
        "fields": ["COLATERAL_TIPO", "SEGMENTO"],
        "articles": [],
        "severity": "medium",
        "source": "Incidente integración SAS→Hadoop — caracteres especiales",
        "feedback": False,
        "body": """# Problema de encoding SAS→Hadoop: caracteres latinos corrompidos

El pipeline exporta particiones a Hadoop en LATIN1, pero el consumidor Hadoop espera UTF-8. Caracteres como 'ó', 'é', 'ñ', 'í' en campos como `SEGMENTO` ('Hipoteca' vs 'HipotÃ©ca') causaban fallos en JOINs con tablas Hadoop.

## Síntoma
```sql
SELECT * FROM contracts_hadoop c
JOIN sas_export s ON c.contrato_id = s.ID_CONTRATO
WHERE c.segmento = 'CORP'  -- funciona
  AND c.segmento LIKE '%Ã©%' -- hay filas con encoding corrupto
```

## Causa raíz
Exportación SAS usa `ENCODING=LATIN1` por defecto. Hadoop asume UTF-8. Los caracteres >0x7F se corrompen.

## Fix
```sas
FILENAME exportado PIPE 'iconv -f LATIN1 -t UTF-8' ENCODING=UTF-8;
```
O configurar `%LET SAS_ENCODING=UTF-8;` antes de la exportación.

## Lección
Problemas de encoding son silenciosos — no generan error, solo datos incorrectos. Cualquier interfaz entre sistemas con distinto encoding debe validarse con un contrato específico conocido (ej: 'VALLE NORTE S.A.' con ñ).""",
    },
    {
        "id": "xs_sigf_taxonomia_mismatch",
        "type": "insight",
        "priority": 0.6,
        "tags": ["cross-system", "SIGF", "taxonomía", "COREP", "FINREP"],
        "fields": ["SEGMENTO", "COLATERAL_TIPO"],
        "articles": ["EBA_ITS_2024_02"],
        "source": "Validación COREP Q2-2025 — mapeo taxonomía SIGF vs pipeline",
        "feedback": False,
        "body": """# Mapeo taxonomía SIGF vs pipeline: diferencias en clasificación COREP

La validación de envío COREP Q2-2025 reveló diferencias en la clasificación de exposiciones IRB entre la taxonomía SIGF (oficial) y el pipeline SAS:

| Categoría | SIGF | Pipeline | Diferencia |
|---|---|---|---|
| CORP→SME | 847M | 798M | -49M (empresas >250 empleados mal clasificadas) |
| RETAIL→OTHER | 234M | 289M | +55M (CLUDs pequeños clasificados como RETAIL) |
| MORTGAGE | 1,234M | 1,234M | 0 (correcto) |

## Causa raíz
El pipeline clasifica SME usando una regla simplificada (importe < 1M€), mientras que SIGF usa la definición completa de PYME de la Comisión Europea (<250 empleados, <50M€ facturación, <43M€ balance).

## Impacto
COREP enviado con clasificación incorrecta en 2 de 6 categorías. Riesgo de requerimiento de aclaración por parte del BdE.

## Fix
Usar la tabla de empleados/facturación de SAP para clasificar SME según definición UE, no por importe.""",
    },
]

PERFORMANCE = [
    {
        "id": "perf_sas_macro_overhead",
        "type": "insight",
        "priority": 0.4,
        "tags": ["performance", "SAS", "macro", "optimization"],
        "fields": [],
        "articles": [],
        "source": "Optimización pipeline SAS — runtime analysis Q1-2025",
        "feedback": False,
        "body": """# Macro LGD_DQ_CALL llamada 47 veces por ciclo — overhead 12min en ejecución mensual

Análisis de logging reveló que la macro `%LGD_DQ_CALL` se ejecuta 47 veces por ciclo de validación (una por cada check DQC). Cada llamada abre/cierra la tabla `mylib.cycles_check`:

```
NOTE: Table MYLIB.CYCLES_CHECK opened at line 123.
NOTE: Table MYLIB.CYCLES_CHECK closed at line 125.
... (repetido 47 veces por ciclo, × ~50k ciclos = 2.35M aperturas/cierres)
```

## Impacto
- Tiempo de ejecución: ~18 min → ~6 min después de optimizar
- 12 minutos/mes perdidos en overhead I/O
- Proporcional al número de ciclos (empeora con crecimiento de cartera)

## Fix
```sas
/* Antes: 47 llamadas separadas */
%LGD_DQ_CALL(CHECK=DQ01);
%LGD_DQ_CALL(CHECK=DQ02);
/* ... 45 más ... */

/* Después: 1 llamada batch */
%LGD_DQ_BATCH();
```
La macro batch ejecuta todos los checks en una sola pasada de la tabla.

## Lección
El overhead de I/O en SAS es significativo para procesos batch con muchas tablas pequeñas. Siempre agrupar operaciones sobre la misma tabla.""",
    },
    {
        "id": "perf_runaway_query",
        "type": "insight",
        "priority": 0.5,
        "tags": ["performance", "query", "runaway", "SAS", "memory"],
        "fields": ["OR_EAD", "LGD_ESTIMADA"],
        "articles": [],
        "source": "Incidente SAS OOM — query runaway en agregación mensual",
        "feedback": False,
        "body": """# Query runaway: CROSS JOIN implícito en agregación mensual causa OOM

En la ejecución de Mar-2025, el proceso SAS se detuvo con `ERROR: Out of memory` tras 47 minutos. La causa era una agregación que hacía un CROSS JOIN implícito:

```sas
/* Query problemática */
PROC SQL;
    CREATE TABLE agg AS
    SELECT a.ID_CICLO, a.LGD_ESTIMADA, b.OR_EAD
    FROM cycles a, cycles_hist b
    WHERE a.SEGMENTO = b.SEGMENTO;  /* 50k × 500k = 25B filas implícitas */
QUIT;
```

## Causa raíz
El desarrollador asumió que `SEGMENTO` era clave única en `cycles_hist`. No lo es. La cardinalidad real:
- `cycles`: 50k filas, 4 valores de SEGMENTO
- `cycles_hist`: 500k filas, 6 valores de SEGMENTO (incluye históricos)

50k × 500k / 4 = 6.25M filas (no 25B) por la condición... pero aun así 6.25M en un solo paso SQL es pesado.

## Fix
Usar `PROC MEANS` con CLASS SEGMENTO en vez de CROSS JOIN, o hacer un merge explícito con BY SEGMENTO después de resumir.

## Lección
Siempre verificar cardinalidad antes de JOINs en SQL. En SAS, los merges implícitos no dan warning hasta que es demasiado tarde.""",
    },
    {
        "id": "perf_partition_skew",
        "type": "insight",
        "priority": 0.4,
        "tags": ["performance", "partition", "skew", "Hadoop", "Spark"],
        "fields": [],
        "articles": [],
        "source": "Análisis rendimiento Spark SQL — partition skew en TABLA_CICLOS",
        "feedback": False,
        "body": """# Partition skew en tabla Hadoop por SEGMENTO_CALIBRACION

El job Spark que consume la salida del pipeline SAS para cálculos de RWA mostraba tiempos de ejecución muy variables (12-47 min). Análisis del plan de ejecución reveló partition skew severo:

| Partición | SEGMENTO_CALIBRACION | Filas | Tiempo |
|---|---|---|---|
| p1 | HIPOTECA | 3,891,234 | 47 min |
| p2 | CORP | 1,247,891 | 15 min |
| p3 | RETAIL | 2,563,456 | 28 min |
| p4 | OTROS | 12,345 | 2 min |

## Causa raíz
La tabla se particiona por `SEGMENTO_CALIBRACION`, pero HIPOTECA tiene 300× más filas que OTROS. Una partición domina el tiempo total.

## Fix
Reparticionar por una clave más granular (ej: `SEGMENTO_CALIBRACION || MONTH(FECHA_DEV)`) o usar bucketing con número de buckets proporcional al tamaño esperado.

## Lección
Particionar por columnas con alta skew (distribución no uniforme) causa problemas de rendimiento. Usar múltiples columnas o bucketing.""",
    },
]

MODEL_VALIDATION = [
    {
        "id": "mv_lgd_calibration_drift",
        "type": "insight",
        "priority": 0.7,
        "tags": ["model validation", "LGD", "calibration", "drift", "backtesting"],
        "fields": ["LGD_ESTIMADA", "LGD_REALIZADA"],
        "articles": ["eba_gl_2017_16"],
        "source": "Backtesting anual LGD — calibración vs realizado 2024",
        "feedback": False,
        "body": """# Backtesting LGD 2024: calibración sobreestima LGD en CORP y subestima en HIPOTECA

Resultados del backtesting anual de calibración LGD (EBA GL 2017/16 §6.4):

| Segmento | LGD estimada media | LGD realizada media | Sesgo | ¿Significativo? |
|---|---|---|---|---|
| CORP | 0.52 | 0.41 | +0.11 | SÍ (p<0.01) |
| HIPOTECA | 0.28 | 0.35 | -0.07 | SÍ (p<0.05) |
| RETAIL | 0.38 | 0.36 | +0.02 | NO (p=0.32) |

## Interpretación
- **CORP**: La calibración sobreestima LGD en 11 puntos. Posible causa: la tasa de recuperación fija 0.4 es demasiado baja para CORP (recovery real ~0.52), lo que infla LGD_REALIZADA de referencia → calibración conservadora.
- **HIPOTECA**: Subestimación de 7 puntos. La tasa de recuperación 0.4 infraestima el recovery real hipotecario (~0.48).
- **RETAIL**: Calibración adecuada.

## Acción
Revisar la tasa de recuperación fija 0.4. Separar por segmento mejoraría la calibración. Impacto en ECL: sobrestimado en CORP (provisiones excesivas) e infraestimado en HIPOTECA.

## Relevancia
EBA GL 2017/16 §6.4 requiere backtesting anual y corrección de sesgos significativos.""",
    },
    {
        "id": "mv_pd_rating_migration",
        "type": "insight",
        "priority": 0.6,
        "tags": ["model validation", "PD", "rating", "migration", "matriz"],
        "fields": ["RATING_GRADO", "PD_ESTIMADA"],
        "articles": ["eba_gl_2017_16"],
        "source": "Matriz de migración rating anual 2024-2025",
        "feedback": False,
        "body": """# Matriz de migración rating: estabilidad anormal en grado 5

La matriz de migración 2024→2025 muestra un patrón inusual en RATING_GRADO=5:

| Rating 2024 → Rating 2025 | 1 | 2 | 3 | 4 | 5 | 6 | Default |
|---|---|---|---|---|---|---|---|
| Grado 1 | 82% | 12% | 4% | 1% | 1% | 0% | 0% |
| Grado 5 | 0% | 0% | 2% | 3% | **90%** | 3% | 2% |

El grado 5 tiene un 90% de persistencia (vs ~70-75% para otros grados). Esto sugiere que el grado 5 actúa como "cajón de sastre" para contratos que el modelo no sabe clasificar.

## Causa posible
El grado 5 en la escala IRB es el grado medio-bajo (watchlist). Los analistas tienden a asignar grado 5 por defecto cuando no hay información suficiente para discriminar, en vez de usar los grados 4 o 6.

## Impacto
- PD_ESTIMADA para grado 5 no refleja el riesgo real (mezcla de perfiles)
- La discriminación entre grados 4, 5, 6 es baja
- El modelo PD subestima el riesgo de contratos que debieran estar en grado 6

## Recomendación
Revisar las guías de asignación de rating para grado 5. Implementar un check de concentración: si RATING_GRADO=5 supera el 25% de la cartera, alertar.""",
    },
    {
        "id": "mv_sharpe_ratio_validation",
        "type": "insight",
        "priority": 0.5,
        "tags": ["model validation", "Sharpe", "discriminación", "PD"],
        "fields": ["PD_ESTIMADA", "RATING_GRADO"],
        "articles": ["eba_gl_2017_16"],
        "source": "Validación anual PD — estadístico Sharpe y poder discriminatorio",
        "feedback": False,
        "body": """# Sharpe Ratio del modelo PD: 0.52 — por debajo del umbral mínimo (0.60)

El estadístico de Sharpe (poder discriminatorio del modelo PD) se calculó en 0.52 para 2024, frente al mínimo regulatorio de 0.60 (EBA GL 2017/16 §5.5).

## Detalle por segmento

| Segmento | Sharpe 2023 | Sharpe 2024 | Cambio | ¿OK? |
|---|---|---|---|---|
| CORP | 0.61 | 0.58 | -0.03 | NO |
| HIPOTECA | 0.65 | 0.63 | -0.02 | SÍ |
| RETAIL | 0.58 | 0.49 | -0.09 | NO |

## Causa probable
RETAIL muestra la mayor caída. Posible causa: el modelo PD no captura el deterioro del segmento de consumo (tipos altos, inflación). Contratos con buen rating histórico empiezan a impagarse sin que el modelo lo anticipe.

## Acciones
- Recalibrar modelo PD retail con datos post-inflación
- Añadir variable macro (tipo de interés, IPC) como predictor
- Informar a BdE con plan de remediación en 6 meses.

## Nota
Este es un hallazgo de validación, no un bug del pipeline. Pero el agente debe conocerlo para contextualizar consultas sobre calidad del modelo.""",
    },
]

REGULATORY_COMMS = [
    {
        "id": "com_bde_requerimiento_154",
        "type": "insight",
        "priority": 0.8,
        "tags": ["BdE", "requerimiento", "LGD", "floor", "COREP"],
        "fields": ["LGD_FLOOR", "LGD_FLOOR_APLICADO", "SEGMENTO", "COLATERAL_TIPO"],
        "articles": ["circular_6_2016_art_15", "CRR_154"],
        "source": "Comunicación BdE 2025-06-15 — requerimiento información LGD floors",
        "feedback": False,
        "body": """# [BdE] Requerimiento de información: LGD floors Art.15 Circular 6/2016

De: supervision@bde.es
Para: cumplimiento.normativo@banco.es
Asunto: Requerimiento información — Aplicación LGD floors IRB

N/Ref: SGR-2025-0042

Estimados,

En el marco de la supervisión continuada de modelos IRB, y de acuerdo con el artículo 15 de la Circular 6/2016, se requiere:

1. **Detalle de LGD_FLOOR aplicado por segmento y tipo de colateral** para todas las exposiciones IRB a cierre de cada mes del ejercicio 2025.
2. **Justificación de desviaciones** respecto a los mínimos del Art. 15 para aquellas combinaciones (SEGMENTO, COLATERAL_TIPO) donde el floor aplicado sea inferior al regulatorio.
3. **Impacto en ECL y RWA** de no aplicar los floors completos en las combinaciones no implementadas.

Plazo: 30 días hábiles.

Atentamente,
Dirección General de Supervisión
Banco de España

## Relevancia
Este requerimiento confirma que el BdE está monitorizando los gaps de floor Art.15 (B2 del análisis). La información requerida (punto 2) apunta directamente a las 4 combinaciones no implementadas.

## Acción
Priorizar implementación de floors faltantes. Preparar respuesta con impacto cuantificado.""",
    },
    {
        "id": "com_bde_carta_circular_4_2022",
        "type": "insight",
        "priority": 0.7,
        "tags": ["BdE", "circular", "PD", "floor", "retail"],
        "fields": ["PD_ESTIMADA"],
        "articles": ["circular_4_2022"],
        "source": "Boletín Oficial del Estado — Circular 4/2022 de 28 de diciembre",
        "feedback": False,
        "body": """# [BOE] Circular 4/2022 — modificación PD floors retail

Disposición publicada en BOE núm. 312, de 29 de diciembre de 2022.

## Extracto relevante (Artículo Único, apartado 3)

"Se modifica el artículo 15 de la Circular 6/2016, que queda redactado como sigue:

Artículo 15. Suelos mínimos.
[...]
3. El suelo de probabilidad de incumplimiento (PD) para exposiciones minoristas será del 0,03 por ciento. Para el resto de exposiciones, el suelo será del 0,05 por ciento."

## Implicaciones
- **Antes de Circular 4/2022**: PD floor ERA 0.05% para todos los segmentos
- **Después**: PD floor retail = 0.03%, CORP/soberano/otros = 0.05%
- **Pipeline SAS** actualmente usa 0.05% uniforme → INCORRECTO para retail

## Estado
Este hallazgo fue identificado por auditoría externa (ver email_auditor_pd_floor). La circular entró en vigor el 1 de enero de 2023. El pipeline lleva 2.5 años con el valor incorrecto.

## Prioridad
ALTA — corrección regulatoria mandatory, no opcional.""",
    },
    {
        "id": "com_bde_inspeccion_calibracion",
        "type": "insight",
        "priority": 0.7,
        "tags": ["BdE", "inspección", "calibración", "LGD", "modelos"],
        "fields": ["LGD_ESTIMADA", "LGD_REALIZADA", "MoC"],
        "articles": ["eba_gl_2017_16"],
        "source": "Carta inspección BdE sobre modelos — conclusiones preliminares",
        "feedback": {
            "type": "finding",
            "original": "Modelo LGD calibrado según guía interna del banco",
            "corrected": "BdE requiere recalibración LGD con ventana mínima 7 años (vs 5 actual) y MoC documentado por fuente de incertidumbre (categorías A/B/C).",
        },
        "body": """# [BdE] Carta de inspección in situ — conclusiones preliminares modelos IRB

De: supervision.inspeccion@bde.es
Para: direccion.riesgos@banco.es
Asunto: Acta de inspección — Modelos IRB LGD/PD (ref. INS-2025-0089)

Durante la inspección in situ realizada del 10 al 28 de marzo de 2025, se han identificado las siguientes deficiencias preliminares:

## Deficiencia 1: Ventana de calibración LGD insuficiente
EBA GL 2017/16 §6.3 requiere mínimo **7 años** para calibración de LGD. El pipeline utiliza **5 años** (parámetro `min_calib_years=5` en `lgd_macros.sas:60`).

## Deficiencia 2: MoC no desglosado por categoría EBA
EBA GL 2017/16 §4.4 requiere que el MoC se desglose en:
- Categoría A: errores de estimación conocidos pero no cuantificables
- Categoría B: fuentes de incertidumbre potencial
- Categoría C: error general de estimación

El pipeline aplica MoC = 5% fijo sin desglose.

## Deficiencia 3: Backtesting sin segmentación
El backtesting LGD se realiza a nivel agregado. EBA GL 2017/16 §6.4 requiere backtesting por segmento de calibración.

## Plazo de respuesta
30 días hábiles para presentar plan de remediación.

## Acción
Las 3 deficiencias requieren cambios en el pipeline SAS. El agente debe conocerlas para contextualizar consultas sobre calibración LGD.""",
    },
]

SYSTEM_ALERTS = [
    {
        "id": "alert_disk_full_lgd_partition",
        "type": "insight",
        "priority": 0.5,
        "tags": ["alert", "disk", "SAS", "storage", "partición"],
        "fields": [],
        "articles": [],
        "source": "Alerta Nagios SAS SERVER — /data/lgd 94% lleno (2025-04-12)",
        "feedback": False,
        "body": """# [ALERTA] Disco /data/lgd al 94% — riesgo de fallo en ejecución mensual

Fuente: Nagios
Host: sas-server-02
Alerta: DISK_CRITICAL - /data/lgd 94% usado (4.2TB/4.5TB)

## Causa raíz
Las tablas intermedias del pipeline LGD no se limpian tras cada ejecución mensual. En particular:
- `mylib.cycles_check` (check DQC): 47 particiones × 3 meses retenidos = 141 tablas
- `mylib.lgd_scenarios`: 12 escenarios × 3 meses = 36 tablas
- Logs SAS: ~200MB/día, rotación a 90 días

## Impacto potencial
Si el disco llega al 100%, la ejecución mensual falla y no se genera COREP/FINREP. SLA de reporting: T+5 días hábiles.

## Fix inmediato
```bash
# Limpiar tablas intermedias con más de 60 días
find /data/lgd/sasdata -name 'cycles_check_*' -mtime +60 -exec rm {} \;
find /data/lgd/sasdata -name 'lgd_scenarios_*' -mtime +60 -exec rm {} \;
```

## Fix permanente
Añadir cleanup al final del pipeline:
```sas
PROC DATASETS LIBRARY=mylib NOLIST;
    DELETE cycles_check_: / MAXAGE=60;
    DELETE lgd_scenarios_: / MAXAGE=60;
RUN;
```

## Lección
Los pipelines batch siempre deben incluir cleanup de tablas intermedias. El espacio es barato hasta que se acaba.""",
    },
    {
        "id": "alert_sas_session_limit",
        "type": "insight",
        "priority": 0.5,
        "tags": ["alert", "SAS", "session", "concurrency", "batch"],
        "fields": [],
        "articles": [],
        "source": "Alerta SAS Workspace Server — límite sesiones concurrentes alcanzado",
        "feedback": False,
        "body": """# [ALERTA] SAS Workspace Server — límite de 50 sesiones alcanzado

Fuente: SAS Management Console
Servidor: SASApp — Workspace Server
Evento: "Maximum number of workspace server sessions (50) reached"

## Causa
El nuevo proceso de validación DQC interactivo lanza una sesión SAS independiente por cada validación. Los usuarios (equipo de negocio) lanzan validaciones desde la herramienta web que no cierran las sesiones al terminar. Sesiones huérfanas se acumulan.

## Impacto
A las 16:30 se alcanzó el límite de 50 sesiones. La ejecución batch del pipeline LGD (que arranca a las 18:00) encontró 0 sesiones disponibles y falló. El reporting mensual se retrasó 1 día.

## Fix inmediato
- Aumentar límite a 100 sesiones temporalmente
- Matar sesiones inactivas >30 min:
  ```bash
  for pid in $(ls -1 /opt/sas/config/Lev1/SASApp/WorkspaceServer/logs/*.log | xargs grep -l 'inactive for 30 minutes' | sed 's/.*_//;s/\.log//'); do kill -9 $pid; done
  ```

## Fix permanente
- Implementar timeout de sesión en Workspace Server (actualmente no tiene)
- La herramienta web debe cerrar sesiones SAS explícitamente vía API

## Lección
Las sesiones SAS son un recurso finito compartido entre batch e interactivo. Monitorear uso y establecer prioridades.""",
    },
    {
        "id": "alert_data_freshness_lag",
        "type": "insight",
        "priority": 0.6,
        "tags": ["alert", "data", "freshness", "SAP", "SAS", "lag"],
        "fields": ["EAD", "FECHA_DEV"],
        "articles": [],
        "source": "Alerta Data Freshness — extracto SAP desactualizado (2025-05-20)",
        "feedback": False,
        "body": """# [ALERTA] Extracto SAP desactualizado: +5 días sin refresco

Fuente: Data Freshness Monitor
Dataset: SAP.EXTRACT_LGD
Última actualización: 2025-05-15 03:00
Alerta: "Data freshness threshold exceeded: 5 days without refresh (threshold: 2 days)"

## Causa
El job de extracción SAP (ABAP program ZRISK_LGD_EXTRACT) falló silenciosamente el 2025-05-16. El error se registró en el log SAP pero no generó alerta al equipo de operaciones. SAS consumió datos del 15-May durante 5 días.

## Impacto
- Ejecuciones SAS entre 16-May y 20-May usaron datos de EAD con 1-5 días de desfase
- Para carteras estables el impacto es menor (~0.1% variación EAD)
- Pero si hubiera una operación corporativa grande en esos días, el impacto sería significativo

## Fix
1. Añadir webhook de error desde SAP ABAP al Data Freshness Monitor
2. En SAS, añadir check de frescura al inicio:
   ```sas
   %LET MAX_AGE_DAYS = 2;
   PROC SQL;
       SELECT MAX(FECHA_DEV) INTO :LAST_EXTRACT FROM mylib.ciclos_recuperacion;
       %IF %SYSEVALF(%SYSFUNC(INTCK(DAY, &LAST_EXTRACT, %SYSFUNC(TODAY())))) > &MAX_AGE_DAYS %THEN %DO;
           %PUT ERROR: Datos desactualizados (&LAST_EXTRACT). Pipeline detenido.;
           ABORT CANCEL;
       %END;
   QUIT;
   ```

## Lección
Los datos desactualizados son más peligrosos que los datos incorrectos: los incorrectos se detectan, los desactualizados no.""",
    },
]

OPERATIONAL = [
    {
        "id": "ops_miss_run_manual_fix",
        "type": "insight",
        "priority": 0.6,
        "tags": ["operational", "run", "manual", "fix", "pipeline"],
        "fields": [],
        "articles": [],
        "source": "Incidente PROD-2025-031 — ejecución manual fuera de secuencia",
        "feedback": False,
        "body": """# Ejecución manual de proj_02 fuera de secuencia causa datos inconsistentes

## Incidente
El 2025-03-15, el equipo de operaciones ejecutó manualmente `proj_02_enriquecimiento_ead.sas` para corregir un error de EAD sin volver a ejecutar `proj_01_carga_datos.sas` primero. El resultado fue que `proj_03_suelos_lgd.sas` encontró datos inconsistentes:
- Ciclos con EAD actualizada pero LGD_ESTIMADA de la ejecución anterior
- Discrepancias en tablas intermedias que asumían consistencia transaccional

## Causa raíz
El pipeline no tiene un mecanismo de versionado o transaccionalidad entre pasos. Cada script SAS asume que los datos de entrada están actualizados.

## Impacto
El reporting mensual se retrasó 3 días mientras se identificaban y corregían las discrepancias. Se ejecutó una rerun completa del pipeline.

## Lecciones
1. El pipeline SAS no es transaccional — las ejecuciones parciales corrompen datos
2. Si se ejecuta un paso manualmente, deben ejecutarse TODOS los pasos posteriores
3. Implementar un check de versión al inicio de cada paso:
   ```sas
   /* Verificar que el paso anterior se ejecutó */
   PROC SQL;
       SELECT MAX(FECHA_EJECUCION) INTO :LAST_RUN FROM mylib.pipeline_log
       WHERE PASO = 'proj_02' AND ESTADO = 'OK';
       SELECT MAX(FECHA_EJECUCION) INTO :CURRENT_RUN FROM mylib.pipeline_log
       WHERE PASO = 'proj_03';
       %IF &LAST_RUN > &CURRENT_RUN %THEN %DO;
           %PUT ERROR: proj_03 requiere re-ejecución tras proj_02 del &LAST_RUN;
       %END;
   QUIT;
   ```""",
    },
    {
        "id": "ops_user_access_sas_dataset",
        "type": "insight",
        "priority": 0.4,
        "tags": ["operational", "access", "permissions", "SAS", "dataset"],
        "fields": [],
        "articles": [],
        "source": "Incidente SEG-2025-009 — permisos tabla mylib.ciclos_recuperacion",
        "feedback": False,
        "body": """# Permisos incorrectos en mylib.ciclos_recuperacion: usuarios con acceso de escritura

## Incidente
La revisión de seguridad trimestral reveló que 12 usuarios del equipo de negocio tenían permisos de escritura (`ALTER`) sobre `mylib.ciclos_recuperacion`. Esta tabla es la entrada principal del pipeline LGD. 3 de esos usuarios habían modificado valores de `COLATERAL_TIPO` manualmente para "corregir" errores que veían en informes.

## Impacto
- 47 ciclos con COLATERAL_TIPO modificado manualmente
- 2 ciclos con EAD modificada (±5% respecto al original)
- Las modificaciones no eran trazables (no hay auditoría a nivel de fila en SAS)

## Fix
1. Revocar permisos de escritura a usuarios de negocio
2. Implementar tabla de correcciones aprobadas (`mylib.correcciones_aprobadas`) con trazabilidad
3. El pipeline debe aplicar correcciones + loguearlas, no permitir modificaciones directas:
   ```sas
   /* Aplicar correcciones aprobadas */
   CREATE TABLE ciclos_corregidos AS
   SELECT a.*, COALESCE(b.COLATERAL_TIPO, a.COLATERAL_TIPO) AS COLATERAL_TIPO_CORREGIDO
   FROM mylib.ciclos_recuperacion a
   LEFT JOIN mylib.correcciones_aprobadas b ON a.ID_CONTRATO = b.ID_CONTRATO;
   ```

## Lección
En entornos SAS, los permisos de dataset son todo/nada. No hay row-level security. Proteger las tablas fuente del pipeline es crítico.""",
    },
]

ALL = CROSS_SYSTEM + PERFORMANCE + MODEL_VALIDATION + REGULATORY_COMMS + SYSTEM_ALERTS + OPERATIONAL


def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    for exp in ALL:
        path = EXP_DIR / f"{exp['id']}.md"
        path.write_text(_md(exp), encoding="utf-8")
        print(f"  ✓ {path.name}")
    print(f"\n{len(ALL)} new experiences created")


if __name__ == "__main__":
    main()

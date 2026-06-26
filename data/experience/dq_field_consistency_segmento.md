---
id: "dq_field_consistency_segmento"
type: "insight"
priority: 0.6
tags: [data quality, segmento, inconsistencia, SAP, maestro]
fields: [SEGMENTO, SEGMENTO_DESC, COLATERAL_TIPO]
articles: []
source: "Análisis calidad dato — tabla maestra contratos 2025-04"
feedback: false
---

# Inconsistencias en campo SEGMENTO entre tablas maestra y operativa

La tabla maestra `mylib.maestro_contratos` y la tabla operativa
`mylib.ciclos_recuperacion` tienen el campo `SEGMENTO` pero con
valores inconsistentes para el mismo contrato.

## Magnitud

| Tipo inconsistencia | N contratos | % |
|---|---|---|
| Maestro=HIPOTECA, Operativa=CORP | 1,247 | 0.8% |
| Maestro=RETAIL, Operativa=HIPOTECA | 892 | 0.6% |
| Maestro=CORP, Operativa=NINGUNA | 456 | 0.3% |
| Maestro vacío, Operativa=valor | 2,891 | 1.9% |
| **Total** | **5,486** | **3.6%** |

## Causa raíz

La tabla operativa se actualiza con datos de SAP cada mes, pero la
maestra es un snapshot estático. Contratos que cambian de segmento
(por reclasificación) actualizan solo la operativa.

## Impacto

- LGD floor incorrecto por SEGMENTO mal asignado
- PD floor diferenciado por segmento no aplica correctamente
- ECL calculado con parámetros equivocados

## Lección

El `SEGMENTO` no debe tratarse como estable. Siempre verificar
consistencia entre maestro y operativa. Usar COALESCE con prioridad
a la tabla más actualizada.

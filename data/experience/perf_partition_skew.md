---
id: "perf_partition_skew"
type: "insight"
priority: 0.4
tags: [performance, partition, skew, Hadoop, Spark]
fields: []
articles: []
source: "Análisis rendimiento Spark SQL — partition skew en TABLA_CICLOS"
feedback: false
---

# Partition skew en tabla Hadoop por SEGMENTO_CALIBRACION

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
Particionar por columnas con alta skew (distribución no uniforme) causa problemas de rendimiento. Usar múltiples columnas o bucketing.

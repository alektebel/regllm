---
id: "perf_runaway_query"
type: "insight"
priority: 0.5
tags: [performance, query, runaway, SAS, memory]
fields: [OR_EAD, LGD_ESTIMADA]
articles: []
source: "Incidente SAS OOM — query runaway en agregación mensual"
feedback: false
---

# Query runaway: CROSS JOIN implícito en agregación mensual causa OOM

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
Siempre verificar cardinalidad antes de JOINs en SQL. En SAS, los merges implícitos no dan warning hasta que es demasiado tarde.

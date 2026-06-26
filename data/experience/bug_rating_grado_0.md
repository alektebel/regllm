---
id: "bug_rating_grado_0"
type: "insight"
priority: 0.5
tags: [bug, rating, PD, recalibracion, RATING_GRADO]
fields: [RATING_GRADO, PD_ESTIMADA]
articles: [circular_4_2022]
source: "tool_calling_v2_dataset tc_025 — revisión recalibración PD"
feedback: false
---

# RATING_GRADO=0 excluido de recalibración PD

```sas
IF RATING_GRADO in (1,2) THEN PD = PD * 1.15;
```

Ciclos con RATING_GRADO=0 (que existen en sistemas fuente) NO pasan por recalibración ×1.15.

## Causa raíz
Condición asume RATING_GRADO siempre ≥1. Código no documenta el caso RATING_GRADO=0 (posiblemente: nuevo sin rating, o rating externo no mapeado).

## Dilema
¿Es RATING_GRADO=0 un valor válido? Si sí, debe recalibrarse. Si no, debe filtrarse o mapearse antes.

## Impacto
PD_ESTIMADA puede estar infravalorada para ciclos con RATING_GRADO=0 que debieran recalibrarse.

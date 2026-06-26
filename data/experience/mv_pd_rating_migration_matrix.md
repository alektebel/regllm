---
id: "mv_pd_rating_migration_matrix"
type: "insight"
priority: 0.6
tags: [model validation, PD, rating, migration, matrix, EBA]
fields: [RATING_CLASE, RATING_GRADO, PD_ESTIMADA]
articles: [eba_gl_2017_16]
source: "Matriz de migración ratings 2024 — estabilidad vs ciclo"
feedback: false
---

# Matriz de migración de ratings muestra inestabilidad anómala en CORP

La matriz de migración anual (2023→2024) para ratings CORP muestra
una tasa de migración de 2+ grados del 8.3%, muy superior al umbral
de estabilidad esperado (<3%).

## Matriz de transición CORP (%)

| Rating origen | 1 | 2 | 3 | 4+ | Default |
|---|---|---|---|---|---|
| 1 | 72.1 | 18.4 | 5.2 | 3.1 | 1.2 |
| 2 | 10.3 | 58.7 | 20.1 | 8.4 | 2.5 |
| 3 | 2.1 | 12.4 | 55.3 | 22.8 | 7.4 |
| 4+ | 0.5 | 3.2 | 15.1 | 60.2 | 21.0 |

## Anomalías detectadas

1. **Rating 1→4+**: 3.1% migración directa de 1 a 4+ (debería ser ~0%)
2. **Rating 3→Default**: 7.4% (benchmark EBA: 3-5%)
3. **No-diagonales**: Alta densidad fuera de diagonal, sugiere
   que el rating no discrimina adecuadamente riesgo

## Causa probable

La recalibración ×1.15 para ratings 1-2 no es suficiente. Los ratings
están comprimidos en pocas categorías con poder discriminatorio bajo.
El score subyacente tiene baja separación entre clases.

## Impacto

- PD_ESTIMADA no refleja correctamente el riesgo relativo
- Ratings 1 tienen PD similar a ratings 2 después de recalibración
- El floor de PD distorsiona la cola baja de la distribución

## Recomendación

Revisar la calibración del score subyacente. Considerar expandir
a 7+ clases rating (actualmente 4) para mejorar discriminación.

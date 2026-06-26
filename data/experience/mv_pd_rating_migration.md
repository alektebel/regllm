---
id: "mv_pd_rating_migration"
type: "insight"
priority: 0.6
tags: [model validation, PD, rating, migration, matriz]
fields: [RATING_GRADO, PD_ESTIMADA]
articles: [eba_gl_2017_16]
source: "Matriz de migración rating anual 2024-2025"
feedback: false
---

# Matriz de migración rating: estabilidad anormal en grado 5

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
Revisar las guías de asignación de rating para grado 5. Implementar un check de concentración: si RATING_GRADO=5 supera el 25% de la cartera, alertar.

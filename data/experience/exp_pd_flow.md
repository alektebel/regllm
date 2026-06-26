---
id: exp_pd_flow
type: insight
priority: 0.6
tags: [PD, floor, recalibración, rating]
fields: [PD_ESTIMADA, RATING_GRADO]
articles: [circular_4_2022]
source: "Sesiones 1-2 de análisis PD (2025-06)"
feedback: false
---

# Flujo PD: rating → recalibración ×1.15 → floor 0.0005 → PD_ESTIMADA

El pipeline completo de PD combina aprendizajes de dos sesiones:

1. **Rating raw** → `RATING_GRADO` desde catálogo
2. **Recalibración** → ×1.15 para ratings 1 y 2
   (changelog 2025-q1-pd-recalibration.md)
3. **PD floor** → 0.0005 (0.05%) para CORP/soberano,
   0.0003 (0.03%) para retail
4. **PD_ESTIMADA** → resultado final

## Sesiones relacionadas

- **Sesión 1**: Análisis de PD floor en pipeline — ubicación en
  `proj_03_suelos_lgd.sas:20-22`
- **Sesión 2**: Recalibración PD ratings 1-2 — factor ×1.15

## Implementación

```sas
/* Recalibración */
if RATING_GRADO in (1, 2) then PD_ESTIMADA = PD_ESTIMADA * 1.15;
/* Floor */
PD_ESTIMADA = max(PD_ESTIMADA, 0.0005);
```

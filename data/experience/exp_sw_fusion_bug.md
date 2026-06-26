---
id: exp_sw_fusion_bug
type: insight
priority: 0.7
tags: [bug, LGD, SW_FUSION, fusion]
fields: [LGD_ESTIMADA, MoC, ECL]
articles: []
source: "Sesión de análisis LGD (2025-06)"
feedback: false
---

# SW_FUSION=1 causa LGD_ESTIMADA missing en fusionados

Bug documentado en `proj_03_suelos_lgd.sas:33-37`.

Cuando `SW_FUSION=1`, el catálogo SAS del contrato fusionado se
pierde (el contrato original es absorbido por el fusionado). Esto
provoca que `LGD_ESTIMADA` resulte missing para ese registro.

## Efecto en cascada

```
SW_FUSION=1 → LGD_ESTIMADA = . (missing)
  → MoC = 0.05 × . = . (missing)
  → LGD_CON_MOC = . + . = . (missing)
  → ECL = PD × . × EAD = . (missing)
```

## Causa raíz

En `proj_03_suelos_lgd.sas`, las líneas 33-37 aplican la lógica
de fusión pero no hay `COALESCE` ni `CASE WHEN` para manejar el
caso de catálogo absorbido.

## Solución propuesta

```sas
if SW_FUSION=1 and missing(LGD_ESTIMADA) then LGD_ESTIMADA = 0;
/* o: usar COALESCE(LGD_ESTIMADA, 0) */
```

---
id: "dq_negative_lgd_guard"
type: "insight"
priority: 0.5
tags: [data quality, LGD, negative, guard, clamp]
fields: [LGD_ESTIMADA, MoC, ECL, RWA]
articles: []
source: "eval_dataset eval_045 — revisión rangos LGD"
feedback: false
---

# Sin guardia contra LGD_ESTIMADA negativa

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
```

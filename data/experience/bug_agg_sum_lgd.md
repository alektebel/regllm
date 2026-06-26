---
id: "bug_agg_sum_lgd"
type: "insight"
priority: 0.7
tags: [bug, LGD, aggregation, SUM, MAX]
fields: [LGD_ESTIMADA, MoC, ECL]
articles: []
source: "toy_lgd/08_hardest_compound — revisión agregación LGD"
feedback: false
---

# SUM en lugar de MAX para LGD_ESTIMADA en agregación con fusiones

Cuando hay fusiones (SW_FUSION=1), el JOIN no único duplica filas y `SUM(LGD_ESTIMADA)` produce valores fuera de rango [0,1].

Caso real: CIC_006 con LGD=0.55, al duplicarse por fusión → SUM da 1.10 (imposible).

## Causa raíz
Agregación diseñada para 1-fila-por-ciclo, no actualizada cuando se introdujeron fusiones. Usar SUM en una tasa (LGD ∈ [0,1]) es semánticamente incorrecto.

## Fix
```sas
MAX(LGD_ESTIMADA) AS LGD_ESTIMADA  /* reemplaza SUM() */
```

## Impacto
LGD_ESTIMADA, MoC, LGD_CON_MOC y ECL pueden exceder el rango regulatorio. RWA sobrestimado.

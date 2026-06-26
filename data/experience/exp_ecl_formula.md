---
id: exp_ecl_formula
type: insight
priority: 0.6
tags: [ECL, cálculo, fórmula]
fields: [ECL, PD_ESTIMADA, LGD_CON_MOC, EAD]
articles: []
source: "Análisis de pipeline LGD (2025-06)"
feedback: false
---

# ECL = PD_ESTIMADA × LGD_CON_MOC × EAD en proj_03:42

La fórmula del Expected Credit Loss en el pipeline:

```sas
ECL = PD_ESTIMADA * LGD_CON_MOC * EAD;  /* línea 42 */
```

## Desglose

```
ECL = PD_ESTIMADA × LGD_CON_MOC × EAD
    = PD_ESTIMADA × (LGD_ESTIMADA + MoC) × EAD
    = PD_ESTIMADA × (LGD_ESTIMADA + 0.05 × LGD_ESTIMADA) × EAD
```

## Dependencias

- `PD_ESTIMADA`: rating recalibrado con floor 0.05% (0.03% retail)
- `LGD_ESTIMADA`: LGD del catálogo con floor por segmento
- `MoC`: 5% de LGD_ESTIMADA
- `EAD`: EAD_TOTAL (después de ajustes CRM)

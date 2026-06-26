---
id: "edge_missing_auto_corrected"
type: "quirk"
priority: 0.5
tags: [quirk, missing, LGD, auto-correction, hipoteca]
fields: [LGD_ESTIMADA, LGD_FLOOR, COLATERAL_TIPO]
articles: []
severity: "medium"
source: "eval_dataset eval_023 — comportamiento serendípito SAS"
feedback: false
---

# Missing LGD auto-corregido por floor hipotecario (comportamiento serendípito)

En SAS, missing < cualquier número es TRUE:
```sas
IF LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;  /* también corrige missing! */
```

Cuando LGD_ESTIMADA es missing por el bug SW_FUSION, y COLATERAL_TIPO='HIPOTECA', el floor accidentalmente lo corrige a 0.30.

## Efecto colateral
Esto ENMASCARA el bug SW_FUSION para hipotecas. Un analista que solo revisa hipotecas nunca detecta el problema. Solo contratos NO hipotecarios con SW_FUSION=1 muestran LGD missing.

## Lección
Las correcciones automáticas pueden ocultar bugs río arriba. Todo default de missing debería loguearse explícitamente.

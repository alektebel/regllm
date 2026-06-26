---
id: "bug_offbyone_dpd"
type: "insight"
priority: 0.7
tags: [bug, DPD, stage, off-by-one, IFRS9]
fields: [DPDS, STAGE_IFRS9, STAGE_RECLASIFICADO]
articles: [IFRS9_B5_5_12]
source: "toy_lgd/03_easy_boundary — revisión backstop DPD"
feedback: false
---

# Off-by-one: `>` en lugar de `>=` en backstop DPD 30

```sas
IF DPDS > 30 AND STAGE_IFRS9 = 1 THEN DO;  /* BUG: debiera ser >= */
    STAGE_RECLASIFICADO = 2;
END;
```

DPDS=30 exactamente NO dispara reclasificación a Stage 2. IFRS 9 B5.5.12 requiere reclasificar a los 30 DPD.

## Causa raíz
Error clásico de fencepost. Probablemente copy-paste de un umbral exclusivo.

## Fix
```sas
IF DPDS >= 30 AND STAGE_IFRS9 = 1 THEN DO;
```

## Impacto
Ciclos con exactamente 30 DPD permanecen incorrectamente en Stage 1. Subestimación de provisiones.

---
id: "gap_stage3_pd_forced"
type: "insight"
priority: 0.6
tags: [gap, Stage 3, PD, regulatorio, CRR]
fields: [PD_ESTIMADA, STAGE_IFRS9, ECL]
articles: [circular_6_2016_art_15, CRR_154]
source: "eval_dataset eval_049 — revisión Stage 3 PD forced"
feedback: false
---

# Stage 3 PD no forzada a 1.0 en pipeline standalone

CRR requiere `PD_ESTIMADA = 1.0` para exposiciones Stage 3 (deterioradas). El pipeline standalone `proj_03_suelos_lgd.sas` NO implementa:
```sas
IF STAGE_IFRS9 = 3 THEN PD_ESTIMADA = 1.0;
```

La macro `lgd_macros.sas:498-501` SÍ lo hace, pero el script principal no la llama.

## Causa raíz
Inconsistencia entre la librería de macros y el pipeline standalone. El script principal se escribió antes de que existiera la macro.

## Impacto
Stage 3 con PD < 1.0 produce ECL artificialmente baja. Riesgo de infra-provisionamiento regulatorio.

---
id: exp_stage3_conflict
type: insight
priority: 1.0
tags: [feedback, LGD, Stage 3, multiplicador, gap]
fields: [LGD_FLOOR_APLICADO, LGD_ESTIMADA, STAGE_IFRS9]
articles: []
source: "Feedback de validador (2025-06)"
feedback:
  type: correction
  original: "Stage 3 multiplicador 1.20 SÍ está implementado"
  corrected: "Stage 3 multiplicador 1.20 NO está implementado — es gap regulatorio"
---

# [Feedback] Stage 3 multiplicador 1.20 NO implementado — gap regulatorio

**Corrección**: El multiplicador de 1.20 sobre `LGD_FLOOR_APLICADO`
para exposiciones en **Stage 3** NO está implementado en el pipeline.

## Detalle

El insight automático anterior (`exp_stage3_auto`) era incorrecto.
Al revisar `proj_03_suelos_lgd.sas` línea 27-30, NO hay lógica
condicional para Stage 3 que multiplique LGD_FLOOR_APLICADO × 1.20.

## Gap regulatorio

CRR requiere que exposiciones en default (Stage 3) tengan un
recargo del 20% sobre el LGD floor. El pipeline actual no
implementa esto, lo que constituye un **gap regulatorio**.

## Acción requerida

Implementar en `proj_03_suelos_lgd.sas`:
```sas
if STAGE_IFRS9 = 3 then LGD_FLOOR_APLICADO = LGD_FLOOR_APLICADO * 1.20;
```

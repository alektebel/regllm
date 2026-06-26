---
id: "bug_calibracion_pd_central_tendency"
type: "insight"
priority: 0.6
tags: [bug, PD, calibración, central tendency, EBA]
fields: [PD_ESTIMADA, RATING_CLASE, PD_CALIBRADA]
articles: [eba_gl_2017_16_sec6]
source: "Revisión calibración PD — central tendency constraint no aplicado"
feedback: false
---

# PD central tendency constraint no aplicado post-recalibración

EBA GL 2017/16 §6.2 requiere que la PD media ponderada por cartera
no se desvíe más del ±5% del default rate observado a largo plazo.

El pipeline recalibra PD por rating clase pero NO verifica el constraint
de central tendency post-recalibración.

## Síntoma

| Rating | PD antes | PD después (×1.15) | Peso cartera | Contribución |
|---|---|---|---|---|
| 1 | 0.10% | 0.115% | 40% | 0.046% |
| 2 | 0.30% | 0.345% | 25% | 0.086% |
| 3 | 0.80% | 0.920% | 20% | 0.184% |
| 4+ | 2.50% | 2.875% | 15% | 0.431% |
| **PD media ponderada** | | | | **0.747%** |

Default rate observado LTP: 0.65%. Desviación: +14.9% (> ±5%).

## Impacto

ECL sistemáticamente sobrestimado. La recalibración ×1.15 no está
calibrada contra el default rate observado.

## Fix

```sas
/* Post-recalibración: ajuste por central tendency */
DATA pd_adjusted;
    SET pd_calibrated;
    PD_AJUSTADA = PD_RECAL * (0.65 / 0.747);  /* shrinkage factor */
RUN;
```

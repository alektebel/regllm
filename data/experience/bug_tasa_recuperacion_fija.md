---
id: "bug_tasa_recuperacion_fija"
type: "insight"
priority: 0.8
tags: [bug, LGD, recovery, tasa fija, calibración, segmento]
fields: [LGD_REALIZADA, LGD_ESTIMADA, TASA_RECUPERACION]
articles: [eba_gl_2017_16_sec6, circular_6_2016_art_15]
source: "Backtesting 2024 — tasa recuperación fija 0.4 desactualizada"
feedback:
  type: correction
  original: "La tasa de recuperación fija 0.4 es adecuada para todos los segmentos"
  corrected: "La tasa 0.4 solo es válida para el promedio global. Por segmento: CORP=0.38, HIPOTECA=0.52, RETAIL=0.29. Usar tasa fija única distorsiona LGD_REALIZADA por segmento."
---

# Tasa de recuperación fija 0.4 invalida backtesting por segmento

`LGD_REALIZADA = 1 - TASA_RECUPERACION = 1 - 0.4 = 0.6` para TODOS
los contratos, independientemente de su recuperación real.

## Problema

La tasa fija 0.4 significa que LGD_REALIZADA es siempre 0.6, sin
importar cuánto se recuperó realmente. Esto invalida:

1. **Backtesting por segmento**: No se puede comparar LGD estimada vs realizada
   porque la realizada siempre es 0.6
2. **Calibración**: No se puede ajustar el modelo si la variable dependiente
   no varía
3. **Validación EBA**: EBA GL 2017/16 §6.4 requiere backtesting con datos
   empíricos, no con tasa fija

## Impacto

| Segmento | Recovery real | LGD realizada real | LGD realizada (fija) | Sesgo |
|---|---|---|---|---|
| CORP | 0.38 | 0.62 | 0.60 | +0.02 |
| HIPOTECA | 0.52 | 0.48 | 0.60 | -0.12 |
| RETAIL | 0.29 | 0.71 | 0.60 | +0.11 |

LGD_REALIZADA con tasa fija NO refleja la realidad económica del
segmento. Hipotecas aparecen con mayor LGD de la real (0.60 vs 0.48).

## Fix

```sas
/* Usar recovery rate por segmento en lugar de fijo */
IF SEGMENTO = "HIPOTECA" THEN TASA_RECUP = 0.52;
ELSE IF SEGMENTO = "CORP" THEN TASA_RECUP = 0.38;
ELSE IF SEGMENTO = "RETAIL" THEN TASA_RECUP = 0.29;
ELSE TASA_RECUP = 0.40;
LGD_REALIZADA = 1 - TASA_RECUP;
```

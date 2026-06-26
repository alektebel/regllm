---
id: "dq_lgd_outside_range"
type: "insight"
priority: 0.7
tags: [data quality, LGD, range, validación, ECL]
fields: [LGD_ESTIMADA, LGD_REALIZADA, LGD_FLOOR]
articles: []
source: "Control calidad datos — LGD fuera de rango [0,1] 2025-05"
feedback: false
---

# LGD_ESTIMADA y LGD_REALIZADA fuera de rango [0,1] en 2.3% de ciclos

El 2.3% de los ciclos tienen LGD_ESTIMADA o LGD_REALIZADA fuera del
rango regulatorio [0, 1].

## Distribución

| Issue | N ciclos | % | Valor típico |
|---|---|---|---|
| LGD_ESTIMADA < 0 | 845 | 0.6% | -0.15 a -0.05 |
| LGD_ESTIMADA > 1 | 1,234 | 0.9% | 1.05 a 2.50 |
| LGD_REALIZADA < 0 | 512 | 0.4% | -0.30 a -0.10 |
| LGD_REALIZADA > 1 | 678 | 0.5% | 1.10 a 1.80 |
| **Total** | **3,269** | **2.3%** | — |

## Causas

1. **LGD_ESTIMADA < 0**: Recovery rate mal calculado (tasa > 100%)
   o LGD_REALIZADA negativa por ingresos > pérdidas
2. **LGD_ESTIMADA > 1**: Bug SUM en agregación con fusiones
   (ver bug_agg_sum_lgd) o OR_EAD mal asignado
3. **LGD_REALIZADA fuera**: Recovery rate empírico puede exceder
   100% si hay recuperación de gastos previos

## Impacto

ECL usa LGD_ESTIMADA truncada a [0,1]:
```sas
LGD_CLAMPED = MAX(0, MIN(1, LGD_ESTIMADA));
```

Esto enmascara errores upstream. Los valores fuera de rango son
síntoma de bugs aguas arriba.

## Acción

No truncar silenciosamente. Loggear cada valor fuera de rango:
```sas
IF LGD_ESTIMADA NOT BETWEEN 0 AND 1 THEN DO;
    PUT "WARNING: LGD fuera rango " LGD_ESTIMADA= CICLO=;
    /* No truncar — propagar señal de error */
END;
```

---
id: "excel_dqc_outliers"
type: "insight"
priority: 0.6
tags: [excel, DQC, outliers, data quality, LGD]
fields: [LGD_ESTIMADA, OR_EAD, PD_ESTIMADA, ECL]
articles: []
source: "Excel DQC mensual Q2-2025 — hoja outliers LGD"
feedback: false
---

# [EXCEL] Extracto DQC mensual — outliers detectados en pipeline LGD

Fuente: DQC_MENSUAL_Q2_2025.xlsx, hoja "OUTLIERS_LGD"

| Ciclo | LGD | OR_EAD | PD | ECL | Flag |
|---|---|---|---|---|---|
| CIC_042 | 1.10 | 250k | 0.02 | 550 | LGD>1 |
| CIC_071 | 0.00 | 180k | 0.01 | 0 | LGD=0 |
| CIC_089 | . | 95k | 0.03 | . | LGD missing |
| CIC_112 | 0.35 | 1.2M | 0.05 | 2100 | OR_EAD 3x mediana |

## Interpretación del agente
- CIC_042 LGD=1.10 → probable bug SUM vs MAX en agregación con fusiones (bug_agg_sum_lgd)
- CIC_071 LGD=0.00 → posible recovery >= exposure, verificar
- CIC_089 LGD missing → posible SW_FUSION=1
- CIC_112 OR_EAD=1.2M → posible JOIN no único inflando OR_EAD

## Patrón
3 de 4 outliers se explican por bugs conocidos del pipeline. Esto sugiere priorizar fixes antes de invertir en data quality upstream.

---
id: "excel_recovery_rates_raw"
type: "insight"
priority: 0.6
tags: [excel, recovery, LGD, tasa, empírico]
fields: [LGD_REALIZADA, OR_EAD, EAD, SEGMENTO]
articles: []
source: "Excel recuperaciones 2024 — recovery rates observados vs 0.4 fijo"
feedback: false
---

# [EXCEL] Recovery rates observados vs tasa fija 0.4 — análisis empírico

Fuente: RECUPERACIONES_2024.xlsx, hoja "RECOVERY_BY_SEGMENT"

Se comparó la tasa de recuperación real observada en 2024 contra la tasa fija 0.4 usada en el pipeline para LGD_REALIZADA.

| Segmento | N | Recovery medio | Recovery mediana | Desv | vs 0.4 |
|---|---|---|---|---|---|
| CORP | 1,247 | 0.38 | 0.35 | 0.22 | -0.02 |
| HIPOTECA | 3,891 | 0.52 | 0.48 | 0.31 | +0.12 |
| RETAIL | 2,563 | 0.29 | 0.25 | 0.18 | -0.11 |
| TOTAL | 7,701 | 0.43 | 0.40 | 0.28 | +0.03 |

## Conclusiones
1. La media global (0.43) está cerca de 0.4, pero hay gran dispersión por segmento
2. HIPOTECA tiene recovery significativamente mayor (+0.12 vs asunción)
3. RETAIL tiene recovery menor (-0.11 vs asunción)
4. Usar recovery fijo 0.4 infraestima LGD_REALIZADA para hipotecas y la sobrestima para retail

## Recomendación
Parametrizar recovery rate por segmento. Mínimo, usar mediana en vez de media para reducir impacto de outliers.

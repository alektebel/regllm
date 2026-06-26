---
id: "edge_v2_v3_version_aware"
type: "insight"
priority: 0.6
tags: [version, V2, V3, thresholds, migration]
fields: [PD_ESTIMADA, LGD_FLOOR, OR_EAD_TIT, COLATERAL_FIN]
articles: []
source: "v2_to_v3_release_notes.md — análisis diferencias entre versiones"
feedback: false
---

# Diferencias V2 vs V3: thresholds version-sensitive

Múltiples umbrales cambiaron entre V2 y V3. Consultas DQC y validaciones deben ser parameterizadas por versión:

| Parámetro | V2 | V3 |
|---|---|---|
| PD floor | 0.03% | 0.05% |
| CORP LGD floor | 0.45 | 0.50 |
| OR_EAD_TIT | no existe | sí existe |
| FINANCIERO | COLATERAL_TIPO='FINANCIERO' | COLATERAL_FIN=1 |

## Riesgo
Aplicar umbrales de V3 a datos V2 produce falsos positivos (y viceversa). El pipeline debe recibir un parámetro de versión.

## Recomendación
Parametrizar todos los umbrales regulatorios por versión en un archivo de configuración, no hardcodearlos en SAS.

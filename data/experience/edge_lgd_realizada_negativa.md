---
id: "edge_lgd_realizada_negativa"
type: "insight"
priority: 0.5
tags: [edge case, LGD_REALIZADA, negativo, recovery]
fields: [LGD_REALIZADA, OR_EAD, EAD]
articles: []
source: "eval_dataset eval_047, eval_050 — revisión LGD realizada"
feedback: false
---

# LGD_REALIZADA puede ser negativa

```sas
LGD_REALIZADA = 1 - (EAD * 0.4) / OR_EAD;
```
La fórmula puede dar negativo cuando `OR_EAD < EAD * 0.4` (recuperaciones superan exposición). Esto es económicamente posible (ej: apreciación de colateral).

## Problemas
1. Tasa de recuperación fija 0.4 NO validada empíricamente
2. LGD_REALIZADA negativa es correcta pero sorprendente
3. No hay documentación sobre cómo interpretar valores negativos

## Recomendación
Validar la tasa 0.4 contra datos históricos reales. Documentar que LGD_REALIZADA negativa significa recuperaciones netas positivas (ganancia).

---
id: "edge_kirb_nonlinearity"
type: "insight"
priority: 0.4
tags: [edge case, K_IRB, SQRT, non-linear, capital]
fields: [K_IRB, PD_ESTIMADA, RWA, CAPITAL_REGULATORIO]
articles: [CRR_154]
source: "eval_dataset eval_042 — revisión fórmula IRB"
feedback: false
---

# K_IRB no lineal por SQRT(PD) — comportamiento no obvio

```sas
K_IRB = SQRT(PD) * 0.06 + PD * 0.5;
```
Cuando PD se duplica (por recalibración ×1.15 en ratings 1-2):
- PD=0.01 → K_IRB=0.011
- PD=0.02 → K_IRB=0.0185 (solo +68%, no +100%)

## Por qué es relevante
Un revisor podría pensar que el multiplicador de recalibración ×1.15 en PD produce ×1.15 en K_IRB. No es así por la raíz cuadrada. Esto NO es un bug, pero debe documentarse para evitar falsas alarmas en auditoría.

## Nota
Es una propiedad matemática de la fórmula IRB simplificada. La fórmula completa del CRR añade correlación R(PD) que agrava la no-linealidad.

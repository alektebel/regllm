---
id: exp_cure_flag
type: insight
priority: 0.5
tags: [CURE_FLAG, cure, LGD, hipoteca]
fields: [CURE_FLAG, ECL_AJUSTADO, ECL, LGD_ESTIMADA]
articles: [art_23_liberacion_provisiones, art_12_periodos_dotacion]
source: "Análisis cure rate LGD"
feedback: false
---

# CURE_FLAG: dos mecanismos de cure distintos

Existen dos mecanismos de cure que NO deben confundirse:

## 1. CURE_FLAG binario → ajuste ECL (-15%)

`CURE_FLAG=1` aplica 15% de reducción sobre ECL para cualquier segmento:
```
ECL_AJUSTADO = ECL * (1 - CURE_RATE_AJUSTE)
CURE_RATE_AJUSTE = COALESCE(CURE_FLAG, 0) * 0.15
```

Implementado en `proj_03_suelos_lgd.sas:45-48` y `lgd_macros.sas:58`.

## 2. Cure rate LGD (×0.95) — SOLO HIPOTECA

El cure-rate re-fitted sobre 2018-2024 reduce LGD ~5% (×0.95) exclusivamente para `COLATERAL_TIPO = 'HIPOTECA'`. NO aplica a CORP, NINGUNA, ni otros segmentos (feedback documentado en `exp_cure_rate_feedback`).

## Regulatorio

- Art.23 requiere **90 días fuera de default** + `PROVISION_PERIOD_MONTHS` ≥ mínimo + `STAGE_IFRS9` ≤ 2
- `CURE_FLAG=1` solo NO es suficiente para liberación de provisiones

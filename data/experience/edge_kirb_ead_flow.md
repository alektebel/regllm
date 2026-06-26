---
id: "edge_kirb_ead_flow"
type: "insight"
priority: 0.6
tags: [KIRB, EAD, securitización, COREP, RWA]
fields: [KIRB, EAD, RWA, ECL]
articles: [crr_sec_4, eba_gl_2018_01]
source: "Implementación KIRB — revisión fórmula RWA securitizaciones"
feedback: false
---

# KIRB usa OR_EAD (bruta) en lugar de EAD_TOTAL (neta) en securitizaciones

El cálculo de KIRB (capital requirement for securitization positions)
usa `OR_EAD` (EAD bruta antes de CRM) en lugar de `EAD_TOTAL` (EAD neta
después de ajustes CRM).

## Fórmula actual

```sas
KIRB = (LGD * PD * OR_EAD) / OR_EAD;  /* incorrecto: usa OR_EAD */
```

## Fórmula correcta según CRR

```sas
KIRB = (LGD * PD * EAD_TOTAL) / EAD_TOTAL;  /* correcto */
```

## Impacto

Cuando hay CRM (garantías, colaterales), `OR_EAD > EAD_TOTAL`.
Usar OR_EAD sobrestima el denominador y subestima KIRB.

Caso real:
- OR_EAD = 1,000,000
- EAD_TOTAL = 750,000 (después de haircut CRM 25%)
- PD = 1%, LGD = 45%
- KIRB actual: 0.45% (incorrecto)
- KIRB correcto: 0.60% (33% mayor)

## Línea de código

`proj_04_kirb_securitization.sas:15`:
```sas
KIRB = (LGD_CON_MOC * PD_ESTIMADA * OR_EAD) / OR_EAD;
```

## Fix

Reemplazar OR_EAD por EAD_TOTAL en numerador y denominador.

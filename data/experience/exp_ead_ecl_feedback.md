---
id: exp_ead_ecl_feedback
type: insight
priority: 1.0
tags: [feedback, EAD, EAD_TOTAL, OR_EAD_TIT, ECL]
fields: [ECL, EAD_TOTAL, OR_EAD_TIT]
articles: []
source: "Feedback de validador (2025-06)"
feedback:
  type: correction
  original: "EAD en ECL es OR_EAD_TIT"
  corrected: "EAD en ECL es EAD_TOTAL (después de ajustes CRM)"
---

# [Feedback] EAD usada en ECL es EAD_TOTAL no OR_EAD_TIT

**Corrección**: La fórmula `ECL = PD × LGD × EAD` usa
**`EAD_TOTAL`** (EAD después de ajustes CRM), no `OR_EAD_TIT`
(EAD bruta antes de ajustes).

## Diferencia

| Campo | Descripción |
|---|---|
| `OR_EAD_TIT` | EAD nominal bruta, antes de ajustes CRM |
| `EAD_TOTAL` | EAD después de aplicar ajustes por mitigación de riesgo (CRM) |

---
id: exp_pd_floor_feedback
type: insight
priority: 1.0
tags: [feedback, PD, floor, retail]
fields: [PD_ESTIMADA]
articles: [circular_4_2022]
source: "Feedback de validador (2025-06)"
feedback:
  type: correction
  original: "PD floor es 0.05% para todas las exposiciones"
  corrected: "PD floor para retail es 0.03%, para CORP/soberano es 0.05%"
---

# [Feedback] PD floor es 0.03% no 0.05% para retail

**Corrección recibida**: el PD floor **NO** es uniforme.

- **Retail**: 0.03% (0.0003)
- **Corporativo y Soberano**: 0.05% (0.0005)

El insight automático anterior generalizaba incorrectamente.
La Circular 4/2022 establece el 0.05% para CORP/soberano,
pero retail tiene un umbral diferenciado del 0.03%.

---
id: exp_cure_rate_feedback
type: insight
priority: 1.0
tags: [feedback, cure, LGD, hipoteca, CORP]
fields: [LGD_ESTIMADA, CURE_FLAG]
articles: []
source: "Feedback de validador (2025-06)"
feedback:
  type: correction
  original: "El cure rate aplica a todos los segmentos"
  corrected: "El cure rate ×0.95 aplica solo a hipotecas según changelog 2025-q1-lgd-cure.md"
---

# [Feedback] Cure rate NO aplica a corporativo, solo hipotecas

**Corrección importante**: el cure rate (`×0.95`) documentado en
`changelog 2025-q1-lgd-cure.md` aplica **exclusivamente a hipotecas**.

No debe aplicarse a:
- Segmento `CORP`
- Segmento `NINGUNA` (sin colateral)
- Otros segmentos no hipotecarios

## Contexto

El changelog indica reducción del 5% en LGD para hipotecas con
curación documentada. El insight automático original asumió
erróneamente que aplicaba globalmente.

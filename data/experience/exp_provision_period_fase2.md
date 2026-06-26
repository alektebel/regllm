---
id: exp_provision_period_fase2
type: insight
priority: 0.5
tags: [provision_period, fase2, stage2, dotaciones]
fields: [PROVISION_PERIOD_MONTHS, STAGE_IFRS9, FASE]
articles: [art_12_periodos_dotacion, art_23_liberacion_provisiones]
source: "Análisis períodos de dotación"
feedback: false
---

# PROVISION_PERIOD_MONTHS — mínimos por segmento y fase

## Art.12 mínimos

| Segmento | Expansión | Contracción (fase 2) | Crisis |
|---|---|---|---|
| CORP | 12 | **18** | 24 |
| RETAIL | 24 | **30** | 36 |
| MORTGAGE | 36 | **42** | 48 |

## Surcharges

- +6 meses si CORP + `COLATERAL_TIPO = NINGUNA`
- +6 meses si `STAGE_IFRS9 = 3`

## Feedback importante

Originalmente documentado como **24 meses** para fase 2, corregido por equipo de negocio a **36 meses** mínimo para fase 2 (según feedback documentado en experience_eval_dataset).

## Interacción con CURE_FLAG

`CURE_FLAG=1` solo NO libera provisiones — requiere también `PROVISION_PERIOD_MONTHS` ≥ mínimo Art.12 + 90d fuera de default + STAGE≤2 (Art.23).

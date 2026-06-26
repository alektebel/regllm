---
id: "excel_provision_period_matrix"
type: "insight"
priority: 0.6
tags: [excel, provision_period, matriz, DPDS, fase]
fields: [PROVISION_PERIOD_MONTHS, DPDS, STAGE_IFRS9, FASE]
articles: [circular_6_2016_art_12]
source: "Matriz provision_period 2025 — revisión vs Circular 6/2016 Art.12"
feedback:
  type: correction
  original: "Provision period fase 2 = 24 meses para todos los segmentos"
  corrected: "Provision period fase 2: CORP=24m, HIPOTECA=36m, RETAIL=12m. Solo CORP usa 24m."
---

# [EXCEL] Matriz provision_period incorrecta — no segmentada

Fuente: PROVISION_PERIOD_VALIDATION.xlsx

## Matriz correcta (Circular 6/2016 Art.12)

| Fase | DPDS | CORP | HIPOTECA | RETAIL |
|---|---|---|---|---|
| 1 | 0 | 12m | 12m | 12m |
| 2 | 1-12 | 24m | 36m | 12m |
| 2 avanzado | 13+ | 24m | 36m | 12m |
| 3 | 360+ | 0m | 0m | 0m |

## Matriz actual del pipeline

```sas
IF FASE = 2 THEN PROVISION_PERIOD = 24;  /* fijo para todos!!! */
```

## Impacto

- HIPOTECA infra-provisionada (24m vs 36m) → ECL 33% menor del requerido
- RETAIL sobre-provisionado (24m vs 12m) → ECL 100% mayor del necesario

## Base regulatoria

Circular 6/2016 Art.12: "El período de provisiones se determinará
en función del tipo de exposición y de la fase de deterioro,
considerando el plazo remanente y las garantías asociadas."

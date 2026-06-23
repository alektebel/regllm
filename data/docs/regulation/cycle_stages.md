# Cycle Stage Classification and Regulatory Mapping

**Table:** `mylib.ciclos_recuperacion`  
**Regulatory source:** Circular 6/2016 BdE, Artículos 8 and 12

## Overview

Each recovery cycle (`CICLO_ID`) is simultaneously classified by two orthogonal stage systems:

1. **IFRS9 stage** (`STAGE_IFRS9`): accounting classification per IFRS 9 impairment model.
2. **Regulatory cycle phase** (derived from `DPDS`): classification per Circular 6/2016 Art. 8, used to determine provision period minimums.

Both classifications must be mutually consistent (see Art. 8 cross-reference table). Inconsistencies are reported in `NO_CONFORMES`.

## STAGE_IFRS9 values

| Value | Meaning | Regulatory phase constraint |
|---|---|---|
| 1 | Performing — no significant credit risk increase | FASE_EXPANSION only |
| 2 | Underperforming — significant credit risk increase | FASE_EXPANSION or FASE_CONTRACCION |
| 3 | Non-performing — credit-impaired (default) | FASE_CONTRACCION or FASE_CRISIS required |

Stage classification drives:
- `ECL` measurement basis (12-month ECL for Stage 1, lifetime ECL for Stage 2/3)
- `PROVISION_PERIOD_MONTHS` minimum (+6 months when STAGE_IFRS9 = 3)
- `PD_ESTIMADA` floor (1.0 for Stage 3 per Art. 15)
- `LGD_FLOOR_APLICADO` multiplier (+20% for Stage 3 per Art. 15)

## Regulatory cycle phase derivation

The credit cycle phase is not stored directly; it is derived from `DPDS`:

| Phase | DPDS range (general) | Exception: CORP + RATING_GRADO |
|---|---|---|
| FASE_EXPANSION | DPDS < 90 | DPDS < 60 if RATING_GRADO ≤ 3 |
| FASE_CONTRACCION | 90 ≤ DPDS < 360 | 60 ≤ DPDS < 270 if RATING_GRADO ≥ 8 |
| FASE_CRISIS | DPDS ≥ 360 | DPDS ≥ 270 if RATING_GRADO ≥ 8 |

For MORTGAGE with `COLATERAL_TIPO` = HIPOTECA: crisis threshold raised to DPDS ≥ 480.

## Stage transition rules

Transitions between STAGE_IFRS9 values are regulated:

- **3 → 2:** Requires `CURE_FLAG` = 1 plus 90-day probation period.
- **2 → 1:** Requires two consecutive observation windows (`VENTANA_OBSERVACION_YEARS`) at Stage 2.
- **3 → 1 (direct):** Only allowed on full contract cancellation.

## Impact on provision calculations

The combination of `STAGE_IFRS9` and cycle phase determines all provision-related parameters:

```
PROVISION_PERIOD_MONTHS_min = base_min(SEGMENTO, phase)
                              + 6 if STAGE_IFRS9 = 3
                              + 6 if CORP and COLATERAL_TIPO = NINGUNA

PD_floor = 0.05% (general) | 100% (STAGE_IFRS9 = 3)
LGD_floor = LGD_FLOOR_APLICADO × 1.20 (if STAGE_IFRS9 = 3)
ECL = PD_ESTIMADA × LGD_FLOOR_APLICADO × EAD
```

## Related fields

| Field | Role |
|---|---|
| `DPDS` | Primary input for cycle phase derivation |
| `STAGE_IFRS9` | IFRS9 accounting stage |
| `PROVISION_PERIOD_MONTHS` | Months elapsed; checked against regulatory minimum |
| `CURE_FLAG` | Signals stage-down eligibility |
| `VENTANA_OBSERVACION_YEARS` | Observation window length for stage transitions |
| `VENTANA_CALIBRACION_YEARS` | Calibration window; phase changes not effective until window closes |
| `RATING_GRADO` | Adjusts DPDS thresholds for CORP segment |
| `COLATERAL_TIPO` | Adjusts DPDS thresholds for MORTGAGE; surcharge for CORP NINGUNA |
| `NO_CONFORMES` | Flagged when STAGE_IFRS9 and regulatory phase are inconsistent |

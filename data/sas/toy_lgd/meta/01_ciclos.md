# CICLOS — Recovery Cycles (cycle level)

One row per recovery cycle. This is the main input table for LGD estimation.

## Fields

| Field | Type | Length | Description |
|-------|------|--------|-------------|
| ID_CONTR_CICLO_LGD | Char | 30 | Primary key: `{ID_CONTRATO}_{YYYYMM}` |
| ID_CONTRATO | Char | 20 | Contract identifier |
| SEGMENTO | Char | 10 | CORP, RETAIL, SME, MORTGAGE |
| COLATERAL_TIPO | Char | 15 | HIPOTECA, NINGUNA, PERSONAL, FINANCIERO |
| PD_ESTIMADA | Num | 8 | Probability of Default [0,1] |
| LGD_ESTIMADA | Num | 8 | Loss Given Default [0,1] |
| EAD | Num | 8 | Exposure at Default (EUR) |
| DPDS | Num | 8 | Days Past Due |
| STAGE_IFRS9 | Num | 8 | IFRS 9 stage (1, 2, or 3) |
| CURE_FLAG | Num | 8 | 1 = cure rate applies, 0 = no |
| FECHA_INCUMPLIMIENTO | Char | 10 | Default date (YYYY-MM-DD) |
| PERIODO_OBSERVACION | Num | 8 | Observation period (YYYYMM) |
| PROVISION_PERIOD_MONTHS | Num | 8 | Months in provisioning period |
| VENTANA_OBSERVACION_YEARS | Num | 8 | Observation window (years) |
| VENTANA_CALIBRACION_YEARS | Num | 8 | Calibration window (years) |
| OR_EAD | Num | 8 | Original EAD from Basel (enriched) |

## Regulatory floors (CRR Art. 161)

| Segment | Collateral | Floor |
|---------|-----------|-------|
| CORP | NINGUNA | 0.45 |
| CORP | FINANCIERO | 0.35 |
| RETAIL | PERSONAL | 0.35 |
| RETAIL | NINGUNA | 0.45 |
| MORTGAGE | HIPOTECA | 0.25 |
| Any | HIPOTECA | 0.30 |

## Pipeline steps

1. Load CICLOS, enrich with CONTRATOS (SW_FUSION, ID_FUSION_FINAL)
2. Apply LGD floors per segment × collateral
3. Compute MoC and LGD_CON_MOC
4. Compute ECL = PD × LGD × EAD
5. Apply IFRS 9 staging (30 DPD backstop)
6. Flag non-conforming records

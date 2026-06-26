# COLATERALES — Collateral Registry

One row per contract. Provides collateral details for
LGD floor determination.

## Fields

| Field | Type | Length | Description |
|-------|------|--------|-------------|
| ID_CONTRATO | Char | 20 | Primary key (FK to CONTRATOS) |
| COLATERAL_TIPO | Char | 15 | Collateral type |
| VALOR_TASACION | Num | 8 | Appraised value (EUR) |
| LTV | Num | 8 | Loan-to-value ratio |
| FECHA_TASACION | Char | 10 | Appraisal date (YYYY-MM-DD) |

## Collateral types

See CICLOS schema for floor table. The COLATERAL_TIPO in
COLATERALES should match CICLOS.COLATERAL_TIPO.

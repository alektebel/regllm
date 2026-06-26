# BASILEA_MENSUAL — Monthly Basel Data

Monthly data from the Basel system. Multiple rows per contract.
Used to enrich cycles with OR_EAD (Original EAD) for LGD computation.

## Fields

| Field | Type | Length | Description |
|-------|------|--------|-------------|
| ID_CONTRATO | Char | 20 | Contract identifier |
| ID_FUSION_FINAL | Char | 20 | Fusion group ID (if applicable) |
| ID_FCH_DATOS | Num | 8 | Data period (YYYYMM) |
| OR_EAD | Num | 8 | Original EAD (EUR) |
| OR_DISPTO | Num | 8 | Amount drawn |
| OR_DISBLE | Num | 8 | Amount available (undrawn) |

## Lookup logic

For normal contracts (SW_FUSION=0):
  JOIN ON ID_CONTRATO AND ID_FCH_DATOS

For fused contracts (SW_FUSION=1):
  JOIN ON ID_FUSION_FINAL AND ID_FCH_DATOS
  WARNING: ID_FUSION_FINAL may not be unique in BASILEA_MENSUAL.
  Multiple original contracts can share the same ID_FUSION_FINAL,
  each with their own OR_EAD. A naive JOIN produces duplicates.

## Example

| ID_CONTRATO | ID_FUSION_FINAL | ID_FCH_DATOS | OR_EAD |
|-------------|-----------------|--------------|--------|
| CONT_005    | FUS_7712        | 201402       | 100000 |
| CONT_A      | FUS_7712        | 201402       | 100000 |
| CONT_B      | FUS_7712        | 201402       | 100000 |

If a cycle uses ID_FUSION_FINAL=FUS_7712 to join:
  → 2 matching rows in BASILEA_MENSUAL
  → CIC_005 gets 2 rows each with OR_EAD=100000
  → SUM(OR_EAD) = 200000 (wrong — should be 100000)

# FLUJOS_RECUPERACION — Recovery Cashflows

Monthly recovery cashflows for each cycle. Multiple rows
per cycle (one per month of the recovery period).
Used to compute NPV-based LGD.

## Fields

| Field | Type | Length | Description |
|-------|------|--------|-------------|
| ID_CONTR_CICLO_LGD | Char | 30 | Cycle identifier (FK to CICLOS) |
| MES_FLUJO | Num | 8 | Month number (1 = first month of recovery) |
| FLUJO_RECUPERADO | Num | 8 | Gross recovery amount (EUR) |
| COSTE_DIRECTO | Num | 8 | Direct recovery costs (EUR) |
| TASA_DESCUENTO | Num | 8 | Monthly discount rate |

## Usage

NPV of recoveries = SUM over months of:
  (FLUJO_RECUPERADO - COSTE_DIRECTO) / (1 + TASA_DESCUENTO) ^ MES_FLUJO

LGD from cashflows = 1 - NPV_RECOVERY / EAD

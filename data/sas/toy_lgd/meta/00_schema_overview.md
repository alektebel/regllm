# Schema Overview

The LGD estimation pipeline uses the following tables:

| Table | Level | Rows per key | Description |
|-------|-------|-------------|-------------|
| CICLOS | Cycle | 1 per cycle | Main cycle-level data (PD, LGD, EAD, etc.) |
| CONTRATOS | Contract | 1 per contract | Contract metadata, fusion flags |
| BASILEA_MENSUAL | Contract-month | Many per contract | Monthly Basel data (OR_EAD) |
| FLUJOS_RECUPERACION | Cycle-month | Many per cycle | Monthly recovery cashflows |
| COLATERALES | Contract | 1 per contract | Collateral details |

## Key identifier formats

- `ID_CONTR_CICLO_LGD`: `{ID_CONTRATO}_{YYYYMM}` — e.g. `12345678_201402`
- `ID_CONTRATO`: numeric string — e.g. `12345678`
- `ID_FUSION_FINAL`: fusion group ID — e.g. `FUS_7712`
- `ID_FCH_DATOS`: numeric period — e.g. `201402` (YYYYMM)

## Relationships

```
CONTRATOS ──1:1──> CICLOS (via ID_CONTRATO in ID_CONTR_CICLO_LGD)
CONTRATOS ──1:N──> BASILEA_MENSUAL (via ID_CONTRATO)
CICLOS    ──1:N──> FLUJOS_RECUPERACION (via ID_CONTR_CICLO_LGD)
CONTRATOS ──1:1──> COLATERALES (via ID_CONTRATO)
```

When SW_FUSION=1, the cycle's ID_CONTRATO may map to a contract that
was absorbed. In that case, BASILEA_MENSUAL lookups should use
ID_FUSION_FINAL instead of ID_CONTRATO.

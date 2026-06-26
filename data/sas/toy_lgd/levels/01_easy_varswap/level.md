# Level 01 — Easy: Variable Swap in Floor Condition

## Scenario

The pipeline applies regulatory LGD floors. For HIPOTECA collateral,
LGD_ESTIMADA below 0.30 should be raised to 0.30 (CRR Art. 154(3)).

## Expected correct output

```
CIC_003: LGD_ESTIMADA = 0.30  (was 0.20, floor should raise it)
CIC_004: LGD_ESTIMADA = 0.40  (already above floor, unchanged)
```

## Actual buggy output

```
CIC_003: LGD_ESTIMADA = 0.20  (floor NOT applied — stays at original)
```

## Hint

Compare the condition in the HIPOTECA floor block with the
CORP floor block directly above it. One variable is wrong.

## Tables needed

- CICLOS (only table used)

## Queries to try

```sql
SELECT ID_CONTR_CICLO_LGD, COLATERAL_TIPO, LGD_ESTIMADA, EAD
FROM work.floored
WHERE COLATERAL_TIPO = 'HIPOTECA';
```

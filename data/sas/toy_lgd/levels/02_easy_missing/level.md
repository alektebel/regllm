# Level 02 — Easy: Missing COALESCE for Fusion Contracts

## Scenario

Fusion contracts (SW_FUSION=1) may have missing LGD_ESTIMADA because
the absorbed entity's system doesn't report it. The code should
default to the regulatory floor (0.45) before computing MoC.

## Expected correct output

```
CIC_005 (fusion, LGD missing): LGD_ESTIMADA=0.45, MoC=0.0225, ECL=2835
CIC_006 (fusion, LGD=0.55):    LGD_ESTIMADA=0.55, MoC=0.0275, ECL=2887.5
```

## Actual buggy output

```
CIC_005: LGD_ESTIMADA=., MoC=., LGD_CON_MOC=., ECL=.
```

## Tables needed

- CICLOS (cycle data)
- CONTRATOS (fusion flags)

## Queries

```sql
SELECT c.ID_CONTR_CICLO_LGD, c.SW_FUSION, e.LGD_ESTIMADA, e.MoC, e.ECL
FROM work.ecl e
JOIN work.ciclos_enriched c ON c.ID_CONTR_CICLO_LGD = e.ID_CONTR_CICLO_LGD
WHERE c.SW_FUSION = 1;
```

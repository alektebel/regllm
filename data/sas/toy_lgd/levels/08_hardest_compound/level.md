# Level 08 — Hardest: Compound Bug (Fusion + Wrong Aggregation)

## Scenario

This level combines two independent bugs that interact:

1. A fusion join duplication (like Level 05) that inflates the
   number of rows in intermediate tables
2. A wrong aggregation function (using SUM instead of MAX for
   LGD_ESTIMADA during de-duplication)

On their own, each bug has a minor effect. Together, they compound
to produce severely wrong ECL values.

## Expected correct output

```
CIC_005: OR_EAD=100000, LGD_ESTIMADA=0.45(default), ECL=2835
CIC_006: OR_EAD=100000, LGD_ESTIMADA=0.55,           ECL=2887.5
```

## Buggy output

```
CIC_005: OR_EAD=200000, LGD_ESTIMADA=0.90(??), ECL=wrong
CIC_006: OR_EAD=200000, LGD_ESTIMADA=1.10(??), ECL=wrong
```

## Tables needed

- CICLOS
- CONTRATOS
- BASILEA_MENSUAL

## Hint

Two bugs on two different lines. One is in the JOIN, one is
in the aggregation. Fixing only one still leaves wrong results.

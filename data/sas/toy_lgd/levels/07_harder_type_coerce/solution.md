# Level 07 — Solution

## Bug

Line: `IF COLATERAL_TIPO = 'HIPOTECA' AND LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;`

**Case-sensitive comparison**: SAS character comparisons are case-sensitive.
The source data has inconsistent casing: 'HIPOTECA', 'hipoteca', 'Hipoteca'.
Only the uppercase variant matches.

## Why it fails

The data comes from multiple systems (after a bank merger) that use
different casing conventions. The upstream ETL didn't standardize the
case. The comparison `COLATERAL_TIPO = 'HIPOTECA'` uses exact match,
so 'hipoteca' and 'Hipoteca' silently fail to match.

SAS generates no warning or error for this — it simply evaluates to
false, and the floor is not applied for those rows.

## Fix

Use the `UPCASE` function (or `LOWCASE`) to normalize before comparing:

```sas
IF UPCASE(COLATERAL_TIPO) = 'HIPOTECA' AND LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;
```

Or standardize the data on input:
```sas
COLATERAL_TIPO = UPCASE(COLATERAL_TIPO);
```

## Root cause

Assumption of data quality. The code assumes COLATERAL_TIPO is
always uppercase, but post-merger data integration often introduces
case inconsistencies. This is a common real-world issue in banking
systems after M&A activity.

# Level 02 — Solution

## Bug

Line: `MoC = 0.05 * LGD_ESTIMADA;`

**No COALESCE/default** before using LGD_ESTIMADA. When it is missing
(for fusion-absorbed contracts), the missing value propagates:
`MoC = .` → `LGD_CON_MOC = .` → `ECL = .`

## Why it fails

Fusion contracts (SW_FUSION=1) from absorbed entities may not have
LGD_ESTIMADA reported. The code assumes LGD_ESTIMADA is always present,
but it can be missing. SAS arithmetic with missing values produces
missing results.

## Fix

Add a COALESCE before MoC:

```sas
IF LGD_ESTIMADA = . THEN LGD_ESTIMADA = 0.45;  /* default to CORP floor */
MoC = 0.05 * LGD_ESTIMADA;
```

Or use a more conservative approach:
```sas
MoC = 0.05 * COALESCE(LGD_ESTIMADA, 0.45);
```

## Root cause

The code doesn't account for data quality issues in fusion scenarios.
The absorbed entity may not report all fields, and the pipeline needs
defensive defaults for these cases.

# Level 03 — Solution

## Bug

Line: `IF DPDS > 30 AND STAGE_IFRS9 = 1 THEN DO;`

**Wrong operator**: `>` instead of `>=`. The boundary value 30 is excluded
when it should be included.

## Why it fails

IFRS 9 B5.5.12 states that exposures with 30+ days past due should
be reclassified to Stage 2. Using `>` means DPDS=30 is treated the
same as DPDS=29 — no reclassification.

## Fix

```sas
IF DPDS >= 30 AND STAGE_IFRS9 = 1 THEN DO;
```

## Root cause

Classic off-by-one fencepost error. The distinction between
"greater than" and "greater than or equal to" is one of the most
common boundary condition mistakes in programming.

# Level 01 — Solution

## Bug

Line: `IF COLATERAL_TIPO = 'HIPOTECA' AND EAD < 0.30 THEN LGD_ESTIMADA = 0.30;`

**Wrong variable**: `EAD` is used instead of `LGD_ESTIMADA` in the condition.

## Why it fails

EAD values are in the hundreds of thousands (e.g., 150000), so `EAD < 0.30`
is always false. The HIPOTECA floor block never executes, and LGD values
below 0.30 (like CIC_003 with 0.20) pass through unchanged.

## Fix

Replace `EAD` with `LGD_ESTIMADA`:

```sas
IF COLATERAL_TIPO = 'HIPOTECA' AND LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;
```

## Root cause

Copy-paste or naming confusion: EAD and LGD_ESTIMADA are adjacent in the
schema and both are numeric, making it easy to grab the wrong one.

# Level 04 — Solution

## Bug

Line: `WHERE PROVISION_PERIOD_MONTHS > 12;`

**Wrong threshold and wrong comparison**: Should filter for `>= 9` but
filters for `> 12` instead. Two mistakes: the operator is wrong (`>` vs
`>=`) AND the threshold is wrong (12 vs 9).

## Why it fails

Circular 6/2016 Art. 12 requires a minimum provision period of 9 months.
The code uses 12 as the threshold, which excludes valid cycles with
9-11 months of history. CIC_004 (9 months) is excluded despite meeting
the regulatory requirement.

## Fix

```sas
WHERE PROVISION_PERIOD_MONTHS >= 9;
```

## Root cause

The threshold 12 might come from a different interpretation ("at least
12 months of data") or from an earlier version of the regulation. The
`>` instead of `>=` is an additional boundary error that compounds the
threshold mistake.

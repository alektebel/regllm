# Level 08 — Solution

## Bug 1: Fusion join duplication (like Level 05)

Line: `LEFT JOIN ... ON t1.ID_FUSION_FINAL = t2.ID_FUSION_FINAL`

Non-unique ID_FUSION_FINAL produces duplicate rows.

## Bug 2: Wrong aggregation function

Line: `SUM(LGD_ESTIMADA) AS LGD_ESTIMADA`

Should be `MAX(LGD_ESTIMADA) AS LGD_ESTIMADA`. When the JOIN produces
2 rows per cycle (each with LGD_ESTIMADA=0.55), SUM gives 1.10 — an
impossible LGD value (outside [0,1]).

## Interaction

| Scenario | Fix Bug 1 only | Fix Bug 2 only | Fix both |
|----------|----------------|----------------|---------|
| 1 row (no fusion) | Correct | Correct | Correct |
| 2 rows (fusion) | Still SUM doubles | Still duplicated | Correct |

Each bug alone would affect OR_EAD but not LGD_ESTIMADA (since OR_EAD
uses SUM which is wrong for duplicates, and LGD_ESTIMADA uses SUM
which is also wrong). Fixing only one bug still leaves the other
producing wrong results.

## Root cause

The aggregation was designed when each cycle had exactly 1 row.
The fusion join broke that assumption, but the aggregation wasn't
updated to handle duplicates. Using SUM for LGD is doubly wrong
because LGD is a rate [0,1], not a summable quantity.

## Fix

```sas
MAX(LGD_ESTIMADA) AS LGD_ESTIMADA,   /* or MIN, or MEAN */
MAX(OR_EAD_BASILEA) AS OR_EAD         /* or DISTINCT + SUM */
```

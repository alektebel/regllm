# Level 05 — Solution

## Bug

Two-part bug in the OR_EAD enrichment:

1. **Non-unique JOIN key**: `LEFT JOIN ... ON ID_FUSION_FINAL` matches
   multiple rows in BASILEA_MENSUAL because multiple contracts share
   the same fusion ID. Each row carries OR_EAD=100000, producing
   2 result rows per cycle.

2. **Wrong aggregation**: `SUM(OR_EAD_BASILEA)` adds the duplicated
   values instead of taking `MAX(OR_EAD_BASILEA)` or using `DISTINCT`.

## Why it fails

The join on ID_FUSION_FINAL is meant to get the OR_EAD of the
absorbed contract, but it matches ALL contracts in the fusion group.
The aggregation should use MAX (which would give 100000 from
[100000, 100000]) but uses SUM (which gives 200000).

## Fix

Replace SUM with MAX in the aggregation:
```sas
MAX(OR_EAD_BASILEA) AS OR_EAD
```

Or deduplicate before the join:
```sas
SELECT DISTINCT ID_FUSION_FINAL, ID_FCH_DATOS, OR_EAD
FROM basilea_mensual
```

## Root cause

The assumption that ID_FUSION_FINAL is unique in BASILEA_MENSUAL.
In reality, a fusion group contains multiple original contracts,
each with their own OR_EAD. The code needs to decide which one
to use (e.g., the first, the max, or an average).

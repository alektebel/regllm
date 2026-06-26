# Level 04 — Medium: Wrong WHERE Filter Excluding Valid Rows

## Scenario

The pipeline filters cycles to those with sufficient provisioning history.
The regulatory requirement (Circular 6/2016 Art. 12) is that cycles with
PROVISION_PERIOD_MONTHS >= 9 are valid for inclusion.

## Expected correct output

```
Rows included: CIC_001 (12mo), CIC_002 (10mo), CIC_004 (9mo), CIC_005 (15mo)
Rows excluded: CIC_003 (6mo)
Total: 4 rows in work.ciclos
```

## Actual buggy output

```
Rows included: CIC_001 (12mo), CIC_002 (10mo), CIC_005 (15mo)
Rows excluded: CIC_003 (6mo), CIC_004 (9mo)  ← CIC_004 incorrectly excluded!
Total: 3 rows in work.ciclos
```

## Hint

Look at the WHERE condition used to filter cycles. Compare the threshold
against the regulatory minimum of 9 months. One row that should be
included is being excluded.

## Tables needed

- CICLOS only

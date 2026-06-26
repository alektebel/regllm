# Level 03 — Easy: Off-by-one in DPD Backstop

## Scenario

IFRS 9 B5.5.12 requires reclassification from Stage 1 to Stage 2
when days past due (DPDS) reaches 30. The condition should be `>= 30`.

## Expected correct output

```
CIC_007 (DPDS=30): STAGE_IFRS9=2, STAGE_RECLASIFICADO=1
CIC_008 (DPDS=29): STAGE_IFRS9=1, STAGE_RECLASIFICADO=0
CIC_009 (DPDS=31): STAGE_IFRS9=2, STAGE_RECLASIFICADO=1
```

## Actual buggy output

```
CIC_007 (DPDS=30): STAGE_IFRS9=1, STAGE_RECLASIFICADO=0  ← WRONG
CIC_008 (DPDS=29): STAGE_IFRS9=1, STAGE_RECLASIFICADO=0  ← correct
CIC_009 (DPDS=31): STAGE_IFRS9=2, STAGE_RECLASIFICADO=1  ← correct
```

## Hint

Compare the comparison operator with the regulatory requirement.
"At 30 days past due" means DPDS >= 30.

## Tables needed

- CICLOS only

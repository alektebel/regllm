# Level 05 — Medium: Fusion Join Duplication

## Scenario

For fused contracts (SW_FUSION=1), OR_EAD is looked up from
BASILEA_MENSUAL using ID_FUSION_FINAL. However, ID_FUSION_FINAL
is NOT unique — multiple original contracts can share the same
fusion ID. A naive LEFT JOIN produces duplicate rows.

## Data setup

Two cycles (CIC_005 from CONT_005, CIC_006 from CONT_006) both
share ID_FUSION_FINAL=FUS_7712. BASILEA_MENSUAL has two entries
for FUS_7712 (CONT_A and CONT_B, each with OR_EAD=100000).

After JOIN: each cycle gets 2 rows → SUM(OR_EAD)=200000 (should be 100000).

## Expected correct output

```
CIC_005: OR_EAD = 100000 (original EAD of the absorbed contract)
CIC_006: OR_EAD = 100000
```

## Buggy output

```
CIC_005: OR_EAD = 200000 (doubled — SUM adds both matching rows)
CIC_006: OR_EAD = 200000
```

## Tables needed

- CICLOS
- CONTRATOS (for fusion flags)
- BASILEA_MENSUAL (monthly Basel data)

## Queries

```sql
SELECT ID_CONTR_CICLO_LGD, SW_FUSION, OR_EAD
FROM work.ciclos_con_ead
WHERE SW_FUSION = 1;
```

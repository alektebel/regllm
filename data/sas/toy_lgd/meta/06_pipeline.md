# Pipeline — LGD Estimation Steps

The pipeline transforms raw cycle data into final LGD, ECL, and RWA.
Each step produces an intermediate table.

## Step 1: Load & Enrich

```
Input:  CICLOS + CONTRATOS
Output: work.ciclos_enriched
Logic: LEFT JOIN CICLOS → CONTRATOS on ID_CONTRATO
       Adds SW_FUSION, ID_FUSION_FINAL, ENTIDAD_ORIGEN
```

## Step 2: OR_EAD enrichment (if applicable)

```
Input:  work.ciclos_enriched + BASILEA_MENSUAL
Output: work.ciclos_con_ead
Logic: LEFT JOIN on ID_CONTRATO (normal) or ID_FUSION_FINAL (fused)
       Aggregates to cycle level with SUM(OR_EAD)
```

## Step 3: Regulatory LGD floors

```
Input:  work.ciclos_con_ead (or work.ciclos_enriched)
Output: work.floored
Logic: IF SEGMENTO/COLATERAL_TIPO match a floor AND
       LGD_ESTIMADA < floor THEN LGD_ESTIMADA = floor
```

## Step 4: MoC and LGD_CON_MOC

```
Input:  work.floored
Output: work.ecl
Logic: MoC = 0.05 * LGD_ESTIMADA
       LGD_CON_MOC = LGD_ESTIMADA + MoC
       PD_ESTIMADA = MAX(PD_ESTIMADA, 0.0003)
       ECL = PD_ESTIMADA * LGD_CON_MOC * EAD
```

## Step 5: IFRS 9 staging

```
Input:  work.ecl
Output: work.final
Logic: IF DPDS >= 30 AND STAGE_IFRS9 = 1 THEN STAGE_IFRS9 = 2
       Stage 3: PD = 1.0
```

## Expected output fields

The final table `work.final` should contain (at minimum):
ID_CONTR_CICLO_LGD, LGD_ESTIMADA, MoC, LGD_CON_MOC, ECL,
ECL_AJUSTADO, STAGE_IFRS9, STAGE_RECLASIFICADO

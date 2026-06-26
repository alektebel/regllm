# Level 06 — Hard: Wrong BY-Group in Segment Average for MoC

## Scenario

The Margin of Conservatism (MoC) is based on the deviation of each
cycle's LGD from its segment mean. The segment mean should be
computed per SEGMENTO (CORP, RETAIL, etc.).

## Expected correct output

```
Segment means: CORP=0.475, RETAIL=0.35
CIC_001 (CORP, LGD=0.45): dev=0.00, MoC=0.0225
CIC_002 (CORP, LGD=0.50): dev=0.05, MoC=0.005 (10% of 0.05)
CIC_007 (RETAIL, LGD=0.35): dev=0.00, MoC=0.0175 (5% floor)
CIC_008 (RETAIL, LGD=0.40): dev=0.05, MoC=0.005 (10% of 0.05)
```

## Buggy output

```
Segment means computed by wrong group (e.g. COLATERAL_TIPO):
  NINGUNA=0.475, PERSONAL=0.375
CIC_001 (NINGUNA, LGD=0.45): MoC wrong
CIC_007 (PERSONAL, LGD=0.35): MoC wrong
```

## Tables needed

- CICLOS only

## Hint

Compare the BY variable in the PROC MEANS / SQL GROUP BY for
segment means against the variable used elsewhere in the pipeline.
Are they the same?

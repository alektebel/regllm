# Field PD_ESTIMADA

`PD_ESTIMADA` is the 1-year point-in-time probability of default
estimated by the master scale.

## Floors

CRR Art. 160(1) sets a regulatory floor of **0.03 %**. The V3 pipeline
raises this floor to **0.05 %** following the 2025-Q1 master-scale
recalibration. The change is implemented in the data step
`work.ecl_calculo`.

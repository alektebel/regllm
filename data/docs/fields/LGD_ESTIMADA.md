# Field LGD_ESTIMADA

Loss Given Default — the proportion of `EAD` expected to be lost net of
recoveries.

## Floors

| Segment / Collateral                | V2 floor | V3 floor |
|--------------------------------------|---------:|---------:|
| HIPOTECA (CRR Art. 154(3))           | 0.30     | 0.30     |
| CORP (CRR Art. 161(1)(b))            | 0.45     | **0.50** |
| RETAIL no collateral                 | none     | none     |

The CORP floor was tightened by 5 pp in V3 (data step
`work.lgd_con_suelos`) per the 2025-Q1 supervisory dialogue.

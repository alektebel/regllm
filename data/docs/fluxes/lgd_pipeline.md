# LGD calibration pipeline

## Stages

1. `work.ciclos` — read raw recovery cycles, filter
   `PROVISION_PERIOD_MONTHS >= 9`.
2. `work.lgd_con_suelos` — apply regulatory LGD floors per
   `COLATERAL_TIPO` × `SEGMENTO`.
3. `work.ecl_calculo` — apply the PD floor, compute
   `ECL = PD × LGD × EAD`, reclassify IFRS 9 stage on the 30-DPD
   backstop.
4. `work.titulizado` *(V3 only)* — derive `OR_EAD_TIT` from `EAD` and
   `SEGMENTO`.
5. `work.no_conformes` — flag rows that fail one of: short calibration
   window, short observation window, mortgage with LGD < floor, PD <
   regulatory floor.

## V2 → V3 differences

- The CORP LGD floor was raised from 0.45 to 0.50.
- The PD floor was raised from 0.0003 to 0.0005.
- A new step `work.titulizado` was inserted before the descriptive
  statistics, producing `OR_EAD_TIT`.
- Downstream `no_conformes` reflects the new PD floor automatically.

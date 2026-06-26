/*****************************************************************************
 * Level 03 — Easy: Off-by-one in DPD backstop
 * Bug: > 30 instead of >= 30 in the IFRS 9 backstop condition.
 * DPDS = 30 should trigger reclassification to Stage 2 but doesn't.
 *****************************************************************************/

DATA work.ciclos;
    LENGTH ID_CONTR_CICLO_LGD ID_CONTRATO SEGMENTO COLATERAL_TIPO $20;
    LENGTH PD_ESTIMADA LGD_ESTIMADA EAD 8;
    LENGTH DPDS STAGE_IFRS9 CURE_FLAG PROVISION_PERIOD_MONTHS 8;
    INPUT ID_CONTR_CICLO_LGD $ ID_CONTRATO $ SEGMENTO $ COLATERAL_TIPO $
          PD_ESTIMADA LGD_ESTIMADA EAD
          DPDS STAGE_IFRS9 CURE_FLAG PROVISION_PERIOD_MONTHS;
    DATALINES;
CIC_007 CONT_007 RETAIL PERSONAL   0.030 0.35  80000 30 1 0 12
CIC_008 CONT_008 RETAIL PERSONAL   0.030 0.35  90000 29 1 0 12
CIC_009 CONT_009 RETAIL PERSONAL   0.030 0.35 100000 31 1 0 12
;
RUN;

/* Step 2: Apply floors */
DATA work.floored;
    SET work.ciclos;
RUN;

/* Step 3: MoC and ECL */
DATA work.ecl;
    SET work.floored;
    MoC = 0.05 * LGD_ESTIMADA;
    LGD_CON_MOC = LGD_ESTIMADA + MoC;
    IF PD_ESTIMADA < 0.0003 THEN PD_ESTIMADA = 0.0003;
    ECL = PD_ESTIMADA * LGD_CON_MOC * EAD;
RUN;

/* Step 4: Staging — BUG: > instead of >= */
DATA work.final;
    SET work.ecl;
    IF DPDS > 30 AND STAGE_IFRS9 = 1 THEN DO;
        STAGE_IFRS9 = 2;
        STAGE_RECLASIFICADO = 1;
    END;
    ELSE STAGE_RECLASIFICADO = 0;
RUN;

/* Expected correct:
   CIC_007: DPDS=30 → STAGE=2, RECLAS=1
   CIC_008: DPDS=29 → STAGE=1, RECLAS=0
   CIC_009: DPDS=31 → STAGE=2, RECLAS=1

   Actual (buggy):
   CIC_007: DPDS=30 → STAGE=1, RECLAS=0  ← wrong
*/

/*****************************************************************************
 * Level 01 — Easy: Variable swap in floor condition
 * Bug: Uses EAD instead of LGD_ESTIMADA in HIPOTECA floor check.
 * Because EAD is in the hundreds of thousands, EAD < 0.30 is always false,
 * so the HIPOTECA floor is never applied.
 *****************************************************************************/

DATA work.ciclos;
    LENGTH ID_CONTR_CICLO_LGD ID_CONTRATO SEGMENTO COLATERAL_TIPO $20;
    LENGTH PD_ESTIMADA LGD_ESTIMADA EAD 8;
    LENGTH DPDS STAGE_IFRS9 CURE_FLAG PROVISION_PERIOD_MONTHS 8;
    INPUT ID_CONTR_CICLO_LGD $ ID_CONTRATO $ SEGMENTO $ COLATERAL_TIPO $
          PD_ESTIMADA LGD_ESTIMADA EAD
          DPDS STAGE_IFRS9 CURE_FLAG PROVISION_PERIOD_MONTHS;
    DATALINES;
CIC_001 CONT_001 CORP NINGUNA      0.010 0.40 100000  0 1 0 12
CIC_002 CONT_002 CORP NINGUNA      0.010 0.50 200000  0 1 0 12
CIC_003 CONT_003 CORP HIPOTECA     0.010 0.20 150000  0 1 0 12
CIC_004 CONT_004 CORP HIPOTECA     0.010 0.40 180000  0 1 0 12
CIC_005 CONT_005 CORP NINGUNA      0.020 0.55 300000  0 1 0 12
;
RUN;

/* Step 2: Apply floors */
DATA work.floored;
    SET work.ciclos;

    /* CORP floor: 45% */
    IF SEGMENTO = 'CORP' AND COLATERAL_TIPO = 'NINGUNA' THEN DO;
        IF LGD_ESTIMADA < 0.45 THEN LGD_ESTIMADA = 0.45;
    END;

    /* HIPOTECA floor: 30% — BUG: EAD instead of LGD_ESTIMADA in condition */
    IF COLATERAL_TIPO = 'HIPOTECA' AND EAD < 0.30 THEN LGD_ESTIMADA = 0.30;
RUN;

/* Step 3: MoC and ECL */
DATA work.ecl;
    SET work.floored;
    MoC = 0.05 * LGD_ESTIMADA;
    LGD_CON_MOC = LGD_ESTIMADA + MoC;
    IF PD_ESTIMADA < 0.0003 THEN PD_ESTIMADA = 0.0003;
    ECL = PD_ESTIMADA * LGD_CON_MOC * EAD;
    IF CURE_FLAG = 1 THEN ECL_AJUSTADO = ECL * 0.85;
    ELSE                ECL_AJUSTADO = ECL;
RUN;

/* Step 4: Staging */
DATA work.final;
    SET work.ecl;
    IF DPDS >= 30 AND STAGE_IFRS9 = 1 THEN DO;
        STAGE_IFRS9 = 2;
        STAGE_RECLASIFICADO = 1;
    END;
    ELSE STAGE_RECLASIFICADO = 0;
RUN;

/* Expected correct:
   CIC_003: LGD_ESTIMADA=0.30, MoC=0.015, LGD_CON_MOC=0.315, ECL=472.5
   Actual (buggy):
   CIC_003: LGD_ESTIMADA=0.20, MoC=0.01,  LGD_CON_MOC=0.21,   ECL=315
*/

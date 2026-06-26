/*****************************************************************************
 * Level 04 — Medium: Wrong WHERE filter
 * Bug: Uses PROVISION_PERIOD_MONTHS > 12 instead of >= 9.
 * Cycles with 9-11 months are erroneously excluded.
 *****************************************************************************/

DATA work.ciclos;
    LENGTH ID_CONTR_CICLO_LGD ID_CONTRATO SEGMENTO COLATERAL_TIPO $20;
    LENGTH PD_ESTIMADA LGD_ESTIMADA EAD 8;
    LENGTH DPDS STAGE_IFRS9 CURE_FLAG PROVISION_PERIOD_MONTHS 8;
    INPUT ID_CONTR_CICLO_LGD $ ID_CONTRATO $ SEGMENTO $ COLATERAL_TIPO $
          PD_ESTIMADA LGD_ESTIMADA EAD
          DPDS STAGE_IFRS9 CURE_FLAG PROVISION_PERIOD_MONTHS;
    DATALINES;
CIC_001 CONT_001 CORP NINGUNA      0.010 0.45 100000  0 1 0 12
CIC_002 CONT_002 CORP NINGUNA      0.010 0.50 200000  0 1 0 10
CIC_003 CONT_003 CORP NINGUNA      0.010 0.45 150000  0 1 0  6
CIC_004 CONT_004 CORP NINGUNA      0.010 0.45 180000  0 1 0  9
CIC_005 CONT_005 CORP NINGUNA      0.020 0.55 300000  0 1 0 15
;
RUN;

/* Step 1: Filter cycles — BUG: > 12 instead of >= 9 */
DATA work.ciclos_filtrados;
    SET work.ciclos;
    WHERE PROVISION_PERIOD_MONTHS > 12;  /* BUG: should be >= 9 */
RUN;

/* Step 2: Apply floors */
DATA work.floored;
    SET work.ciclos_filtrados;
    IF SEGMENTO = 'CORP' AND COLATERAL_TIPO = 'NINGUNA' THEN DO;
        IF LGD_ESTIMADA < 0.45 THEN LGD_ESTIMADA = 0.45;
    END;
RUN;

/* Step 3: MoC and ECL */
DATA work.ecl;
    SET work.floored;
    MoC = 0.05 * LGD_ESTIMADA;
    LGD_CON_MOC = LGD_ESTIMADA + MoC;
    IF PD_ESTIMADA < 0.0003 THEN PD_ESTIMADA = 0.0003;
    ECL = PD_ESTIMADA * LGD_CON_MOC * EAD;
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
   4 rows: CIC_001 (12), CIC_002 (10), CIC_004 (9), CIC_005 (15)
   Actual (buggy):
   3 rows: CIC_001 (12), CIC_002 (10), CIC_005 (15)
   Missing: CIC_004 (9) — wrongfully excluded
*/

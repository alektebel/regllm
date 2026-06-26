/*****************************************************************************
 * Level 06 — Hard: Wrong BY-group in segment average for MoC
 * Bug: Computes segment averages by COLATERAL_TIPO instead of SEGMENTO.
 * This subtly distorts MoC values because the grouping is wrong.
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
CIC_002 CONT_002 CORP NINGUNA      0.010 0.50 200000  0 1 0 12
CIC_007 CONT_007 RETAIL PERSONAL   0.030 0.35  80000 30 1 0 12
CIC_008 CONT_008 RETAIL PERSONAL   0.030 0.40  90000 29 1 0 12
CIC_009 CONT_009 CORP FINANCIERO   0.010 0.40 120000  5 1 0 12
;
RUN;

/* Step 1: Apply floors */
DATA work.floored;
    SET work.ciclos;
    IF SEGMENTO = 'CORP' AND COLATERAL_TIPO = 'NINGUNA' THEN DO;
        IF LGD_ESTIMADA < 0.45 THEN LGD_ESTIMADA = 0.45;
    END;
RUN;

/* Step 2: Compute segment means — BUG: uses COLATERAL_TIPO not SEGMENTO */
PROC SQL;
    CREATE TABLE work.segment_means AS
    SELECT
        COLATERAL_TIPO AS LGD_GROUP,  /* BUG: should be SEGMENTO */
        MEAN(LGD_ESTIMADA) AS LGD_MEDIA_GROUP
    FROM work.floored
    GROUP BY COLATERAL_TIPO;  /* BUG: should be SEGMENTO */
QUIT;

/* Step 3: MoC and ECL using wrong group */
DATA work.ecl;
    MERGE work.floored (IN=a) work.segment_means (IN=b);
    BY COLATERAL_TIPO;  /* BUG: MERGE on wrong key */
    IF a;
    LGD_DESVIACION = LGD_ESTIMADA - LGD_MEDIA_GROUP;
    IF LGD_DESVIACION > 0 THEN MoC = 0.10 * LGD_DESVIACION;
    ELSE                     MoC = 0.05 * LGD_ESTIMADA;
    LGD_CON_MOC = LGD_ESTIMADA + MoC;
    IF PD_ESTIMADA < 0.0003 THEN PD_ESTIMADA = 0.0003;
    ECL = PD_ESTIMADA * LGD_CON_MOC * EAD;
RUN;

/* Expected:
   CORP mean = (0.45+0.50)/2 = 0.475 (from CIC_001, CIC_002)
   RETAIL mean = (0.35+0.40)/2 = 0.375 (from CIC_007, CIC_008)
   CIC_001: LGD_GROUP should be CORP → dev=0.00 → MoC=0.0225(floor)
   CIC_007: LGD_GROUP should be RETAIL → dev=-0.025 → MoC=0.0175(floor)

   Actual (buggy):
   Groups by COLATERAL_TIPO: NINGUNA=0.475, PERSONAL=0.375, FINANCIERO=0.40
   CIC_009 uses FINANCIERO mean (0.40) instead of CORP mean → MoC wrong
*/

/*****************************************************************************
 * Level 07 — Harder: Hidden type coercion
 * Bug: COLATERAL_TIPO values have inconsistent casing in source data.
 * The comparison IF COLATERAL_TIPO = 'HIPOTECA' is case-sensitive in SAS,
 * so 'Hipoteca' and 'hipoteca' don't match. SAS also has a numeric
 * comparison trap: comparing a character variable to a numeric literal
 * triggers implicit conversion that may always fail silently.
 *****************************************************************************/

DATA work.ciclos;
    LENGTH ID_CONTR_CICLO_LGD ID_CONTRATO SEGMENTO COLATERAL_TIPO $20;
    LENGTH PD_ESTIMADA LGD_ESTIMADA EAD 8;
    LENGTH DPDS STAGE_IFRS9 CURE_FLAG PROVISION_PERIOD_MONTHS 8;
    INPUT ID_CONTR_CICLO_LGD $ ID_CONTRATO $ SEGMENTO $ COLATERAL_TIPO $
          PD_ESTIMADA LGD_ESTIMADA EAD
          DPDS STAGE_IFRS9 CURE_FLAG PROVISION_PERIOD_MONTHS;
    DATALINES;
CIC_001 CONT_001 CORP HIPOTECA     0.010 0.20 100000  0 1 0 12
CIC_002 CONT_002 CORP hipoteca     0.010 0.20 200000  0 1 0 12
CIC_003 CONT_003 CORP Hipoteca     0.010 0.20 150000  0 1 0 12
CIC_004 CONT_004 CORP NINGUNA      0.010 0.50 180000  0 1 0 12
;
RUN;

/* Step 2: Apply floors
   BUG: COLATERAL_TIPO comparison is case-sensitive.
   The data contains 'HIPOTECA', 'hipoteca', and 'Hipoteca'.
   Only 'HIPOTECA' matches — the other two case variants pass through
   without the floor being applied.

   Additional trap: COLATERAL_TIPO is a CHAR variable but the constant
   is also CHAR, so SAS compares as CHAR. There's no type coercion here
   but the case mismatch is the hidden issue. */
DATA work.floored;
    SET work.ciclos;
    IF COLATERAL_TIPO = 'HIPOTECA' AND LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;
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

/* Expected: All HIPOTECA rows (CIC_001, CIC_002, CIC_003) have
   LGD_ESTIMADA=0.30 (floor applied).

   Actual (buggy):
   CIC_001: LGD=0.30 (HIPOTECA matches 'HIPOTECA' — floor applied)
   CIC_002: LGD=0.20 (hipoteca !== 'HIPOTECA' — floor NOT applied)
   CIC_003: LGD=0.20 (Hipoteca !== 'HIPOTECA' — floor NOT applied)
*/

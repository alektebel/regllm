/*****************************************************************************
 * TOY LGD — CORRECT VERSION (no bugs)
 * =====================================
 * Reference implementation with the three bugs fixed.
 * Compare output with toy_lgd.sas to see the differences.
 *****************************************************************************/

DATA work.cycles;
    LENGTH CICLO_ID CONTRATO SEGMENTO COLATERAL_TIPO $20;
    LENGTH SW_FUSION PD_ESTIMADA LGD_ESTIMADA EAD 8;
    LENGTH DPDS STAGE_IFRS9 CURE_FLAG PROVISION_PERIOD_MONTHS 8;
    INPUT CICLO_ID $ CONTRATO $ SEGMENTO $ COLATERAL_TIPO $
          SW_FUSION PD_ESTIMADA LGD_ESTIMADA EAD
          DPDS STAGE_IFRS9 CURE_FLAG PROVISION_PERIOD_MONTHS;
    DATALINES;
CIC_001 CONT_001 CORP NINGUNA      0 0.010 0.40 100000  0 1 0 12
CIC_002 CONT_002 CORP NINGUNA      0 0.010 0.50 200000  0 1 0 12
CIC_003 CONT_003 CORP HIPOTECA     0 0.010 0.20 150000  0 1 0 12
CIC_004 CONT_004 CORP HIPOTECA     0 0.010 0.40 180000  0 1 0 12
CIC_005 CONT_005 CORP NINGUNA      1 0.020 .    300000 15 1 0 12
CIC_006 CONT_006 CORP NINGUNA      1 0.020 0.55 250000 20 1 0 12
CIC_007 CONT_007 RETAIL PERSONAL   0 0.030 0.35  80000 30 1 0 12
CIC_008 CONT_008 RETAIL PERSONAL   0 0.030 0.35  90000 29 1 0 12
CIC_009 CONT_009 CORP NINGUNA      0 0.010 0.45 120000  5 1 1 12
CIC_010 CONT_010 CORP NINGUNA      0 0.010 0.45 110000  5 1 0 12
;
RUN;

/* ── Corrected floor application ────────────────────────────── */
DATA work.floored;
    SET work.cycles;

    /* CORP floor: 45% */
    IF SEGMENTO = 'CORP' AND COLATERAL_TIPO = 'NINGUNA' THEN DO;
        IF LGD_ESTIMADA < 0.45 THEN LGD_ESTIMADA = 0.45;
    END;

    /* HIPOTECA floor: 30% — FIXED: uses LGD_ESTIMADA, not EAD */
    IF COLATERAL_TIPO = 'HIPOTECA' AND LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;

RUN;

/* ── Corrected MoC with COALESCE ────────────────────────────── */
DATA work.ecl;
    SET work.floored;

    /* FIXED: default missing LGD to regulatory floor */
    IF LGD_ESTIMADA = . THEN LGD_ESTIMADA = 0.45;

    MoC = 0.05 * LGD_ESTIMADA;
    LGD_CON_MOC = LGD_ESTIMADA + MoC;

    IF PD_ESTIMADA < 0.0003 THEN PD_ESTIMADA = 0.0003;

    ECL = PD_ESTIMADA * LGD_CON_MOC * EAD;

    IF CURE_FLAG = 1 THEN ECL_AJUSTADO = ECL * (1 - 0.15);
    ELSE                ECL_AJUSTADO = ECL;

RUN;

/* ── Corrected DPD backstop ─────────────────────────────────── */
DATA work.final;
    SET work.ecl;

    /* FIXED: >= 30 to match IFRS 9 B5.5.12 */
    IF DPDS >= 30 AND STAGE_IFRS9 = 1 THEN DO;
        STAGE_IFRS9 = 2;
        STAGE_RECLASIFICADO = 1;
    END;
    ELSE STAGE_RECLASIFICADO = 0;

RUN;

/* ── Print results ──────────────────────────────────────────── */
TITLE "CORRECT — work.floored";
PROC PRINT DATA=work.floored;
    VAR CICLO_ID SEGMENTO COLATERAL_TIPO LGD_ESTIMADA EAD;
RUN;

TITLE "CORRECT — work.ecl";
PROC PRINT DATA=work.ecl;
    VAR CICLO_ID SW_FUSION LGD_ESTIMADA MoC LGD_CON_MOC ECL ECL_AJUSTADO;
RUN;

TITLE "CORRECT — work.final";
PROC PRINT DATA=work.final;
    VAR CICLO_ID DPDS STAGE_IFRS9 STAGE_RECLASIFICADO ECL ECL_AJUSTADO;
RUN;

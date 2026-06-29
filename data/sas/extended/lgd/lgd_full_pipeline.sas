/*****************************************************************************
 * lgd_full_pipeline.sas
 * ======================
 * Pipeline completo de estimacion LGD.
 * Recovery rate por segmento, ajuste colateral, floors, MoC, cure rate,
 * LGD_NPV con descuento mensual.
 *****************************************************************************/

%put NOTE: [lgd_full_pipeline] Inicio del pipeline LGD.;

/* Incluir macros */
%include "data/sas/extended/macros/regllm_macros.sas";

/* ── Datos de ejemplo ────────────────────────────────────────────────── */
DATA work.lgd_input;
    LENGTH CICLO_ID $12 SEGMENTO $12 COLATERAL_TIPO $12;
    INPUT CICLO_ID $ SEGMENTO $ COLATERAL_TIPO $
          EAD VALOR_GARANTIA LTV RECOVERY_AMOUNT CURE_INDICATOR;
    DATALINES;
CONTR_001 CORP FINANCIERO 1000000 800000 0.80 400000 0
CONTR_002 RETAIL PERSONAL 150000 30000 0.20 60000 0
CONTR_003 MORTGAGE HIPOTECA 1800000 1500000 0.83 900000 1
CONTR_004 SME NINGUNA 300000 0 0 90000 0
CONTR_005 CORP FINANCIERO 4000000 2000000 0.50 1600000 0
CONTR_006 RETAIL NINGUNA 50000 0 0 15000 0
CONTR_007 MORTGAGE HIPOTECA 1600000 1200000 0.75 800000 1
CONTR_008 SME NINGUNA 200000 0 0 60000 0
CONTR_009 CORP FINANCIERO 2500000 1500000 0.60 1000000 0
CONTR_010 RETAIL PERSONAL 100000 10000 0.10 40000 0
;
RUN;

%put NOTE: [lgd_full_pipeline] &sqlobs registros cargados.;

/* ── Calculo LGD completo ────────────────────────────────────────────── */
DATA work.lgd_result;
    SET work.lgd_input;

    /* Recovery rate base por segmento */
    SELECT (SEGMENTO);
        WHEN ('CORP')    RECOVERY_RATE = 0.40;
        WHEN ('RETAIL')  RECOVERY_RATE = 0.35;
        WHEN ('MORTGAGE') RECOVERY_RATE = 0.50;
        WHEN ('SME')     RECOVERY_RATE = 0.30;
        OTHERWISE        RECOVERY_RATE = 0.35;
    END;

    /* Ajuste por colateral (coverage ratio) */
    IF EAD > 0 THEN COLATERAL_COVERAGE = VALOR_GARANTIA / EAD;
    ELSE COLATERAL_COVERAGE = 0;
    COLATERAL_COVERAGE = MIN(1, COLATERAL_COVERAGE);

    /* Recovery ajustado por colateral */
    RECOVERY_ADJ = RECOVERY_RATE + (1 - RECOVERY_RATE) * COLATERAL_COVERAGE * 0.50;
    RECOVERY_ADJ = MIN(1, RECOVERY_ADJ);

    /* LGD base */
    LGD_BASE = 1 - RECOVERY_ADJ;

    /* LGD floor via macro */
    IF SEGMENTO = 'CORP' AND COLATERAL_TIPO = 'FINANCIERO' THEN LGD_FLOOR_VAL = 0.35;
    ELSE IF SEGMENTO = 'CORP' AND COLATERAL_TIPO = 'NINGUNA' THEN LGD_FLOOR_VAL = 0.50;
    ELSE IF SEGMENTO = 'RETAIL' AND COLATERAL_TIPO = 'PERSONAL' THEN LGD_FLOOR_VAL = 0.35;
    ELSE IF SEGMENTO = 'RETAIL' AND COLATERAL_TIPO = 'NINGUNA' THEN LGD_FLOOR_VAL = 0.45;
    ELSE IF SEGMENTO = 'MORTGAGE' AND COLATERAL_TIPO = 'HIPOTECA' THEN LGD_FLOOR_VAL = 0.25;
    ELSE IF SEGMENTO = 'MORTGAGE' AND COLATERAL_TIPO = 'NINGUNA' THEN LGD_FLOOR_VAL = 0.40;
    ELSE LGD_FLOOR_VAL = 0;

    LGD_FLOORED = MAX(LGD_BASE, LGD_FLOOR_VAL);

    /* Margin of Constraint (MoC) 5% */
    MOC_RATE = 0.05;
    LGD_MOC = LGD_FLOORED * (1 + MOC_RATE);
    LGD_MOC = MIN(1, LGD_MOC);

    /* Cure rate adjustment: 15% para hipotecas con LTV < 75% */
    IF SEGMENTO = 'MORTGAGE' AND COLATERAL_TIPO = 'HIPOTECA' AND LTV < 0.75 THEN
        LGD_CURE = LGD_MOC * (1 - 0.15);
    ELSE
        LGD_CURE = LGD_MOC;

    /* LGD_NPV con descuento mensual (5% anual) */
    TASA_DESC_MENSUAL = (1 + 0.05) ** (1/12) - 1;
    MESES_RECUPERACION = 24; /* supuesto: 24 meses para recuperacion */
    FACTOR_DESC = 1 / ((1 + TASA_DESC_MENSUAL) ** MESES_RECUPERACION);

    IF EAD > 0 THEN
        LGD_NPV = 1 - (RECOVERY_ADJ * FACTOR_DESC);
    ELSE
        LGD_NPV = 1;

    LGD_NPV = MAX(LGD_FLOOR_VAL, MIN(1, LGD_NPV));

    DEC = LGD_MOC * EAD;

    FORMAT RECOVERY_RATE RECOVERY_ADJ COLATERAL_COVERAGE 8.3
           LGD_BASE LGD_FLOOR_VAL LGD_FLOORED LGD_MOC LGD_CURE LGD_NPV 8.4
           DEC 12.0;
RUN;

%put NOTE: [lgd_full_pipeline] LGD calculado para todos los registros.;

/* ── Resultados ──────────────────────────────────────────────────────── */
PROC PRINT DATA=work.lgd_result;
    VAR CICLO_ID SEGMENTO COLATERAL_TIPO RECOVERY_RATE COLATERAL_COVERAGE
        LGD_BASE LGD_FLOOR_VAL LGD_FLOORED LGD_MOC LGD_CURE LGD_NPV DEC;
    TITLE "LGD Full Pipeline - Resultados";
RUN;

PROC MEANS DATA=work.lgd_result N MEAN MIN MAX;
    CLASS SEGMENTO;
    VAR LGD_BASE LGD_FLOORED LGD_MOC LGD_NPV DEC;
    TITLE "Resumen LGD por Segmento";
RUN;

%put NOTE: [lgd_full_pipeline] Proceso completado.;

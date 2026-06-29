/*****************************************************************************
 * pd_rating_model.sas
 * ====================
 * Scorecard compuesto para estimacion de PD y asignacion de rating.
 * Variables: financiero (40%), behavioral (30%), colateral (15%), industria (15%)
 * Blend: 70% modelo / 30% experto (simulado con semilla fija).
 *****************************************************************************/

%put NOTE: [pd_rating_model] Inicio del modelo de rating PD.;

/* Incluir macros */
%include "data/sas/extended/macros/regllm_macros.sas";

/* ── Datos de ejemplo ────────────────────────────────────────────────── */
DATA work.contratos_pd;
    LENGTH CICLO_ID $12 SEGMENTO $12 SECTOR_ECONOMICO $20 COLATERAL_TIPO $12;
    INPUT CICLO_ID $ SEGMENTO $ SECTOR_ECONOMICO $ EBITDA DEUDA_NETA ACTIVO_TOTAL
          PATRIMONIO DIAS_MORA_AVG PAGO_PROMPTO_PCT TENURE_MONTHS
          VALOR_GARANTIA COLATERAL_TIPO $ EAD;
    DATALINES;
CONTR_001 CORP MANUFACTURA 500000 300000 1500000 600000 15 0.85 36 800000 FINANCIERO 1000000
CONTR_002 RETAIL SERVICIOS 80000 50000 200000 80000 45 0.60 24 30000 PERSONAL 150000
CONTR_003 MORTGAGE CONSTRUCCION 200000 400000 2000000 500000 0 1.00 60 1500000 HIPOTECA 1800000
CONTR_004 SME COMERCIO 150000 200000 600000 150000 90 0.40 18 0 NINGUNA 300000
CONTR_005 CORP TECNOLOGIA 1200000 800000 5000000 2000000 5 0.95 48 2000000 FINANCIERO 4000000
CONTR_006 RETAIL AGRICULTURA 30000 20000 100000 40000 120 0.30 12 5000 PERSONAL 50000
CONTR_007 MORTGAGE SERVICIOS 180000 350000 1800000 450000 0 1.00 72 1200000 HIPOTECA 1600000
CONTR_008 SME MANUFACTURA 100000 150000 400000 100000 30 0.75 24 0 NINGUNA 200000
CONTR_009 CORP COMERCIO 800000 600000 3000000 1200000 10 0.90 60 1500000 FINANCIERO 2500000
CONTR_010 RETAIL TECNOLOGIA 50000 40000 150000 60000 60 0.55 36 10000 PERSONAL 100000
;
RUN;

%put NOTE: [pd_rating_model] Datos cargados (&sqlobs registros).;

/* ── Scorecard compuesto ─────────────────────────────────────────────── */
DATA work.pd_scored;
    SET work.contratos_pd;

    /* Score financiero (40%) */
    SCORE_FIN = %score_financial(EBITDA, DEUDA_NETA, ACTIVO_TOTAL, PATRIMONIO);

    /* Score behavioral (30%) */
    SCORE_BEH = %score_behavioral(DIAS_MORA_AVG, PAGO_PROMPTO_PCT, TENURE_MONTHS);

    /* Score colateral (15%) */
    SCORE_COL = %score_collateral(VALOR_GARANTIA, EAD, COLATERAL_TIPO);

    /* Score industria (15%) - basado en sector */
    SELECT (SECTOR_ECONOMICO);
        WHEN ('TECNOLOGIA')  SCORE_IND = 80;
        WHEN ('MANUFACTURA') SCORE_IND = 60;
        WHEN ('SERVICIOS')   SCORE_IND = 50;
        WHEN ('COMERCIO')    SCORE_IND = 45;
        WHEN ('CONSTRUCCION') SCORE_IND = 35;
        WHEN ('AGRICULTURA') SCORE_IND = 30;
        OTHERWISE            SCORE_IND = 40;
    END;

    /* Score total compuesto */
    SCORE_TOTAL = SCORE_FIN * 0.40 + SCORE_BEH * 0.30 +
                  SCORE_COL * 0.15 + SCORE_IND * 0.15;

    /* PD via logistica */
    PD_ESTIMADA = %pd_from_score(SCORE_TOTAL);
    RATING_ASIGNADO = %rating_from_pd(PD_ESTIMADA);
    FORMAT PD_ESTIMADA 8.4;

    /* Blend: 70% modelo / 30% experto (semilla fija) */
    RETAIN SEMILLA 12345;
    EXPERT_SCORE = 20 + INT(RANUNI(SEMILLA) * 60);
    SCORE_BLEND  = SCORE_TOTAL * 0.70 + EXPERT_SCORE * 0.30;
    PD_BLEND     = %pd_from_score(SCORE_BLEND);
    RATING_INICIAL = %rating_from_pd(PD_BLEND);
    FORMAT PD_BLEND 8.4;
    DROP SEMILLA EXPERT_SCORE;
RUN;

%put NOTE: [pd_rating_model] Scorecard calculado.;

/* ── Resultados ──────────────────────────────────────────────────────── */
PROC PRINT DATA=work.pd_scored;
    VAR CICLO_ID SEGMENTO SCORE_FIN SCORE_BEH SCORE_COL SCORE_IND SCORE_TOTAL
        PD_ESTIMADA RATING_ASIGNADO SCORE_BLEND PD_BLEND RATING_INICIAL;
    TITLE "PD Rating Model - Scorecard Compuesto";
RUN;

PROC FREQ DATA=work.pd_scored;
    TABLES RATING_ASIGNADO RATING_INICIAL;
    TITLE "Distribucion de Ratings";
RUN;

%put NOTE: [pd_rating_model] Proceso completado.;

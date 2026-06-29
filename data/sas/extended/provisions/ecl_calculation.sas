/*****************************************************************************
 * ecl_calculation.sas
 * ====================
 * Calculo de Expected Credit Loss (ECL) bajo IFRS9.
 * ECL 12m para Stage 1, ECL lifetime para Stage 2/3.
 * Stage 3: forced PD=1.0. Credit card EAD ajustado.
 *****************************************************************************/

%put NOTE: [ecl_calculation] Inicio de calculo ECL.;

/* Incluir macros */
%include "data/sas/extended/macros/regllm_macros.sas";

/* ── Datos de ejemplo ────────────────────────────────────────────────── */
DATA work.ecl_input;
    LENGTH CICLO_ID $12 SEGMENTO $12 TIPO_PRODUCTO $16;
    INPUT CICLO_ID $ SEGMENTO $ TIPO_PRODUCTO $
          STAGE_IFRS9 PD_ESTIMADA LGD_ESTIMADA EAD
          YEARS_TO_MATURITY SALDO_DISPONIBLE LINEA_TOTAL;
    DATALINES;
CONTR_001 CORP PRESTAMO 1 0.0012 0.35 1000000 4.5 0 0
CONTR_002 RETAIL LINEA_CREDITO 2 0.0250 0.45 150000 2.0 120000 300000
CONTR_003 MORTGAGE PRESTAMO 1 0.0008 0.25 1800000 8.0 0 0
CONTR_004 SME TARJETA 3 0.0450 0.60 300000 1.5 280000 300000
CONTR_005 CORP PRESTAMO 1 0.0050 0.40 4000000 3.0 0 0
CONTR_006 RETAIL DESCUENTO 2 0.0800 0.55 50000 2.5 0 0
CONTR_007 MORTGAGE PRESTAMO 1 0.0030 0.28 1600000 10.0 0 0
CONTR_008 SME LINEA_CREDITO 3 0.1200 0.65 200000 0.5 180000 200000
CONTR_009 CORP PRESTAMO 2 0.0100 0.38 2500000 5.0 0 0
CONTR_010 RETAIL TARJETA 1 0.0050 0.42 100000 1.0 80000 100000
;
RUN;

%put NOTE: [ecl_calculation] &sqlobs registros cargados.;

/* ── Calculo ECL ─────────────────────────────────────────────────────── */
DATA work.ecl_result;
    SET work.ecl_input;

    /* PD original preservada */
    PD_ORIGINAL = PD_ESTIMADA;

    /* Stage 3: forced PD=1.0 (politica conservadora) */
    IF STAGE_IFRS9 = 3 THEN PD_ECL = 1.0;
    ELSE PD_ECL = PD_ESTIMADA;

    /* Credit card EAD ajustado (utilizacion media) */
    IF TIPO_PRODUCTO = 'TARJETA' AND LINEA_TOTAL > 0 THEN DO;
        USAGE_RATE = SALDO_DISPONIBLE / LINEA_TOTAL;
        EAD_CARD = LINEA_TOTAL * (0.5 + USAGE_RATE * 0.5);
        EAD_FINAL = EAD_CARD;
    END;
    ELSE DO;
        EAD_CARD = .;
        EAD_FINAL = EAD;
    END;

    /* ECL 12 meses */
    ECL_12M = %ecl_12m(PD_ECL, LGD_ESTIMADA, EAD_FINAL);

    /* ECL lifetime (con min(years_to_maturity, 5)) */
    LIFETIME_YEARS = MIN(YEARS_TO_MATURITY, 5);
    ECL_LIFETIME = %ecl_lifetime(PD_ECL, LGD_ESTIMADA, EAD_FINAL, YEARS_TO_MATURITY);

    /* Provision final segun Stage */
    IF STAGE_IFRS9 = 1 THEN PROVISION = ECL_12M;
    ELSE PROVISION = ECL_LIFETIME;

    /* Coverage ratio */
    IF EAD_FINAL > 0 THEN COVERAGE_RATIO = PROVISION / EAD_FINAL;
    ELSE COVERAGE_RATIO = 0;

    /* ECL en bps */
    ECL_BPS = COVERAGE_RATIO * 10000;

    FORMAT PD_ESTIMADA PD_ECL 8.4 ECL_12M ECL_LIFETIME PROVISION 14.2
           COVERAGE_RATIO 8.5 ECL_BPS 8.1 EAD_FINAL 12.0;
RUN;

%put NOTE: [ecl_calculation] ECL calculado.;

/* ── Resultados ──────────────────────────────────────────────────────── */
PROC PRINT DATA=work.ecl_result;
    VAR CICLO_ID SEGMENTO STAGE_IFRS9 PD_ESTIMADA PD_ECL LGD_ESTIMADA
        EAD_FINAL ECL_12M ECL_LIFETIME PROVISION COVERAGE_RATIO ECL_BPS;
    TITLE "ECL Calculation - Resultados IFRS9";
RUN;

PROC MEANS DATA=work.ecl_result N MEAN SUM;
    CLASS STAGE_IFRS9;
    VAR EAD_FINAL PROVISION COVERAGE_RATIO ECL_BPS;
    TITLE "Resumen ECL por Stage";
RUN;

PROC MEANS DATA=work.ecl_result N MEAN SUM;
    CLASS TIPO_PRODUCTO;
    VAR PROVISION EAD_FINAL COVERAGE_RATIO;
    TITLE "Resumen ECL por Tipo Producto";
RUN;

%put NOTE: [ecl_calculation] Proceso completado.;

/*****************************************************************************
 * ifrs9_staging.sas
 * ==================
 * Determinacion de Stage IFRS9 con SICR detection, matriz de transicion,
 * y calculo de Lifetime PD.
 *****************************************************************************/

%put NOTE: [ifrs9_staging] Inicio de staging IFRS9.;

/* Incluir macros */
%include "data/sas/extended/macros/regllm_macros.sas";

/* ── Datos de ejemplo ────────────────────────────────────────────────── */
DATA work.staging_input;
    LENGTH CICLO_ID $12 SEGMENTO $12;
    INPUT CICLO_ID $ SEGMENTO $ PD_ESTIMADA DPD FORBEARANCE_FLAG
          MAX_DPD_12M VECES_MORA_12M YEARS_TO_MATURITY;
    DATALINES;
CONTR_001 CORP 0.0012 0 0 15 0 4.5
CONTR_002 RETAIL 0.0250 35 0 45 2 2.0
CONTR_003 MORTGAGE 0.0008 0 0 0 0 8.0
CONTR_004 SME 0.0450 95 0 120 4 1.5
CONTR_005 CORP 0.0050 0 1 30 1 3.0
CONTR_006 RETAIL 0.0800 60 0 90 3 2.5
CONTR_007 MORTGAGE 0.0030 0 0 10 0 10.0
CONTR_008 SME 0.1200 150 0 180 6 0.5
CONTR_009 CORP 0.0100 20 0 25 1 5.0
CONTR_010 RETAIL 0.0050 0 0 0 0 1.0
;
RUN;

%put NOTE: [ifrs9_staging] &sqlobs registros cargados.;

/* ── Deteccion SICR y Stage ──────────────────────────────────────────── */
DATA work.staging_result;
    SET work.staging_input;

    /* SICR detection */
    IF DPD > 30 OR FORBEARANCE_FLAG = 1 OR
       (VECES_MORA_12M >= 3 AND MAX_DPD_12M >= 30) THEN SICR_FLAG = 1;
    ELSE SICR_FLAG = 0;

    /* Stage determination */
    IF DPD > 90 OR FORBEARANCE_FLAG = 1 THEN STAGE_IFRS9 = 3;
    ELSE IF SICR_FLAG = 1 THEN STAGE_IFRS9 = 2;
    ELSE STAGE_IFRS9 = 1;

    /* Lifetime PD: MIN(years_to_maturity, 5) */
    LIFETIME_YEARS = MIN(YEARS_TO_MATURITY, 5);
    PD_LIFETIME = PD_ESTIMADA * LIFETIME_YEARS;

    /* Matriz de transicion (simulada) */
    TRANS_RATE_S1_S2 = 0.08;  /* 8% transicion Stage 1 -> 2 */
    TRANS_RATE_S2_S3 = 0.15;  /* 15% transicion Stage 2 -> 3 */
    TRANS_RATE_S3_S1 = 0.02;  /* 2% mejora Stage 3 -> 1 */

    FORMAT PD_ESTIMADA PD_LIFETIME 8.4;
RUN;

%put NOTE: [ifrs9_staging] Staging completado.;

/* ── Resultados ──────────────────────────────────────────────────────── */
PROC PRINT DATA=work.staging_result;
    VAR CICLO_ID SEGMENTO PD_ESTIMADA DPD FORBEARANCE_FLAG
        SICR_FLAG STAGE_IFRS9 LIFETIME_YEARS PD_LIFETIME;
    TITLE "IFRS9 Staging - SICR y Asignacion de Stage";
RUN;

PROC FREQ DATA=work.staging_result;
    TABLES STAGE_IFRS9 * SICR_FLAG / NOROW NOCOL NOPERCENT;
    TITLE "Matriz Stage x SICR";
RUN;

PROC MEANS DATA=work.staging_result N MEAN;
    CLASS STAGE_IFRS9;
    VAR PD_ESTIMADA PD_LIFETIME;
    TITLE "PD Promedio por Stage";
RUN;

%put NOTE: [ifrs9_staging] Proceso completado.;

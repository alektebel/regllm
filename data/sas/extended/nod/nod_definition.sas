/*****************************************************************************
 * nod_definition.sas
 * ===================
 * Deteccion de NOD segun EBA Art.178 (DPD>90, forbearance, insolvencia).
 * cure_status con timeline: IN_DEFAULT, PROBATION, MONITORING, PERFORMING.
 *****************************************************************************/

%put NOTE: [nod_definition] Inicio de deteccion NOD.;

/* Incluir macros */
%include "data/sas/extended/macros/regllm_macros.sas";

/* ── Datos de ejemplo ────────────────────────────────────────────────── */
DATA work.nod_input;
    LENGTH CICLO_ID $12 FORBEARANCE_FLAG INSOLVENCIA_FLAG CURE_FLAG 3;
    FORMAT FECHA_DEFAULT DATE9. FECHA_CURE DATE9. FECHA_EVAL DATE9.;
    INPUT CICLO_ID $ DPD FORBEARANCE_FLAG INSOLVENCIA_FLAG
          FECHA_DEFAULT FECHA_CURE CURE_FLAG FECHA_EVAL;
    DATALINES;
CONTR_001 95 0 0 01JAN2024 . 0 15MAR2024
CONTR_002 45 1 0 01FEB2024 . 0 15MAR2024
CONTR_003 0 0 0 . . 0 15MAR2024
CONTR_004 180 0 0 01NOV2023 01NOV2024 1 15MAR2024
CONTR_005 30 0 0 . . 0 15MAR2024
CONTR_006 365 0 1 01JAN2023 . 0 15MAR2024
CONTR_007 0 1 0 . . 0 15MAR2024
CONTR_008 120 0 0 01DEC2023 01DEC2024 1 15MAR2024
CONTR_009 60 0 0 . . 0 15MAR2024
CONTR_010 0 0 0 . . 0 15MAR2024
;
RUN;

%put NOTE: [nod_definition] &sqlobs registros cargados.;

/* ── Deteccion NOD y cure_status ────────────────────────────────────── */
DATA work.nod_result;
    SET work.nod_input;

    /* NOD flag segun EBA Art.178 */
    IF DPD > 90 OR FORBEARANCE_FLAG = 1 OR INSOLVENCIA_FLAG = 1 THEN NOD_FLAG = 1;
    ELSE NOD_FLAG = 0;

    /* Dias desde default (si aplica) */
    IF FECHA_DEFAULT ^= . THEN
        DAYS_SINCE_DEFAULT = FECHA_EVAL - FECHA_DEFAULT;
    ELSE
        DAYS_SINCE_DEFAULT = 0;

    /* Dias desde cure (si aplica) */
    IF FECHA_CURE ^= . THEN
        DAYS_SINCE_CURE = FECHA_EVAL - FECHA_CURE;
    ELSE
        DAYS_SINCE_CURE = 0;

    /* Timelines de cure_status */
    IF NOD_FLAG = 1 THEN DO;
        IF CURE_FLAG = 1 AND DAYS_SINCE_CURE >= 365 THEN
            CURE_STATUS = 'PERFORMING';
        ELSE IF CURE_FLAG = 1 AND DAYS_SINCE_CURE >= 0 THEN
            CURE_STATUS = 'PROBATION';
        ELSE IF DAYS_SINCE_DEFAULT <= 90 THEN
            CURE_STATUS = 'IN_DEFAULT';
        ELSE IF DAYS_SINCE_DEFAULT <= 180 THEN
            CURE_STATUS = 'PROBATION';
        ELSE IF DAYS_SINCE_DEFAULT <= 365 THEN
            CURE_STATUS = 'MONITORING';
        ELSE
            CURE_STATUS = 'PERFORMING';
    END;
    ELSE CURE_STATUS = 'PERFORMING';

    /* Probation complete flag */
    IF CURE_FLAG = 1 AND DAYS_SINCE_CURE >= 365 THEN
        PROBATION_COMPLETE = 1;
    ELSE
        PROBATION_COMPLETE = 0;

    /* DPD bucket */
    LENGTH DPD_BUCKET $8;
    IF DPD = 0 THEN DPD_BUCKET = '0';
    ELSE IF DPD <= 29 THEN DPD_BUCKET = '1-29';
    ELSE IF DPD <= 59 THEN DPD_BUCKET = '30-59';
    ELSE IF DPD <= 89 THEN DPD_BUCKET = '60-89';
    ELSE IF DPD <= 179 THEN DPD_BUCKET = '90-179';
    ELSE DPD_BUCKET = '180+';

    FORMAT DAYS_SINCE_DEFAULT DAYS_SINCE_CURE 8.;
RUN;

%put NOTE: [nod_definition] Deteccion NOD completada.;

/* ── Resultados ──────────────────────────────────────────────────────── */
PROC PRINT DATA=work.nod_result;
    VAR CICLO_ID DPD DPD_BUCKET FORBEARANCE_FLAG INSOLVENCIA_FLAG
        NOD_FLAG DAYS_SINCE_DEFAULT DAYS_SINCE_CURE CURE_FLAG
        CURE_STATUS PROBATION_COMPLETE;
    TITLE "NOD Definition - EBA Art.178";
RUN;

PROC FREQ DATA=work.nod_result;
    TABLES CURE_STATUS NOD_FLAG DPD_BUCKET;
    TITLE "Distribucion NOD y Cure Status";
RUN;

%put NOTE: [nod_definition] Proceso completado.;

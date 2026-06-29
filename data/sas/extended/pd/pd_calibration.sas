/*****************************************************************************
 * pd_calibration.sas
 * ===================
 * Calibracion de PD: floors por segmento, recalibracion ×1.15 para ratings
 * 1-2 antes de floor, backtesting vs observed_dr.
 *****************************************************************************/

%put NOTE: [pd_calibration] Inicio de calibracion PD.;

/* Incluir macros */
%include "data/sas/extended/macros/regllm_macros.sas";

/* ── Datos de ejemplo ────────────────────────────────────────────────── */
DATA work.calib_input;
    LENGTH CICLO_ID $12 SEGMENTO $12 RATING_ASIGNADO $6;
    INPUT CICLO_ID $ SEGMENTO $ PD_ESTIMADA RATING_ASIGNADO $ OBSERVED_DR;
    DATALINES;
CONTR_001 CORP 0.0012 AAA 0.0010
CONTR_002 RETAIL 0.0025 AA 0.0028
CONTR_003 MORTGAGE 0.0008 AAA 0.0009
CONTR_004 SME 0.0150 BB 0.0180
CONTR_005 CORP 0.0005 AAA 0.0004
CONTR_006 RETAIL 0.0080 A 0.0095
CONTR_007 MORTGAGE 0.0030 AA 0.0035
CONTR_008 SME 0.0450 B 0.0520
CONTR_009 CORP 0.0250 BBB 0.0300
CONTR_010 RETAIL 0.1200 CCC 0.1500
CONTR_011 CORP 0.0100 BBB 0.0090
CONTR_012 MORTGAGE 0.0050 A 0.0060
CONTR_013 SME 0.0600 CCC 0.0700
CONTR_014 CORP 0.0008 AAA 0.0010
CONTR_015 RETAIL 0.0350 BB 0.0400
;
RUN;

%put NOTE: [pd_calibration] &sqlobs registros cargados.;

/* ── Recalibracion y floors ──────────────────────────────────────────── */
DATA work.pd_calibrated;
    SET work.calib_input;

    /* Recalibracion ×1.15 para ratings 1-2 (AAA, AA) antes de floor */
    IF RATING_ASIGNADO IN ('AAA', 'AA') THEN
        PD_RECALIB = PD_ESTIMADA * 1.15;
    ELSE
        PD_RECALIB = PD_ESTIMADA;

    /* Floor por segmento */
    IF SEGMENTO = 'CORP' AND PD_RECALIB < 0.0005 THEN PD_FINAL = 0.0005;
    ELSE IF SEGMENTO = 'RETAIL' AND PD_RECALIB < 0.0003 THEN PD_FINAL = 0.0003;
    ELSE IF SEGMENTO = 'MORTGAGE' AND PD_RECALIB < 0.0005 THEN PD_FINAL = 0.0005;
    ELSE IF SEGMENTO = 'SME' AND PD_RECALIB < 0.0005 THEN PD_FINAL = 0.0005;
    ELSE PD_FINAL = PD_RECALIB;

    /* Backtesting: desviacion observada vs esperada */
    DESVIACION = OBSERVED_DR - PD_FINAL;
    DESVIACION_PCT = DESVIACION / (PD_FINAL + 1E-10);

    /* Flags de validacion */
    IF ABS(DESVIACION_PCT) > 0.25 THEN FLAG_BACKTEST = 'ALERTA';
    ELSE IF ABS(DESVIACION_PCT) > 0.10 THEN FLAG_BACKTEST = 'OBSERVAR';
    ELSE FLAG_BACKTEST = 'OK';

    FORMAT PD_ESTIMADA PD_RECALIB PD_FINAL OBSERVED_DR DESVIACION 8.4
           DESVIACION_PCT 8.2;
RUN;

%put NOTE: [pd_calibration] Calibracion completada.;

/* ── Resultados ──────────────────────────────────────────────────────── */
PROC PRINT DATA=work.pd_calibrated;
    VAR CICLO_ID SEGMENTO RATING_ASIGNADO PD_ESTIMADA PD_RECALIB PD_FINAL
        OBSERVED_DR DESVIACION DESVIACION_PCT FLAG_BACKTEST;
    TITLE "PD Calibration - Floors, Recalibracion y Backtesting";
RUN;

PROC MEANS DATA=work.pd_calibrated N MEAN MIN MAX;
    CLASS FLAG_BACKTEST;
    VAR DESVIACION_PCT;
    TITLE "Resumen Backtesting por Flag";
RUN;

%put NOTE: [pd_calibration] Proceso completado.;

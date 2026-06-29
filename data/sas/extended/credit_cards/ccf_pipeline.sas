/*****************************************************************************
 * ccf_pipeline.sas
 * =================
 * Pipeline de Credit Conversion Factor (CCF) para productos de credito.
 * CCF por producto, ajuste por usage_rate, revenue y riesgo por tarjeta,
 * segmentos de uso.
 *****************************************************************************/

%put NOTE: [ccf_pipeline] Inicio del pipeline CCF.;

/* Incluir macros */
%include "data/sas/extended/macros/regllm_macros.sas";

/* ── Datos de ejemplo ────────────────────────────────────────────────── */
DATA work.ccf_input;
    LENGTH CICLO_ID $12 TIPO_PRODUCTO $16 TIPO_USO $22;
    INPUT CICLO_ID $ TIPO_PRODUCTO $ TIPO_USO $
          EAD SALDO_DISPONIBLE LINEA_TOTAL TASA_INTERES
          PD_ESTIMADA LGD_ESTIMADA COMISIONES;
    DATALINES;
CONTR_001 TARJETA HEAVY_REVOLVER 50000 45000 100000 0.24 0.0250 0.45 1200
CONTR_002 LINEA_CREDITO REGULAR_REVOLVER 200000 120000 300000 0.12 0.0100 0.35 800
CONTR_003 PRESTAMO LIGHT_USER 1000000 0 0 0.08 0.0050 0.30 0
CONTR_004 TARJETA TRANSACTOR 15000 2000 50000 0.22 0.0150 0.50 300
CONTR_005 LINEA_CREDITO HEAVY_REVOLVER 400000 350000 500000 0.10 0.0080 0.40 1500
CONTR_006 DESCUENTO REGULAR_REVOLVER 80000 0 0 0.15 0.0350 0.55 200
CONTR_007 TARJETA LIGHT_USER 25000 5000 80000 0.20 0.0120 0.42 150
CONTR_008 AVAL REGULAR_REVOLVER 300000 0 0 0.09 0.0200 0.60 0
CONTR_009 TARJETA HEAVY_REVOLVER 60000 55000 120000 0.26 0.0300 0.48 1800
CONTR_010 LINEA_CREDITO TRANSACTOR 150000 20000 200000 0.11 0.0060 0.38 400
;
RUN;

%put NOTE: [ccf_pipeline] &sqlobs registros cargados.;

/* ── Calculo de CCF y ajustes ────────────────────────────────────────── */
DATA work.ccf_result;
    SET work.ccf_input;

    /* CCF base por producto */
    SELECT (TIPO_PRODUCTO);
        WHEN ('TARJETA')       CCF_BASE = 0.75;
        WHEN ('LINEA_CREDITO') CCF_BASE = 0.50;
        WHEN ('DESCUENTO')     CCF_BASE = 0.40;
        WHEN ('AVAL')          CCF_BASE = 1.00;
        OTHERWISE              CCF_BASE = 1.00;
    END;

    /* Usage rate */
    IF LINEA_TOTAL > 0 THEN USAGE_RATE = SALDO_DISPONIBLE / LINEA_TOTAL;
    ELSE USAGE_RATE = 0;

    /* CCF ajustado via macro */
    CCF_AJUSTADO = %ead_ccf(CCF_BASE, USAGE_RATE);

    /* EAD ajustado por CCF */
    EAD_CCF = EAD * CCF_AJUSTADO;

    /* Revenue por tarjeta: intereses + comisiones */
    IF TIPO_PRODUCTO = 'TARJETA' THEN DO;
        REVENUE_INTERESES = SALDO_DISPONIBLE * TASA_INTERES;
        REVENUE_TOTAL = REVENUE_INTERESES + COMISIONES;
    END;
    ELSE DO;
        REVENUE_INTERESES = EAD * TASA_INTERES;
        REVENUE_TOTAL = REVENUE_INTERESES + COMISIONES;
    END;

    /* Riesgo por tarjeta: EAD_CCF * PD * LGD */
    RIESGO_ESPERADO = EAD_CCF * PD_ESTIMADA * LGD_ESTIMADA;

    ROE = REVENUE_TOTAL / (RIESGO_ESPERADO + 1);

    FORMAT CCF_BASE CCF_AJUSTADO USAGE_RATE 8.3 EAD_CCF 12.0
           REVENUE_INTERESES REVENUE_TOTAL RIESGO_ESPERADO 12.2 ROE 8.2;
RUN;

%put NOTE: [ccf_pipeline] CCF y riesgo calculados.;

/* ── Resultados ──────────────────────────────────────────────────────── */
PROC PRINT DATA=work.ccf_result;
    VAR CICLO_ID TIPO_PRODUCTO TIPO_USO CCF_BASE USAGE_RATE CCF_AJUSTADO
        EAD EAD_CCF REVENUE_TOTAL RIESGO_ESPERADO ROE;
    TITLE "CCF Pipeline - Resultados por Producto";
RUN;

PROC MEANS DATA=work.ccf_result N MEAN SUM;
    CLASS TIPO_PRODUCTO;
    VAR EAD_CCF REVENUE_TOTAL RIESGO_ESPERADO;
    TITLE "Resumen por Tipo de Producto";
RUN;

PROC MEANS DATA=work.ccf_result N MEAN;
    CLASS TIPO_USO;
    VAR CCF_BASE CCF_AJUSTADO USAGE_RATE ROE;
    TITLE "Metricas por Segmento de Uso";
RUN;

%put NOTE: [ccf_pipeline] Proceso completado.;

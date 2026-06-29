/*****************************************************************************
 * scorecards.sas
 * ===============
 * Scorecards de solicitud (application) y comportamentales (behavioral).
 * Application: capacity(25)+collateral(20)+stability(15)+revenue(15)+industry(15)+product(10)
 * Behavioral: payment(40)+utilization(20)+tenure(20)+external(20)
 * Overall = application*0.4 + behavioral*0.6
 *****************************************************************************/

%put NOTE: [scorecards] Inicio de scorecards.;

/* Incluir macros */
%include "data/sas/extended/macros/regllm_macros.sas";

/* ── Datos de ejemplo ────────────────────────────────────────────────── */
DATA work.scorecard_input;
    LENGTH CICLO_ID $12 TIPO_PRODUCTO $16 SECTOR_ECONOMICO $20
           TIPO_USO $22;
    INPUT CICLO_ID $ SECTOR_ECONOMICO $ TIPO_PRODUCTO $
          INGRESOS GASTOS_OPER ACTIVOS_TOT PASIVOS_TOT
          VALOR_GARANTIA COLATERAL_TIPO $ ANTIGUEDAD_MESES
          INGRESOS_ANUALES EBITDA_MARGEN
          TIPO_USO $ SALDO_PROMEDIO LINEA_TOTAL PAGOS_PUNTUALES_PCT
          TENURE_MONTHS SCORE_EXTERNO;
    DATALINES;
CONTR_001 MANUFACTURA PRESTAMO 200000 150000 500000 200000 400000 FINANCIERO 60 800000 0.25 HEAVY_REVOLVER 45000 100000 0.85 48 720
CONTR_002 SERVICIOS LINEA_CREDITO 80000 50000 200000 100000 50000 PERSONAL 24 300000 0.15 REGULAR_REVOLVER 25000 80000 0.70 24 680
CONTR_003 CONSTRUCCION PRESTAMO 300000 250000 2000000 800000 1200000 HIPOTECA 120 1500000 0.20 LIGHT_USER 10000 150000 0.95 120 750
CONTR_004 COMERCIO TARJETA 100000 80000 300000 150000 0 NINGUNA 36 400000 0.10 TRANSACTOR 5000 200000 0.60 36 650
CONTR_005 TECNOLOGIA PRESTAMO 500000 350000 1500000 600000 600000 FINANCIERO 48 2000000 0.30 HEAVY_REVOLVER 60000 120000 0.90 60 800
CONTR_006 AGRICULTURA DESCUENTO 40000 35000 120000 70000 20000 PERSONAL 18 150000 0.05 REGULAR_REVOLVER 15000 50000 0.50 18 580
CONTR_007 SERVICIOS LINEA_CREDITO 180000 120000 600000 250000 300000 HIPOTECA 84 700000 0.22 LIGHT_USER 8000 80000 1.00 84 700
CONTR_008 MANUFACTURA AVAL 150000 100000 400000 200000 0 NINGUNA 30 600000 0.18 REGULAR_REVOLVER 20000 70000 0.75 30 640
CONTR_009 COMERCIO PRESTAMO 250000 180000 800000 350000 500000 FINANCIERO 72 1000000 0.28 HEAVY_REVOLVER 50000 90000 0.80 72 760
CONTR_010 TECNOLOGIA TARJETA 90000 60000 250000 100000 0 NINGUNA 24 350000 0.20 TRANSACTOR 3000 100000 0.65 24 620
;
RUN;

%put NOTE: [scorecards] &sqlobs registros cargados.;

/* ── Application Score ───────────────────────────────────────────────── */
DATA work.app_scored;
    SET work.scorecard_input;

    /* Capacity (25): EBITDA / Gastos financieros proxy */
    IF GASTOS_OPER > 0 THEN
        CAPACITY_SCORE = MIN(25, (INGRESOS - GASTOS_OPER) / GASTOS_OPER * 10);
    ELSE CAPACITY_SCORE = 25;
    CAPACITY_SCORE = MAX(0, CAPACITY_SCORE);

    /* Collateral (20) */
    IF COLATERAL_TIPO = 'HIPOTECA' THEN COLLATERAL_SCORE = 20;
    ELSE IF COLATERAL_TIPO = 'FINANCIERO' THEN COLLATERAL_SCORE = 15;
    ELSE IF COLATERAL_TIPO = 'PERSONAL' THEN COLLATERAL_SCORE = 8;
    ELSE COLLATERAL_SCORE = 0;
    IF VALOR_GARANTIA > 0 AND ACTIVOS_TOT > 0 THEN
        COLLATERAL_SCORE = COLLATERAL_SCORE * MIN(1, VALOR_GARANTIA / ACTIVOS_TOT);

    /* Stability (15): antiguedad en meses */
    STABILITY_SCORE = MIN(15, ANTIGUEDAD_MESES / 12 * 3);

    /* Revenue (15): ingresos anuales escalados */
    REVENUE_SCORE = MIN(15, INGRESOS_ANUALES / 100000 * 3);

    /* Industry (15) */
    SELECT (SECTOR_ECONOMICO);
        WHEN ('TECNOLOGIA')  INDUSTRY_SCORE = 15;
        WHEN ('MANUFACTURA') INDUSTRY_SCORE = 12;
        WHEN ('SERVICIOS')   INDUSTRY_SCORE = 10;
        WHEN ('COMERCIO')    INDUSTRY_SCORE = 8;
        WHEN ('CONSTRUCCION') INDUSTRY_SCORE = 6;
        WHEN ('AGRICULTURA') INDUSTRY_SCORE = 4;
        OTHERWISE            INDUSTRY_SCORE = 8;
    END;

    /* Product (10) */
    SELECT (TIPO_PRODUCTO);
        WHEN ('PRESTAMO')     PRODUCT_SCORE = 10;
        WHEN ('LINEA_CREDITO') PRODUCT_SCORE = 8;
        WHEN ('DESCUENTO')    PRODUCT_SCORE = 6;
        WHEN ('AVAL')         PRODUCT_SCORE = 4;
        WHEN ('TARJETA')      PRODUCT_SCORE = 5;
        OTHERWISE             PRODUCT_SCORE = 5;
    END;

    /* Application score total */
    APP_SCORE = CAPACITY_SCORE + COLLATERAL_SCORE + STABILITY_SCORE +
                REVENUE_SCORE + INDUSTRY_SCORE + PRODUCT_SCORE;
    APP_SCORE = MIN(100, MAX(0, APP_SCORE));

    FORMAT CAPACITY_SCORE COLLATERAL_SCORE STABILITY_SCORE REVENUE_SCORE
           INDUSTRY_SCORE PRODUCT_SCORE APP_SCORE 8.1;
RUN;

%put NOTE: [scorecards] Application score calculado.;

/* ── Behavioral Score ────────────────────────────────────────────────── */
DATA work.beh_scored;
    SET work.app_scored;

    /* Payment (40): puntualidad en pagos */
    PAYMENT_SCORE = PAGOS_PUNTUALES_PCT * 40;

    /* Utilization (20): uso de linea */
    IF LINEA_TOTAL > 0 THEN
        UTILIZATION_SCORE = (1 - MIN(1, SALDO_PROMEDIO / LINEA_TOTAL)) * 20;
    ELSE UTILIZATION_SCORE = 20;

    /* Tenure (20): antiguedad */
    TENURE_SCORE = MIN(20, TENURE_MONTHS / 60 * 20);

    /* External (20): score externo normalizado */
    EXTERNAL_SCORE = MIN(20, SCORE_EXTERNO / 1000 * 20);

    /* Behavioral score total */
    BEH_SCORE = PAYMENT_SCORE + UTILIZATION_SCORE + TENURE_SCORE + EXTERNAL_SCORE;
    BEH_SCORE = MIN(100, MAX(0, BEH_SCORE));

    /* Overall score */
    OVERALL_SCORE = APP_SCORE * 0.40 + BEH_SCORE * 0.60;

    /* PD desde overall score */
    PD_SCORECARD = %pd_from_score(OVERALL_SCORE);
    RATING_SCORECARD = %rating_from_pd(PD_SCORECARD);

    FORMAT PAYMENT_SCORE UTILIZATION_SCORE TENURE_SCORE EXTERNAL_SCORE
           BEH_SCORE OVERALL_SCORE 8.1 PD_SCORECARD 8.4;
RUN;

%put NOTE: [scorecards] Behavioral y Overall score calculados.;

/* ── Resultados ──────────────────────────────────────────────────────── */
PROC PRINT DATA=work.beh_scored;
    VAR CICLO_ID APP_SCORE BEH_SCORE OVERALL_SCORE PD_SCORECARD RATING_SCORECARD;
    TITLE "Scorecards - Resultados Finales";
RUN;

PROC MEANS DATA=work.beh_scored N MEAN STD MIN MAX;
    VAR APP_SCORE BEH_SCORE OVERALL_SCORE PD_SCORECARD;
    TITLE "Estadisticas Descriptivas de Scores";
RUN;

%put NOTE: [scorecards] Proceso completado.;

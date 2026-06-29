/*****************************************************************************
 * management_reports.sas
 * =======================
 * Reportes gerenciales: portfolio summary, rating distribution, ECL by product,
 * top 10 obligors, credit card metrics, stage heatmap, scenario impact,
 * NPL aging buckets, vintage analysis.
 *****************************************************************************/

%put NOTE: [management_reports] Inicio de reportes gerenciales.;

/* Incluir macros */
%include "data/sas/extended/macros/regllm_macros.sas";

/* ── Datos de ejemplo ────────────────────────────────────────────────── */
DATA work.mgmt_input;
    LENGTH CICLO_ID $12 SEGMENTO $12 TIPO_PRODUCTO $16
           RATING_ASIGNADO $6 ANIO_ORIGINACION 8;
    INPUT CICLO_ID $ SEGMENTO $ TIPO_PRODUCTO $ RATING_ASIGNADO $
          STAGE_IFRS9 EAD PD_ESTIMADA LGD_ESTIMADA ECL_PROVISION
          ANIO_ORIGINACION DIAS_MORA INGRESOS_FINANCIEROS;
    DATALINES;
CONTR_001 CORP PRESTAMO AAA 1 1000000 0.0012 0.35 420 2020 0 50000
CONTR_002 RETAIL LINEA_CREDITO A 2 150000 0.0250 0.45 1688 2021 35 12000
CONTR_003 MORTGAGE PRESTAMO AAA 1 1800000 0.0008 0.25 360 2019 0 72000
CONTR_004 SME TARJETA B 3 300000 0.0450 0.60 8100 2022 95 18000
CONTR_005 CORP PRESTAMO AA 1 4000000 0.0050 0.40 8000 2020 0 200000
CONTR_006 RETAIL DESCUENTO BB 2 50000 0.0800 0.55 2200 2023 60 5000
CONTR_007 MORTGAGE PRESTAMO AA 1 1600000 0.0030 0.28 1344 2018 0 64000
CONTR_008 SME LINEA_CREDITO CCC 3 200000 0.1200 0.65 15600 2022 150 16000
CONTR_009 CORP PRESTAMO BBB 2 2500000 0.0100 0.38 9500 2021 25 125000
CONTR_010 RETAIL TARJETA A 1 100000 0.0050 0.42 210 2023 0 8000
CONTR_011 CORP PRESTAMO AA 1 3000000 0.0030 0.30 2700 2020 0 150000
CONTR_012 RETAIL LINEA_CREDITO BB 2 200000 0.0500 0.50 5000 2021 45 18000
CONTR_013 MORTGAGE PRESTAMO AAA 1 2000000 0.0010 0.25 500 2019 0 80000
CONTR_014 SME TARJETA B 3 350000 0.0600 0.55 11550 2022 120 21000
CONTR_015 CORP PRESTAMO A 1 3500000 0.0020 0.35 2450 2021 0 175000
;
RUN;

%put NOTE: [management_reports] &sqlobs registros cargados.;

/* ── 1. Portfolio Summary por Segmento ───────────────────────────────── */
PROC SUMMARY DATA=work.mgmt_input NWAY;
    CLASS SEGMENTO;
    VAR EAD PD_ESTIMADA LGD_ESTIMADA ECL_PROVISION DIAS_MORA INGRESOS_FINANCIEROS;
    OUTPUT OUT=work.portfolio_summary
        N=n_contracts SUM(EAD ECL_PROVISION INGRESOS_FINANCIEROS)=
        MEAN(PD_ESTIMADA LGD_ESTIMADA DIAS_MORA)=;
RUN;

DATA work.portfolio_summary;
    SET work.portfolio_summary;
    COVERAGE_PCT = ECL_PROVISION / (EAD + 1E-10) * 100;
    REN_INGRESOS = INGRESOS_FINANCIEROS;
    LABEL n_contracts = 'N Contratos';
    FORMAT EAD ECL_PROVISION INGRESOS_FINANCIEROS 14.0
           PD_ESTIMADA LGD_ESTIMADA 8.4 COVERAGE_PCT 8.2;
RUN;

%put NOTE: [management_reports] Portfolio summary calculado.;

/* ── 2. Rating Distribution ──────────────────────────────────────────── */
PROC SUMMARY DATA=work.mgmt_input NWAY;
    CLASS RATING_ASIGNADO;
    VAR EAD ECL_PROVISION;
    OUTPUT OUT=work.rating_dist N=n_contracts SUM=;
RUN;

%put NOTE: [management_reports] Rating distribution calculada.;

/* ── 3. ECL by Product Type ──────────────────────────────────────────── */
PROC SUMMARY DATA=work.mgmt_input NWAY;
    CLASS TIPO_PRODUCTO;
    VAR EAD ECL_PROVISION;
    OUTPUT OUT=work.ecl_by_product N=n_contracts SUM= MEAN(ECL_PROVISION)=ECL_PROM;
RUN;

%put NOTE: [management_reports] ECL por producto calculado.;

/* ── 4. Top 10 Obligors by Exposure ──────────────────────────────────── */
PROC SORT DATA=work.mgmt_input OUT=work.top10_obligors;
    BY DESCENDING EAD;
RUN;

DATA work.top10_obligors;
    SET work.top10_obligors (OBS=10);
    RANK = _N_;
RUN;

%put NOTE: [management_reports] Top 10 obligors generado.;

/* ── 5. Credit Card Performance ──────────────────────────────────────── */
DATA work.credit_cards;
    SET work.mgmt_input;
    WHERE TIPO_PRODUCTO = 'TARJETA';
    RENTABILIDAD = INGRESOS_FINANCIEROS / (EAD + 1E-10);
    INTENSIDAD_ECL = ECL_PROVISION / (EAD + 1E-10) * 10000; /* bps */
RUN;

%put NOTE: [management_reports] Credit card metrics calculadas.;

/* ── 6. Stage × Segmento Heatmap ────────────────────────────────────── */
PROC SUMMARY DATA=work.mgmt_input NWAY;
    CLASS STAGE_IFRS9 SEGMENTO;
    VAR EAD ECL_PROVISION;
    OUTPUT OUT=work.stage_segment_heat N=n_contracts SUM=;
RUN;

%put NOTE: [management_reports] Stage × Segmento heatmap generado.;

/* ── 7. Scenario Impact (Base / Adverse) ─────────────────────────────── */
DATA work.scenario_impact;
    SET work.mgmt_input;

    /* Base scenario */
    ECL_BASE = ECL_PROVISION;

    /* Adverse scenario: PD × 2 */
    PD_ADVERSE = MIN(1, PD_ESTIMADA * 2);
    IF STAGE_IFRS9 = 1 THEN
        ECL_ADVERSE = PD_ADVERSE * LGD_ESTIMADA * EAD;
    ELSE IF STAGE_IFRS9 = 3 THEN
        ECL_ADVERSE = 1.0 * LGD_ESTIMADA * EAD * MIN(5,3);
    ELSE
        ECL_ADVERSE = PD_ADVERSE * LGD_ESTIMADA * EAD * 2;

    IMPACTO = ECL_ADVERSE - ECL_BASE;
    IMPACTO_PCT = IMPACTO / (ECL_BASE + 1E-10) * 100;
RUN;

PROC SUMMARY DATA=work.scenario_impact NWAY;
    CLASS STAGE_IFRS9;
    VAR ECL_BASE ECL_ADVERSE IMPACTO IMPACTO_PCT;
    OUTPUT OUT=work.scenario_summary SUM(ECL_BASE ECL_ADVERSE IMPACTO)=
           MEAN(IMPACTO_PCT)=;
RUN;

%put NOTE: [management_reports] Scenario impact calculado.;

/* ── 8. NPL Aging Buckets ───────────────────────────────────────────── */
DATA work.npl_aging;
    SET work.mgmt_input;
    LENGTH NPL_BUCKET $12;
    IF DIAS_MORA = 0 THEN NPL_BUCKET = '0';
    ELSE IF DIAS_MORA <= 29 THEN NPL_BUCKET = '1-29';
    ELSE IF DIAS_MORA <= 59 THEN NPL_BUCKET = '30-59';
    ELSE IF DIAS_MORA <= 89 THEN NPL_BUCKET = '60-89';
    ELSE IF DIAS_MORA <= 179 THEN NPL_BUCKET = '90-179';
    ELSE NPL_BUCKET = '180+';
RUN;

PROC SUMMARY DATA=work.npl_aging NWAY;
    CLASS NPL_BUCKET;
    VAR EAD ECL_PROVISION;
    OUTPUT OUT=work.npl_aging_summary N=n_contracts SUM=;
RUN;

%put NOTE: [management_reports] NPL aging buckets generados.;

/* ── 9. Vintage Analysis ─────────────────────────────────────────────── */
PROC SUMMARY DATA=work.mgmt_input NWAY;
    CLASS ANIO_ORIGINACION;
    VAR EAD ECL_PROVISION DIAS_MORA;
    OUTPUT OUT=work.vintage_analysis N=n_contracts SUM(EAD ECL_PROVISION)=
           MEAN(DIAS_MORA)=MORA_PROM;
RUN;

%put NOTE: [management_reports] Vintage analysis generado.;

/* ── Resultados ──────────────────────────────────────────────────────── */
TITLE "1. Portfolio Summary por Segmento";
PROC PRINT DATA=work.portfolio_summary;
    VAR SEGMENTO n_contracts EAD PD_ESTIMADA LGD_ESTIMADA
        ECL_PROVISION COVERAGE_PCT INGRESOS_FINANCIEROS;
RUN;

TITLE "2. Rating Distribution";
PROC PRINT DATA=work.rating_dist;
    VAR RATING_ASIGNADO n_contracts EAD ECL_PROVISION;
RUN;

TITLE "3. ECL by Product Type";
PROC PRINT DATA=work.ecl_by_product;
    VAR TIPO_PRODUCTO n_contracts EAD ECL_PROVISION ECL_PROM;
RUN;

TITLE "4. Top 10 Obligors by Exposure";
PROC PRINT DATA=work.top10_obligors;
    VAR RANK CICLO_ID SEGMENTO RATING_ASIGNADO EAD ECL_PROVISION;
    WHERE RANK <= 10;
RUN;

TITLE "5. Credit Card Performance";
PROC PRINT DATA=work.credit_cards;
    VAR CICLO_ID EAD PD_ESTIMADA ECL_PROVISION INGRESOS_FINANCIEROS
        RENTABILIDAD INTENSIDAD_ECL;
RUN;

TITLE "6. Stage × Segmento Heatmap (EAD Total)";
PROC PRINT DATA=work.stage_segment_heat;
    VAR STAGE_IFRS9 SEGMENTO n_contracts EAD ECL_PROVISION;
RUN;

TITLE "7. Scenario Impact (Base vs Adverse)";
PROC PRINT DATA=work.scenario_summary;
    VAR STAGE_IFRS9 ECL_BASE ECL_ADVERSE IMPACTO IMPACTO_PCT;
RUN;

TITLE "8. NPL Aging Buckets";
PROC PRINT DATA=work.npl_aging_summary;
    VAR NPL_BUCKET n_contracts EAD ECL_PROVISION;
RUN;

TITLE "9. Vintage Analysis (por Anio de Originacion)";
PROC PRINT DATA=work.vintage_analysis;
    VAR ANIO_ORIGINACION n_contracts EAD ECL_PROVISION MORA_PROM;
RUN;

%put NOTE: [management_reports] Todos los reportes generados correctamente.;
%put NOTE: [management_reports] Proceso completado.;

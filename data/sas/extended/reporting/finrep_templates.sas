/*****************************************************************************
 * finrep_templates.sas
 * =====================
 * Plantillas FINREP: Gross Carrying Amount por stage, Loss Allowance,
 * NPL coverage ratio, Impaired/Non-performing (F.40.10 movement schedule).
 *****************************************************************************/

%put NOTE: [finrep_templates] Inicio de plantillas FINREP.;

/* Incluir macros */
%include "data/sas/extended/macros/regllm_macros.sas";

/* ── Datos de ejemplo ────────────────────────────────────────────────── */
DATA work.finrep_input;
    LENGTH CICLO_ID $12 SEGMENTO $12 TIPO_PRODUCTO $16
           IMPAIRED_FLAG NPL_FLAG $3;
    INPUT CICLO_ID $ SEGMENTO $ TIPO_PRODUCTO $
          STAGE_IFRS9 EAD PD_ESTIMADA LGD_ESTIMADA YEARS_TO_MATURITY
          IMPAIRED_FLAG $ NPL_FLAG $;
    DATALINES;
CONTR_001 CORP PRESTAMO 1 1000000 0.0012 0.35 4.5 No No
CONTR_002 RETAIL LINEA_CREDITO 2 150000 0.0250 0.45 2.0 Si Si
CONTR_003 MORTGAGE PRESTAMO 1 1800000 0.0008 0.25 8.0 No No
CONTR_004 SME TARJETA 3 300000 0.0450 0.60 1.5 Si Si
CONTR_005 CORP PRESTAMO 1 4000000 0.0050 0.40 3.0 No No
CONTR_006 RETAIL DESCUENTO 2 50000 0.0800 0.55 2.5 Si No
CONTR_007 MORTGAGE PRESTAMO 1 1600000 0.0030 0.28 10.0 No No
CONTR_008 SME LINEA_CREDITO 3 200000 0.1200 0.65 0.5 Si Si
CONTR_009 CORP PRESTAMO 2 2500000 0.0100 0.38 5.0 Si Si
CONTR_010 RETAIL TARJETA 1 100000 0.0050 0.42 1.0 No No
;
RUN;

%put NOTE: [finrep_templates] &sqlobs registros cargados.;

/* ── Calculo FINREP ──────────────────────────────────────────────────── */
DATA work.finrep_result;
    SET work.finrep_input;

    /* Loss Allowance (ECL total) */
    LIFETIME_YEARS = MIN(YEARS_TO_MATURITY, 5);
    IF STAGE_IFRS9 = 3 THEN PD_ECL = 1.0;
    ELSE PD_ECL = PD_ESTIMADA;

    IF STAGE_IFRS9 = 1 THEN
        LOSS_ALLOWANCE = PD_ECL * LGD_ESTIMADA * EAD;
    ELSE
        LOSS_ALLOWANCE = PD_ECL * LGD_ESTIMADA * EAD * LIFETIME_YEARS;

    /* Gross Carrying Amount (GCA) = EAD */
    GROSS_CARRYING_AMT = EAD;

    /* Net Carrying Amount */
    NET_CARRYING_AMT = GROSS_CARRYING_AMT - LOSS_ALLOWANCE;

    /* NPL Coverage Ratio */
    IF NPL_FLAG = 'Si' AND GROSS_CARRYING_AMT > 0 THEN
        NPL_COVERAGE_RATIO = LOSS_ALLOWANCE / GROSS_CARRYING_AMT;
    ELSE NPL_COVERAGE_RATIO = .;

    /* Impaired / Non-performing breakdown */
    IF IMPAIRED_FLAG = 'Si' THEN IMPAIRED_AMOUNT = GROSS_CARRYING_AMT;
    ELSE IMPAIRED_AMOUNT = 0;
    IF NPL_FLAG = 'Si' THEN NPL_AMOUNT = GROSS_CARRYING_AMT;
    ELSE NPL_AMOUNT = 0;

    FORMAT LOSS_ALLOWANCE GROSS_CARRYING_AMT NET_CARRYING_AMT
           IMPAIRED_AMOUNT NPL_AMOUNT 14.2 NPL_COVERAGE_RATIO 8.5;
RUN;

%put NOTE: [finrep_templates] Datos FINREP calculados.;

/* ── Movement Schedule F.40.10 ────────────────────────────────────────── */
PROC SUMMARY DATA=work.finrep_result NWAY;
    CLASS STAGE_IFRS9;
    VAR GROSS_CARRYING_AMT LOSS_ALLOWANCE NET_CARRYING_AMT;
    OUTPUT OUT=work.finrep_mvmnt SUM=;
RUN;

%put NOTE: [finrep_templates] Movement schedule F.40.10 generado.;

/* ── Resultados ──────────────────────────────────────────────────────── */
PROC PRINT DATA=work.finrep_result;
    VAR CICLO_ID SEGMENTO STAGE_IFRS9 GROSS_CARRYING_AMT LOSS_ALLOWANCE
        NET_CARRYING_AMT IMPAIRED_FLAG IMPAIRED_AMOUNT
        NPL_FLAG NPL_AMOUNT NPL_COVERAGE_RATIO;
    TITLE "FINREP Template - GCA, Loss Allowance y NPL";
RUN;

PROC PRINT DATA=work.finrep_mvmnt;
    VAR STAGE_IFRS9 GROSS_CARRYING_AMT LOSS_ALLOWANCE NET_CARRYING_AMT;
    TITLE "FINREP F.40.10 - Movement Schedule por Stage";
RUN;

PROC MEANS DATA=work.finrep_result N MEAN SUM;
    CLASS IMPAIRED_FLAG NPL_FLAG;
    VAR GROSS_CARRYING_AMT LOSS_ALLOWANCE NPL_COVERAGE_RATIO;
    TITLE "Resumen por Estado de Deterioro y NPL";
RUN;

%put NOTE: [finrep_templates] Proceso completado.;

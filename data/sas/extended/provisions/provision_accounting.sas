/*****************************************************************************
 * provision_accounting.sas
 * =========================
 * Contabilidad de provisiones: asientos contables, P&L charge,
 * movement schedule por stage, formato disclosure IFRS9.
 *****************************************************************************/

%put NOTE: [provision_accounting] Inicio de contabilidad de provisiones.;

/* Incluir macros */
%include "data/sas/extended/macros/regllm_macros.sas";

/* ── Datos de ejemplo ────────────────────────────────────────────────── */
DATA work.prov_input;
    LENGTH CICLO_ID $12 SEGMENTO $12 TIPO_PRODUCTO $16;
    INPUT CICLO_ID $ SEGMENTO $ TIPO_PRODUCTO $
          STAGE_IFRS9 EAD PD_ESTIMADA LGD_ESTIMADA YEARS_TO_MATURITY;
    DATALINES;
CONTR_001 CORP PRESTAMO 1 1000000 0.0012 0.35 4.5
CONTR_002 RETAIL LINEA_CREDITO 2 150000 0.0250 0.45 2.0
CONTR_003 MORTGAGE PRESTAMO 1 1800000 0.0008 0.25 8.0
CONTR_004 SME TARJETA 3 300000 0.0450 0.60 1.5
CONTR_005 CORP PRESTAMO 1 4000000 0.0050 0.40 3.0
CONTR_006 RETAIL DESCUENTO 2 50000 0.0800 0.55 2.5
CONTR_007 MORTGAGE PRESTAMO 1 1600000 0.0030 0.28 10.0
CONTR_008 SME LINEA_CREDITO 3 200000 0.1200 0.65 0.5
CONTR_009 CORP PRESTAMO 2 2500000 0.0100 0.38 5.0
CONTR_010 RETAIL TARJETA 1 100000 0.0050 0.42 1.0
;
RUN;

%put NOTE: [provision_accounting] &sqlobs registros cargados.;

/* ── Calculo de provisiones y asientos ────────────────────────────────── */
DATA work.provision_accounting;
    SET work.prov_input;

    /* PD y LGD para ECL */
    LIFETIME_YEARS = MIN(YEARS_TO_MATURITY, 5);

    IF STAGE_IFRS9 = 3 THEN PD_ECL = 1.0;
    ELSE PD_ECL = PD_ESTIMADA;

    /* Closing provision (final del periodo) */
    IF STAGE_IFRS9 = 1 THEN
        PROVISION_CLOSING = PD_ECL * LGD_ESTIMADA * EAD;
    ELSE
        PROVISION_CLOSING = PD_ECL * LGD_ESTIMADA * EAD * LIFETIME_YEARS;

    /* Opening provision = 80% de closing (simulado) */
    PROVISION_OPENING = PROVISION_CLOSING * 0.80;

    /* P&L charge */
    PNL_CHARGE = PROVISION_CLOSING - PROVISION_OPENING;

    /* Movement schedule */
    IF PNL_CHARGE >= 0 THEN MOVEMENT_TYPE = 'INCREASE';
    ELSE MOVEMENT_TYPE = 'DECREASE';

    /* IFRS9 disclosure format fields */
    GROSS_CARRYING_AMOUNT = EAD;
    LOSS_ALLOWANCE = PROVISION_CLOSING;
    COVERAGE_RATIO = PROVISION_CLOSING / (EAD + 1E-10);

    FORMAT PROVISION_OPENING PROVISION_CLOSING PNL_CHARGE 14.2
           GROSS_CARRYING_AMOUNT LOSS_ALLOWANCE 14.2 COVERAGE_RATIO 8.5;
RUN;

%put NOTE: [provision_accounting] Asientos contables calculados.;

/* ── Agregacion por Stage ────────────────────────────────────────────── */
PROC SUMMARY DATA=work.provision_accounting NWAY;
    CLASS STAGE_IFRS9;
    VAR PROVISION_OPENING PROVISION_CLOSING PNL_CHARGE GROSS_CARRYING_AMOUNT LOSS_ALLOWANCE;
    OUTPUT OUT=work.prov_by_stage SUM=;
RUN;

%put NOTE: [provision_accounting] Movimiento por stage calculado.;

/* ── Resultados ──────────────────────────────────────────────────────── */
PROC PRINT DATA=work.provision_accounting;
    VAR CICLO_ID SEGMENTO STAGE_IFRS9 EAD PD_ECL LGD_ESTIMADA
        PROVISION_OPENING PROVISION_CLOSING PNL_CHARGE MOVEMENT_TYPE
        GROSS_CARRYING_AMOUNT LOSS_ALLOWANCE COVERAGE_RATIO;
    TITLE "Provision Accounting - Asientos Contables";
RUN;

PROC PRINT DATA=work.prov_by_stage;
    VAR STAGE_IFRS9 PROVISION_OPENING PROVISION_CLOSING PNL_CHARGE
        GROSS_CARRYING_AMOUNT LOSS_ALLOWANCE;
    TITLE "IFRS9 Disclosure - Movement Schedule por Stage";
RUN;

%put NOTE: [provision_accounting] Proceso completado.;

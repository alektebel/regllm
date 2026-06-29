/*****************************************************************************
 * corep_templates.sas
 * ====================
 * Plantillas COREP: K_IRB simplificado, RWA, Capital P1, EL shortfall,
 * RWA density, rating distribution (formato C.100.10).
 *****************************************************************************/

%put NOTE: [corep_templates] Inicio de plantillas COREP.;

/* Incluir macros */
%include "data/sas/extended/macros/regllm_macros.sas";

/* ── Datos de ejemplo ────────────────────────────────────────────────── */
DATA work.corep_input;
    LENGTH CICLO_ID $12 SEGMENTO $12 RATING_ASIGNADO $6;
    INPUT CICLO_ID $ SEGMENTO $ RATING_ASIGNADO $
          PD_ESTIMADA LGD_ESTIMADA EAD MADUREZ
          ECL_PROVISION;
    DATALINES;
CONTR_001 CORP AAA 0.0012 0.35 1000000 2.5 420
CONTR_002 RETAIL A 0.0250 0.45 150000 1.5 1688
CONTR_003 MORTGAGE AAA 0.0008 0.25 1800000 5.0 360
CONTR_004 SME B 0.0450 0.60 300000 1.0 8100
CONTR_005 CORP AA 0.0050 0.40 4000000 3.0 8000
CONTR_006 RETAIL BB 0.0800 0.55 50000 2.0 2200
CONTR_007 MORTGAGE AA 0.0030 0.28 1600000 10.0 1344
CONTR_008 SME CCC 0.1200 0.65 200000 0.5 15600
CONTR_009 CORP BBB 0.0100 0.38 2500000 4.0 9500
CONTR_010 RETAIL A 0.0050 0.42 100000 1.0 210
;
RUN;

%put NOTE: [corep_templates] &sqlobs registros cargados.;

/* ── Calculo IRB K, RWA, Capital ─────────────────────────────────────── */
DATA work.corep_result;
    SET work.corep_input;

    /* Correlacion por segmento */
    SELECT (SEGMENTO);
        WHEN ('CORP')    CORRELACION = 0.24;
        WHEN ('RETAIL')  CORRELACION = 0.15;
        WHEN ('MORTGAGE') CORRELACION = 0.15;
        WHEN ('SME')     CORRELACION = 0.20;
        OTHERWISE        CORRELACION = 0.24;
    END;

    /* K_IRB simplificado: sqrt(PD)*0.06 + PD*0.5 */
    K_IRB = (SQRT(PD_ESTIMADA) * 0.06 + PD_ESTIMADA * 0.5)
            * (1 + (MADUREZ - 2.5) * (1 - 2 * CORRELACION));
    K_IRB = MAX(0, K_IRB);

    /* RWA = K * 1.06 * 12.5 * EAD */
    RWA = K_IRB * 1.06 * 12.5 * EAD;

    /* Capital P1 = RWA * 0.08 */
    CAPITAL_P1 = RWA * 0.08;

    /* Expected Loss = PD * LGD * EAD */
    EL = PD_ESTIMADA * LGD_ESTIMADA * EAD;

    /* EL shortfall = ECL - provision */
    EL_SHORTFALL = EL - ECL_PROVISION;

    /* RWA density */
    IF EAD > 0 THEN RWA_DENSITY = RWA / EAD;
    ELSE RWA_DENSITY = 0;

    FORMAT PD_ESTIMADA 8.4 K_IRB 8.6 RWA 14.0 CAPITAL_P1 12.2
           EL 14.2 EL_SHORTFALL 14.2 RWA_DENSITY 8.4;
RUN;

%put NOTE: [corep_templates] K_IRB, RWA y Capital calculados.;

/* ── Rating Distribution (C.100.10) ──────────────────────────────────── */
PROC SUMMARY DATA=work.corep_result NWAY;
    CLASS RATING_ASIGNADO;
    VAR EAD RWA CAPITAL_P1 EL;
    OUTPUT OUT=work.rating_dist SUM=;
RUN;

%put NOTE: [corep_templates] Distribucion por rating calculada.;

/* ── Resultados ──────────────────────────────────────────────────────── */
PROC PRINT DATA=work.corep_result;
    VAR CICLO_ID SEGMENTO RATING_ASIGNADO PD_ESTIMADA LGD_ESTIMADA EAD
        CORRELACION K_IRB RWA CAPITAL_P1 EL EL_SHORTFALL RWA_DENSITY;
    TITLE "COREP Template - K_IRB, RWA y Capital";
RUN;

PROC PRINT DATA=work.rating_dist;
    VAR RATING_ASIGNADO EAD RWA CAPITAL_P1 EL;
    TITLE "COREP C.100.10 - Rating Distribution";
RUN;

PROC MEANS DATA=work.corep_result N MEAN SUM;
    CLASS SEGMENTO;
    VAR EAD RWA CAPITAL_P1 RWA_DENSITY;
    TITLE "Resumen por Segmento";
RUN;

%put NOTE: [corep_templates] Proceso completado.;

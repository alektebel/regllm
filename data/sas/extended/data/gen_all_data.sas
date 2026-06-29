/*****************************************************************************
 * gen_all_data.sas
 * =================
 * Generador deterministico de datos de prueba para el pipeline IRB extendido.
 * Produce: contratos, obligors, tarjetas, colaterales, historiales de pago,
 * defaults, ratios financieros.
 *****************************************************************************/

%put NOTE: [gen_all_data] Inicio de generacion de datos.;

/* ── 1. Contratos (50) ──────────────────────────────────────────────── */
DATA work.contratos;
    LENGTH CICLO_ID $12 SEGMENTO $12 TIPO_PRODUCTO $16
           SW_FUSION 3 STAGE_IFRS9 8;
    FORMAT FECHA_ORIGEN DATE9. VENCIMIENTO DATE9.;
    DO i = 1 TO 50;
        CICLO_ID = 'CONTR_' !! PUT(i, Z3.);
        SELECT (MOD(i, 4));
            WHEN (0) SEGMENTO = 'CORP';
            WHEN (1) SEGMENTO = 'RETAIL';
            WHEN (2) SEGMENTO = 'MORTGAGE';
            WHEN (3) SEGMENTO = 'SME';
        END;
        PD_ESTIMADA  = 0.001 + (i / 50) * 0.149;
        LGD_ESTIMADA = 0.10 + (i / 50) * 0.65;
        EAD          = 10000 + (i * 100000);
        STAGE_IFRS9  = 1 + MOD(i, 3);
        SW_FUSION    = (i IN (10, 25, 40));
        FECHA_ORIGEN = '01JAN2020'D + (i * 30);
        VENCIMIENTO  = FECHA_ORIGEN + 1825 + (i * 10);
        TASA_INTERES = 0.02 + (i * 0.005);
        SELECT (MOD(i, 5));
            WHEN (0) TIPO_PRODUCTO = 'PRESTAMO';
            WHEN (1) TIPO_PRODUCTO = 'LINEA_CREDITO';
            WHEN (2) TIPO_PRODUCTO = 'TARJETA';
            WHEN (3) TIPO_PRODUCTO = 'DESCUENTO';
            WHEN (4) TIPO_PRODUCTO = 'AVAL';
        END;
        OUTPUT;
    END;
    DROP i;
RUN;

%put NOTE: [gen_all_data] 50 contratos generados.;

/* ── 2. Obligors (38) ───────────────────────────────────────────────── */
DATA work.obligors;
    LENGTH OBLIG_ID $12 SECTOR_ECONOMICO $20 COD_ALTERNATIVE $10 RATING_EXTERNO $6;
    DO i = 1 TO 38;
        OBLIG_ID = 'OBLIG_' !! PUT(i, Z3.);
        SELECT (MOD(i, 6));
            WHEN (0) SECTOR_ECONOMICO = 'MANUFACTURA';
            WHEN (1) SECTOR_ECONOMICO = 'SERVICIOS';
            WHEN (2) SECTOR_ECONOMICO = 'CONSTRUCCION';
            WHEN (3) SECTOR_ECONOMICO = 'AGRICULTURA';
            WHEN (4) SECTOR_ECONOMICO = 'COMERCIO';
            WHEN (5) SECTOR_ECONOMICO = 'TECNOLOGIA';
        END;
        IF MOD(i, 4) = 0 THEN COD_ALTERNATIVE = 'ALT_' !! PUT(i, Z3.);
        ELSE COD_ALTERNATIVE = '';
        SELECT (MOD(i, 8));
            WHEN (0) RATING_EXTERNO = 'AAA';
            WHEN (1) RATING_EXTERNO = 'AA';
            WHEN (2) RATING_EXTERNO = 'A';
            WHEN (3) RATING_EXTERNO = 'BBB';
            WHEN (4) RATING_EXTERNO = 'BB';
            WHEN (5) RATING_EXTERNO = 'B';
            WHEN (6) RATING_EXTERNO = 'CCC';
            WHEN (7) RATING_EXTERNO = 'Default';
        END;
        OUTPUT;
    END;
    DROP i;
RUN;

%put NOTE: [gen_all_data] 38 obligors generados.;

/* ── 3. Tarjetas de credito (8) ─────────────────────────────────────── */
DATA work.tarjetas;
    LENGTH TARJ_ID $12 TIPO_USO $22;
    FORMAT FECHA_EMISION DATE9.;
    DO i = 1 TO 8;
        TARJ_ID = 'TARJ_' !! PUT(i, Z3.);
        CUPON_TOTAL     = 50000 + (i * 100000);
        SALDO_DISPONIBLE = CUPON_TOTAL * (0.2 + (i * 0.05));
        TASA_INTERES     = 0.18 + (i * 0.02);
        SELECT (MOD(i, 4));
            WHEN (0) TIPO_USO = 'HEAVY_REVOLVER';
            WHEN (1) TIPO_USO = 'REGULAR_REVOLVER';
            WHEN (2) TIPO_USO = 'LIGHT_USER';
            WHEN (3) TIPO_USO = 'TRANSACTOR';
        END;
        FECHA_EMISION = '01JAN2021'D + (i * 45);
        OUTPUT;
    END;
    DROP i;
RUN;

%put NOTE: [gen_all_data] 8 tarjetas generadas.;

/* ── 4. Colaterales (21) ────────────────────────────────────────────── */
DATA work.colaterales;
    LENGTH COLA_ID $12 COLATERAL_TIPO $12;
    FORMAT FECHA_TASACION DATE9.;
    DO i = 1 TO 21;
        COLA_ID = 'COLA_' !! PUT(i, Z3.);
        SELECT (MOD(i, 4));
            WHEN (0) COLATERAL_TIPO = 'HIPOTECA';
            WHEN (1) COLATERAL_TIPO = 'FINANCIERO';
            WHEN (2) COLATERAL_TIPO = 'PERSONAL';
            WHEN (3) COLATERAL_TIPO = 'NINGUNA';
        END;
        VALOR_GARANTIA = 50000 + (i * 25000);
        FECHA_TASACION = '01JAN2022'D + (i * 30);
        OUTPUT;
    END;
    DROP i;
RUN;

%put NOTE: [gen_all_data] 21 colaterales generados.;

/* ── 5. Historiales de pago (50) ────────────────────────────────────── */
DATA work.historial_pagos;
    LENGTH CICLO_ID $12;
    FORMAT FECHA_PAGO DATE9.;
    DO i = 1 TO 50;
        CICLO_ID = 'CONTR_' !! PUT(i, Z3.);
        DO j = 1 TO 12;
            FECHA_PAGO    = '01JAN2023'D + ((j - 1) * 30) + (i * 2);
            DIAS_MORA     = MAX(0, INT(RANUNI(i + j) * 180) - (i / 2));
            IMPORTE_PAGADO = 500 + INT(RANUNI(i * j) * 5000);
            OUTPUT;
        END;
    END;
    DROP i j;
RUN;

%put NOTE: [gen_all_data] 50 historiales de pago generados.;

/* ── 6. Defaults (7) ────────────────────────────────────────────────── */
DATA work.defaults;
    LENGTH CICLO_ID $12;
    FORMAT FECHA_DEFAULT DATE9. FECHA_CURE DATE9.;
    DO i = 1 TO 7;
        CICLO_ID = 'CONTR_' !! PUT(i * 7, Z3.);
        FECHA_DEFAULT = '01JUN2023'D + (i * 60);
        IF MOD(i, 2) = 0 THEN DO;
            FECHA_CURE = FECHA_DEFAULT + 365;
            CURE_FLAG  = 1;
        END;
        ELSE DO;
            FECHA_CURE = .;
            CURE_FLAG  = 0;
        END;
        OUTPUT;
    END;
    DROP i;
RUN;

%put NOTE: [gen_all_data] 7 defaults generados.;

/* ── 7. Ratios financieros (10) ─────────────────────────────────────── */
DATA work.ratios_financieros;
    LENGTH CICLO_ID $12;
    FORMAT FECHA DATE9.;
    DO i = 1 TO 10;
        CICLO_ID    = 'CONTR_' !! PUT(i * 5, Z3.);
        EBITDA      = 100000 + INT(RANUNI(i) * 500000);
        DEUDA_NETA  = 200000 + INT(RANUNI(i + 1) * 800000);
        ACTIVO_TOTAL = 500000 + INT(RANUNI(i * 2) * 2000000);
        PATRIMONIO   = 100000 + INT(RANUNI(i + 3) * 500000);
        FECHA        = '01JAN2024'D;
        OUTPUT;
    END;
    DROP i;
RUN;

%put NOTE: [gen_all_data] 10 ratios financieros generados.;

/* ── Resumen ────────────────────────────────────────────────────────── */
%put NOTE: [gen_all_data] Todos los datos generados correctamente.;

PROC CONTENTS DATA=work.contratos; RUN;
PROC PRINT DATA=work.contratos (OBS=5); RUN;
PROC PRINT DATA=work.obligors (OBS=5); RUN;
PROC PRINT DATA=work.tarjetas; RUN;
PROC PRINT DATA=work.colaterales (OBS=5); RUN;
PROC PRINT DATA=work.historial_pagos (OBS=5); RUN;
PROC PRINT DATA=work.defaults; RUN;
PROC PRINT DATA=work.ratios_financieros; RUN;

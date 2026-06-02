/******************************************************************************
 * comprehensive_irb.sas
 * Covers: assignment, math functions, IF/THEN/ELSE, nested DO blocks,
 *         WHERE, KEEP/DROP, RETAIN, FIRST./LAST. (first record per group),
 *         COUNT via RETAIN, ARRAY + DO loop, SELECT/WHEN, MERGE,
 *         PROC SORT, PROC SQL, PROC MEANS, PROC FREQ, PROC EXPORT
 ******************************************************************************/

LIBNAME mylib '/data/irb';

/* ── STEP 1: Basic assignment, math functions, conditional floors ── */
DATA work.base;
    SET mylib.contratos;

    /* Simple arithmetic */
    EXPOSURE     = EAD * (1 + CCF);
    ECL_RAW      = PD_ESTIMADA * LGD_ESTIMADA * EAD;
    RWA          = EAD * 1.06 * PD_ESTIMADA * LGD_ESTIMADA * 12.5;

    /* Math functions */
    PD_BOUNDED   = MAX(0.0003, MIN(PD_ESTIMADA, 1.0));
    LGD_ROUNDED  = ROUND(LGD_ESTIMADA, 0.01);
    EAD_ABS      = ABS(EAD);
    LOG_EAD      = LOG(EAD + 1);

    /* String functions */
    SEGMENTO_UP  = UPCASE(SEGMENTO);
    COD_TRUNC    = SUBSTR(CONTRATO, 1, 6);
    NOMBRE_TRIM  = TRIM(NOMBRE_TITULAR);

    /* Date calculations */
    DIAS_DESDE_FI = TODAY() - FECHA_INCUMPLIMIENTO;
    ANIO_INCUMPL  = YEAR(FECHA_INCUMPLIMIENTO);

    /* Simple IF/THEN/ELSE */
    IF DPDS >= 90 THEN INCUMPLIMIENTO = 1;
    ELSE INCUMPLIMIENTO = 0;

    /* Multi-level ELSE IF */
    IF      DPDS = 0          THEN BUCKET_DPD = 'CURRENT';
    ELSE IF DPDS <= 30        THEN BUCKET_DPD = 'DPD_30';
    ELSE IF DPDS <= 60        THEN BUCKET_DPD = 'DPD_60';
    ELSE IF DPDS <= 90        THEN BUCKET_DPD = 'DPD_90';
    ELSE                           BUCKET_DPD = 'DEFAULT';

    /* Nested IF THEN DO — regulatory floors by segment */
    IF SEGMENTO = 'CORP' THEN DO;
        IF LGD_ESTIMADA < 0.45 THEN LGD_ESTIMADA = 0.45;
        IF PD_ESTIMADA  < 0.0003 THEN PD_ESTIMADA = 0.0003;
    END;
    ELSE IF SEGMENTO = 'RETAIL' THEN DO;
        IF COLATERAL_TIPO = 'HIPOTECA' THEN DO;
            IF LGD_ESTIMADA < 0.25 THEN LGD_ESTIMADA = 0.25;
        END;
        ELSE IF COLATERAL_TIPO = 'PERSONAL' THEN DO;
            IF LGD_ESTIMADA < 0.45 THEN LGD_ESTIMADA = 0.45;
        END;
        ELSE DO;
            IF LGD_ESTIMADA < 0.50 THEN LGD_ESTIMADA = 0.50;
        END;
    END;
    ELSE IF SEGMENTO = 'MORTGAGE' THEN DO;
        IF LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;
        IF LOAN_TO_VALUE > 0.80 THEN HAIRCUT = 0.15;
        ELSE HAIRCUT = 0.05;
    END;

    /* ECL after floors */
    ECL = PD_ESTIMADA * LGD_ESTIMADA * EAD;

    /* STAGE classification — IFRS9 */
    IF DPDS >= 30 AND STAGE_IFRS9 = 1 THEN STAGE_IFRS9 = 2;
    IF INCUMPLIMIENTO = 1 THEN STAGE_IFRS9 = 3;
RUN;

/* ── STEP 2: WHERE filter + KEEP ── */
DATA work.activos;
    SET work.base;
    WHERE ESTADO = 'ACTIVO' AND EAD > 0 AND VENTANA_CALIBRACION_YEARS >= 7;
    KEEP CONTRATO SEGMENTO COLATERAL_TIPO
         PD_ESTIMADA LGD_ESTIMADA EAD ECL RWA
         INCUMPLIMIENTO BUCKET_DPD STAGE_IFRS9
         VENTANA_CALIBRACION_YEARS VENTANA_OBSERVACION_YEARS;
RUN;

/* ── STEP 3: PROC SORT — order by segment then descending ECL ── */
PROC SORT DATA=work.activos OUT=work.activos_sorted;
    BY SEGMENTO DESCENDING ECL;
RUN;

/* ── STEP 4: RETAIN + FIRST./LAST. — count and aggregate per group ── */
DATA work.resumen_grupos;
    SET work.activos_sorted;
    BY SEGMENTO;

    RETAIN N_CONTRATOS 0
           ECL_TOTAL   0
           EAD_TOTAL   0
           N_DEFAULT   0;

    /* Reset accumulators at start of each group */
    IF FIRST.SEGMENTO THEN DO;
        N_CONTRATOS = 0;
        ECL_TOTAL   = 0;
        EAD_TOTAL   = 0;
        N_DEFAULT   = 0;
        PRIMER_ECL  = ECL;          /* first record in group */
    END;

    /* Accumulate */
    N_CONTRATOS + 1;
    ECL_TOTAL   + ECL;
    EAD_TOTAL   + EAD;
    IF INCUMPLIMIENTO = 1 THEN N_DEFAULT + 1;

    /* Output summary at end of group */
    IF LAST.SEGMENTO THEN DO;
        ECL_MEDIO      = ECL_TOTAL  / N_CONTRATOS;
        TASA_DEFAULT   = N_DEFAULT  / N_CONTRATOS;
        EAD_MEDIO      = EAD_TOTAL  / N_CONTRATOS;
        OUTPUT;
    END;
RUN;

/* ── STEP 5: ARRAY + DO loop — apply regulatory floors to all PD columns ── */
DATA work.pd_floors_applied;
    SET work.activos;
    ARRAY pd_cols{3} PD_1Y PD_2Y PD_3Y;
    DO i = 1 TO 3;
        IF pd_cols{i} < 0.0003 THEN pd_cols{i} = 0.0003;
        IF pd_cols{i} > 1.0    THEN pd_cols{i} = 1.0;
    END;
    DROP i;
RUN;

/* ── STEP 6: SELECT / WHEN — risk classification ── */
DATA work.clasificacion;
    SET work.activos;

    /* SELECT with WHEN conditions */
    SELECT;
        WHEN (PD_ESTIMADA < 0.001)  CLASE_RIESGO = 'AAA-A';
        WHEN (PD_ESTIMADA < 0.005)  CLASE_RIESGO = 'BBB';
        WHEN (PD_ESTIMADA < 0.02)   CLASE_RIESGO = 'BB';
        WHEN (PD_ESTIMADA < 0.10)   CLASE_RIESGO = 'B';
        OTHERWISE                   CLASE_RIESGO = 'CCC-D';
    END;

    /* SELECT on a discrete variable */
    SELECT (STAGE_IFRS9);
        WHEN (1) PROVISION_HORIZON = 12;
        WHEN (2) PROVISION_HORIZON = 999;   /* lifetime */
        WHEN (3) PROVISION_HORIZON = 999;
        OTHERWISE PROVISION_HORIZON = .;
    END;

    PROVISION = PD_ESTIMADA * LGD_ESTIMADA * EAD * PROVISION_HORIZON / 12;
RUN;

/* ── STEP 7: MERGE — join with collateral table ── */
PROC SORT DATA=work.clasificacion;   BY CONTRATO; RUN;
PROC SORT DATA=mylib.garantias;      BY CONTRATO; RUN;

DATA work.con_garantias;
    MERGE work.clasificacion  (IN=a)
          mylib.garantias     (IN=b KEEP=CONTRATO VALOR_GARANTIA TIPO_GARANTIA);
    BY CONTRATO;
    IF a;                               /* left join — keep all from clasificacion */

    TIENE_GARANTIA = b;
    IF TIENE_GARANTIA THEN DO;
        COBERTURA      = VALOR_GARANTIA / EAD;
        IF COBERTURA > 1.2 THEN LGD_AJUSTADA = LGD_ESTIMADA * 0.5;
        ELSE                     LGD_AJUSTADA = LGD_ESTIMADA * (1 - COBERTURA * 0.3);
    END;
    ELSE LGD_AJUSTADA = LGD_ESTIMADA;

    ECL_AJUSTADO = PD_ESTIMADA * LGD_AJUSTADA * EAD;
RUN;

/* ── STEP 8: Non-conforming rows ── */
DATA work.no_conformes;
    SET work.con_garantias;
    WHERE VENTANA_CALIBRACION_YEARS < 7
       OR VENTANA_OBSERVACION_YEARS < 5
       OR (COLATERAL_TIPO = 'HIPOTECA' AND LGD_ESTIMADA < 0.25)
       OR (SEGMENTO = 'CORP' AND LGD_ESTIMADA < 0.45)
       OR PD_ESTIMADA < 0.0003;
RUN;

/* ── PROC SQL — aggregate by segment ── */
PROC SQL;
    CREATE TABLE work.sql_resumen AS
    SELECT
        SEGMENTO,
        CLASE_RIESGO,
        COUNT(*)            AS N_CONTRATOS,
        SUM(EAD)            AS EAD_TOTAL     FORMAT=20.2,
        AVG(PD_ESTIMADA)    AS PD_MEDIA      FORMAT=8.6,
        AVG(LGD_AJUSTADA)   AS LGD_MEDIA     FORMAT=8.4,
        SUM(ECL_AJUSTADO)   AS ECL_TOTAL     FORMAT=20.2,
        SUM(RWA)            AS RWA_TOTAL     FORMAT=20.2,
        SUM(INCUMPLIMIENTO) AS N_DEFAULT
    FROM work.con_garantias
    GROUP BY SEGMENTO, CLASE_RIESGO
    ORDER BY EAD_TOTAL DESC;
QUIT;

/* ── PROC MEANS ── */
PROC MEANS DATA=work.con_garantias N MEAN STD MIN MAX SUM;
    CLASS SEGMENTO CLASE_RIESGO;
    VAR PD_ESTIMADA LGD_AJUSTADA EAD ECL_AJUSTADO RWA;
    TITLE 'Estadisticos descriptivos por segmento y clase de riesgo';
RUN;

/* ── PROC FREQ ── */
PROC FREQ DATA=work.con_garantias;
    TABLES SEGMENTO * STAGE_IFRS9 / NOCOL NOPERCENT;
    TABLES CLASE_RIESGO * INCUMPLIMIENTO / CHISQ;
RUN;

/* ── Export ── */
PROC EXPORT DATA=work.no_conformes
    OUTFILE='/data/output/no_conformes.csv'
    DBMS=CSV REPLACE;
RUN;

/*****************************************************************************
 * TOY LGD — Self-contained LGD estimation with planted bugs
 * ==========================================================
 *
 * Purpose: toy SAS codebase for the RegLLM agent to backtrace.
 * The agent reads this code, inspects intermediate tables via
 * user-provided PROC SQL queries, and finds the planted bugs.
 *
 * How to use:
 *   1. Run this file in SAS to create all tables (work.cycles → work.final)
 *   2. The agent will request PROC SQL/PROC PRINT queries to inspect data
 *   3. Run those queries and paste the output back to the agent
 *   4. The agent backtraces through the lineage to identify the bug
 *
 * Planted bugs (3 total):
 *   - Bug A: line 52 — wrong variable reference in floor condition
 *   - Bug B: line 72 — missing COALESCE for fusion contracts
 *   - Bug C: line 86 — off-by-one in DPD backstop comparison
 *****************************************************************************/

 /* ── 1. Inline input data ──────────────────────────────────────────── */
 /* 10 rows covering key scenarios. LGD_ESTIMADA values are chosen so
    that some are below regulatory floors (should be raised) and some
    are above (should pass through). */

DATA work.cycles;
    LENGTH CICLO_ID CONTRATO SEGMENTO COLATERAL_TIPO $20;
    LENGTH SW_FUSION PD_ESTIMADA LGD_ESTIMADA EAD 8;
    LENGTH DPDS STAGE_IFRS9 CURE_FLAG PROVISION_PERIOD_MONTHS 8;
    INPUT CICLO_ID $ CONTRATO $ SEGMENTO $ COLATERAL_TIPO $
          SW_FUSION PD_ESTIMADA LGD_ESTIMADA EAD
          DPDS STAGE_IFRS9 CURE_FLAG PROVISION_PERIOD_MONTHS;
    DATALINES;
CIC_001 CONT_001 CORP NINGUNA      0 0.010 0.40 100000  0 1 0 12
CIC_002 CONT_002 CORP NINGUNA      0 0.010 0.50 200000  0 1 0 12
CIC_003 CONT_003 CORP HIPOTECA     0 0.010 0.20 150000  0 1 0 12
CIC_004 CONT_004 CORP HIPOTECA     0 0.010 0.40 180000  0 1 0 12
CIC_005 CONT_005 CORP NINGUNA      1 0.020 .    300000 15 1 0 12
CIC_006 CONT_006 CORP NINGUNA      1 0.020 0.55 250000 20 1 0 12
CIC_007 CONT_007 RETAIL PERSONAL   0 0.030 0.35  80000 30 1 0 12
CIC_008 CONT_008 RETAIL PERSONAL   0 0.030 0.35  90000 29 1 0 12
CIC_009 CONT_009 CORP NINGUNA      0 0.010 0.45 120000  5 1 1 12
CIC_010 CONT_010 CORP NINGUNA      0 0.010 0.45 110000  5 1 0 12
;
RUN;


/* ── 2. Apply regulatory LGD floors ───────────────────────────────── */

DATA work.floored;
    SET work.cycles;

    /* CORP floor: 45% (CRR Art. 161(1)(b)) */
    IF SEGMENTO = 'CORP' AND COLATERAL_TIPO = 'NINGUNA' THEN DO;
        IF LGD_ESTIMADA < 0.45 THEN LGD_ESTIMADA = 0.45;
    END;

    /* HIPOTECA floor: 30% (CRR Art. 154(3))
       >>> BUG A: uses EAD instead of LGD_ESTIMADA in the condition <<<
       EAD is in the hundreds of thousands, so EAD < 0.30 is ALWAYS false.
       Result: the HIPOTECA floor never gets applied, and low LGD values
       (e.g. 0.20 on CIC_003) remain unchanged. Correct value: 0.30. */
    IF COLATERAL_TIPO = 'HIPOTECA' AND EAD < 0.30 THEN LGD_ESTIMADA = 0.30;

RUN;


/* ── 3. Margin of Conservatism (MoC) and ECL ─────────────────────── */

DATA work.ecl;
    SET work.floored;

    /* MoC = 5% of LGD (simplified approach)
       >>> BUG B: no COALESCE before using LGD_ESTIMADA <<<
       When LGD_ESTIMADA is missing (fusion contracts like CIC_005),
       MoC becomes . (missing), then LGD_CON_MOC becomes ., then ECL
       becomes missing. Should default to the regulatory floor instead.
       Correct: IF LGD_ESTIMADA = . THEN LGD_ESTIMADA = 0.45; */
    MoC = 0.05 * LGD_ESTIMADA;
    LGD_CON_MOC = LGD_ESTIMADA + MoC;

    /* PD floor: 0.03% (CRR Art. 160(1)) */
    IF PD_ESTIMADA < 0.0003 THEN PD_ESTIMADA = 0.0003;

    /* ECL = PD × LGD × EAD */
    ECL = PD_ESTIMADA * LGD_CON_MOC * EAD;

    /* Cure rate adjustment */
    IF CURE_FLAG = 1 THEN ECL_AJUSTADO = ECL * (1 - 0.15);
    ELSE                ECL_AJUSTADO = ECL;

RUN;


/* ── 4. IFRS 9 staging with backstop ──────────────────────────────── */

DATA work.final;
    SET work.ecl;

    /* 30-DPD backstop (IFRS 9 B5.5.12)
       >>> BUG C: > instead of >= <<<
       DPDS = 30 should trigger reclassification to Stage 2,
       but > 30 means DPDS = 30 is NOT caught.
       Correct: IF DPDS >= 30 AND STAGE_IFRS9 = 1 THEN ... */
    IF DPDS > 30 AND STAGE_IFRS9 = 1 THEN DO;
        STAGE_IFRS9 = 2;
        STAGE_RECLASIFICADO = 1;
    END;
    ELSE STAGE_RECLASIFICADO = 0;

RUN;


/* ── 5. Expected correct values (for reference) ───────────────────── */
/*
   The table below shows what work.final SHOULD contain after a
   correct run. Compare with actual output to identify anomalous rows.

   CICLO_ID | LGD_EST | MoC   | LGD_CON_MOC | ECL    | STAGE | RECLAS
   ---------+---------+-------+-------------+--------+-------+-------
   CIC_001  | 0.45    | 0.0225| 0.4725      |  472.5 | 1     | 0
   CIC_002  | 0.50    | 0.0250| 0.5250      | 1050   | 1     | 0
   CIC_003  | 0.30    | 0.0150| 0.3150      |  472.5 | 1     | 0
   CIC_004  | 0.40    | 0.0200| 0.4200      |  756   | 1     | 0
   CIC_005  | 0.45    | 0.0225| 0.4725      | 2835   | 1     | 0
   CIC_006  | 0.55    | 0.0275| 0.5775      | 2887.5 | 1     | 0
   CIC_007  | 0.35    | 0.0175| 0.3675      |  882   | 2     | 1
   CIC_008  | 0.35    | 0.0175| 0.3675      |  992.25| 1     | 0
   CIC_009  | 0.45    | 0.0225| 0.4725      |  567   | 1     | 0
   CIC_010  | 0.45    | 0.0225| 0.4725      |  519.75| 1     | 0
*/


/* ── 6. Query helper — agent will ask for these ───────────────────── */
/* Uncomment and modify as requested by the agent:

   PROC PRINT DATA=work.cycles; RUN;
   PROC PRINT DATA=work.floored; RUN;
   PROC PRINT DATA=work.ecl; RUN;
   PROC PRINT DATA=work.final; RUN;

   PROC SQL;
       SELECT CICLO_ID, LGD_ESTIMADA, EAD
       FROM work.floored
       WHERE COLATERAL_TIPO = 'HIPOTECA';
   QUIT;

   PROC SQL;
       SELECT CICLO_ID, SW_FUSION, LGD_ESTIMADA, MoC, LGD_CON_MOC, ECL
       FROM work.ecl
       WHERE SW_FUSION = 1;
   QUIT;

   PROC SQL;
       SELECT CICLO_ID, DPDS, STAGE_IFRS9, STAGE_RECLASIFICADO
       FROM work.final
       WHERE DPDS >= 30;
   QUIT;
*/

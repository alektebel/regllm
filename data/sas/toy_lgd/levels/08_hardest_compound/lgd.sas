/*****************************************************************************
 * Level 08 — Hardest: Compound bug (fusion + wrong aggregation)
 * Two independent bugs that interact to produce severely wrong output:
 *
 * Bug 1: LEFT JOIN on ID_FUSION_FINAL creates duplicate rows (non-unique key)
 * Bug 2: SUM(LGD_ESTIMADA) in aggregation instead of MAX/LAST — SUM of
 *        the duplicated LGD values inflates LGD_ESTIMADA beyond [0,1]
 *
 * Together: fusion joins produce 2 rows per cycle, SUM doubles LGD.
 *****************************************************************************/

DATA work.ciclos;
    LENGTH ID_CONTR_CICLO_LGD ID_CONTRATO SEGMENTO COLATERAL_TIPO $20;
    LENGTH PD_ESTIMADA LGD_ESTIMADA EAD 8;
    LENGTH DPDS STAGE_IFRS9 CURE_FLAG PROVISION_PERIOD_MONTHS 8;
    INPUT ID_CONTR_CICLO_LGD $ ID_CONTRATO $ SEGMENTO $ COLATERAL_TIPO $
          PD_ESTIMADA LGD_ESTIMADA EAD
          DPDS STAGE_IFRS9 CURE_FLAG PROVISION_PERIOD_MONTHS;
    DATALINES;
CIC_001 CONT_001 CORP NINGUNA      0.010 0.40 100000  0 1 0 12
CIC_002 CONT_002 CORP NINGUNA      0.010 0.50 200000  0 1 0 12
CIC_005 CONT_005 CORP NINGUNA      0.020 .    300000 15 1 0 12
CIC_006 CONT_006 CORP NINGUNA      0.020 0.55 250000 20 1 0 12
;
RUN;

DATA work.contratos;
    LENGTH ID_CONTRATO ID_FUSION_FINAL ENTIDAD_ORIGEN $20;
    LENGTH SW_FUSION 8;
    INPUT ID_CONTRATO $ SW_FUSION ID_FUSION_FINAL $ ENTIDAD_ORIGEN $;
    DATALINES;
CONT_001 0 .         BANCO_A
CONT_002 0 .         BANCO_A
CONT_005 1 FUS_7712  BANCO_B
CONT_006 1 FUS_7712  BANCO_C
;
RUN;

DATA work.basilea_mensual;
    LENGTH ID_CONTRATO ID_FUSION_FINAL $20;
    LENGTH ID_FCH_DATOS OR_EAD 8;
    INPUT ID_CONTRATO $ ID_FUSION_FINAL $ ID_FCH_DATOS OR_EAD;
    DATALINES;
CONT_005 FUS_7712 201402 100000
CONT_A   FUS_7712 201402 100000
CONT_B   FUS_7712 201402 100000
;
RUN;

/* Step 1: Enrich with fusion flags */
DATA work.ciclos_enriched;
    MERGE work.ciclos (IN=a) work.contratos (IN=b);
    BY ID_CONTRATO;
    IF a;
RUN;

/* Step 2: Enrich with OR_EAD from BASILEA_MENSUAL
   BUG 1: JOIN on non-unique ID_FUSION_FINAL duplicates rows */
PROC SQL;
    CREATE TABLE work.ciclos_con_ead AS
    SELECT
        t1.*,
        t2.OR_EAD AS OR_EAD_BASILEA
    FROM work.ciclos_enriched AS t1
    LEFT JOIN work.basilea_mensual AS t2
        ON  t1.ID_FUSION_FINAL = t2.ID_FUSION_FINAL
        AND t1.PERIODO_OBSERVACION = t2.ID_FCH_DATOS
    WHERE t1.SW_FUSION = 1;
QUIT;

/* Aggregate — BUG 2: SUM instead of MAX for LGD_ESTIMADA
   Non-fusion rows (single row) → SUM = correct value
   Fusion rows (duplicated) → SUM doubles LGD_ESTIMADA
   For CIC_006: two rows with LGD=0.55 → SUM = 1.10 (absurd!)
   For CIC_005: two rows with LGD=. → SUM = . (still missing) */
PROC SQL;
    CREATE TABLE work.ciclos_agg AS
    SELECT
        ID_CONTR_CICLO_LGD,
        ID_CONTRATO,
        SEGMENTO,
        SW_FUSION,
        ID_FUSION_FINAL,
        MAX(PD_ESTIMADA) AS PD_ESTIMADA,
        SUM(LGD_ESTIMADA) AS LGD_ESTIMADA,  /* BUG 2: should be MAX */
        MAX(EAD) AS EAD,
        MAX(DPDS) AS DPDS,
        MAX(STAGE_IFRS9) AS STAGE_IFRS9,
        MAX(CURE_FLAG) AS CURE_FLAG,
        SUM(OR_EAD_BASILEA) AS OR_EAD
    FROM work.ciclos_con_ead
    GROUP BY ID_CONTR_CICLO_LGD, ID_CONTRATO, SEGMENTO, SW_FUSION, ID_FUSION_FINAL;
QUIT;

/* Step 3: Apply floors + MoC + ECL */
DATA work.ecl;
    SET work.ciclos_agg;
    IF LGD_ESTIMADA = . THEN LGD_ESTIMADA = 0.45;
    MoC = 0.05 * LGD_ESTIMADA;
    LGD_CON_MOC = LGD_ESTIMADA + MoC;
    IF PD_ESTIMADA < 0.0003 THEN PD_ESTIMADA = 0.0003;
    ECL = PD_ESTIMADA * LGD_CON_MOC * EAD;
RUN;

/* Expected correct:
   CIC_005: LGD=0.45, OR_EAD=100000, MoC=0.0225, ECL=2835
   CIC_006: LGD=0.55, OR_EAD=100000, MoC=0.0275, ECL=2887.5

   Actual (both bugs):
   CIC_005: LGD=0.45 (missing+COALESCE), OR_EAD=200000 (inflated)
   CIC_006: LGD=1.10 (doubled!),         OR_EAD=200000 (inflated)

   If only Bug 1 is fixed but not Bug 2:
   CIC_005: LGD=0.45, OR_EAD=100000 (correct if MAX vs SUM for OR_EAD too)
   CIC_006: LGD=0.55, OR_EAD=100000 (correct if MAX vs SUM for OR_EAD too)

   The compound effect: Bug 1 creates the duplicate rows that Bug 2
   then double-counts. Neither alone causes the wrong LGD_ESTIMADA.
*/

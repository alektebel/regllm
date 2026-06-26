/*****************************************************************************
 * Level 05 — Medium: Fusion join duplication
 * Bug: LEFT JOIN on ID_FUSION_FINAL produces duplicate rows when multiple
 * contracts share the same fusion ID. SUM(OR_EAD) then inflates the value.
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
CONT_005 1 FUS_7712  BANCO_B
CONT_006 1 FUS_7712  BANCO_C
;
RUN;

/* BASILEA_MENSUAL has 2 entries for FUS_7712 (CONT_A and CONT_B).
   Both have OR_EAD=100000. A JOIN on ID_FUSION_FINAL = 'FUS_7712'
   matches BOTH rows for each cycle. */
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
   BUG: JOIN on ID_FUSION_FINAL is not unique.
   Two rows match → SUM(OR_EAD_BASILEA) doubles the value. */
PROC SQL;
    CREATE TABLE work.ciclos_con_ead AS
    SELECT
        t1.*,
        t2.OR_EAD AS OR_EAD_BASILEA
    FROM work.ciclos_enriched AS t1
    LEFT JOIN work.basilea_mensual AS t2
        ON  t1.ID_FUSION_FINAL = t2.ID_FUSION_FINAL
        AND t1.PERIODO_OBSERVACION = t2.ID_FCH_DATOS
    WHERE t1.SW_FUSION = 1
    ORDER BY t1.ID_CONTR_CICLO_LGD;
QUIT;

/* Aggregate: BUG — SUM over duplicated rows inflates OR_EAD */
PROC SQL;
    CREATE TABLE work.ciclos_agg AS
    SELECT
        ID_CONTR_CICLO_LGD,
        ID_CONTRATO,
        SEGMENTO,
        SW_FUSION,
        ID_FUSION_FINAL,
        MAX(PD_ESTIMADA) AS PD_ESTIMADA,
        MAX(LGD_ESTIMADA) AS LGD_ESTIMADA,
        MAX(EAD) AS EAD,
        MAX(DPDS) AS DPDS,
        MAX(STAGE_IFRS9) AS STAGE_IFRS9,
        MAX(CURE_FLAG) AS CURE_FLAG,
        SUM(OR_EAD_BASILEA) AS OR_EAD
    FROM work.ciclos_con_ead
    GROUP BY ID_CONTR_CICLO_LGD, ID_CONTRATO, SEGMENTO, SW_FUSION, ID_FUSION_FINAL;
QUIT;

/* Expected:
   CIC_005: OR_EAD=100000
   CIC_006: OR_EAD=100000
   Actual:
   CIC_005: OR_EAD=200000 (2 rows × 100000 each = 200000)
   CIC_006: OR_EAD=200000
*/

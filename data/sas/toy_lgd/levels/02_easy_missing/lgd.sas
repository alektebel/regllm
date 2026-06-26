/*****************************************************************************
 * Level 02 — Easy: Missing COALESCE for fusion contracts
 * Bug: No default for missing LGD_ESTIMADA before computing MoC.
 * When LGD_ESTIMADA is missing (fusion absorbed entities), MoC becomes
 * missing, LGD_CON_MOC becomes missing, and ECL becomes missing.
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

/* Step 1: Enrich with fusion flags */
DATA work.ciclos_enriched;
    MERGE work.ciclos (IN=a) work.contratos (IN=b);
    BY ID_CONTRATO;
    IF a;
RUN;

/* Step 2: Apply floors */
DATA work.floored;
    SET work.ciclos_enriched;
    IF SEGMENTO = 'CORP' AND COLATERAL_TIPO = 'NINGUNA' THEN DO;
        IF LGD_ESTIMADA < 0.45 THEN LGD_ESTIMADA = 0.45;
    END;
RUN;

/* Step 3: MoC and ECL — BUG: no COALESCE for missing LGD */
DATA work.ecl;
    SET work.floored;
    MoC = 0.05 * LGD_ESTIMADA;
    LGD_CON_MOC = LGD_ESTIMADA + MoC;
    IF PD_ESTIMADA < 0.0003 THEN PD_ESTIMADA = 0.0003;
    ECL = PD_ESTIMADA * LGD_CON_MOC * EAD;
    IF CURE_FLAG = 1 THEN ECL_AJUSTADO = ECL * 0.85;
    ELSE                ECL_AJUSTADO = ECL;
RUN;

/* Step 4: Staging */
DATA work.final;
    SET work.ecl;
    IF DPDS >= 30 AND STAGE_IFRS9 = 1 THEN DO;
        STAGE_IFRS9 = 2;
        STAGE_RECLASIFICADO = 1;
    END;
    ELSE STAGE_RECLASIFICADO = 0;
RUN;

/* Expected correct:
   CIC_005: LGD=0.45(default), MoC=0.0225, LGD_CON_MOC=0.4725, ECL=2835
   Actual (buggy):
   CIC_005: LGD=., MoC=., LGD_CON_MOC=., ECL=.
*/

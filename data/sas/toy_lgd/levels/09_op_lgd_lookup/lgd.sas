/*****************************************************************************
 * Level 09 — Hard: OP_LGD security cross-reference lookup
 *
 * OP_LGD is the contract ID of the adjudicación (foreclosure/assignment).
 * When a contract has OP_LGD assigned, the referenced contract should be
 * secured (have HIPOTECA collateral) OR the OP_LGD should have a secured
 * cycle somewhere in the database.
 *
 * The correct logic is:
 *   IF OP_LGD is assigned AND self is NOT secured (HIPOTECA) THEN
 *     Look up whether the OP_LGD reference has a secured cycle
 *     If not found → flag the contract
 *
 * Bug: _LOOKUP uses ID_CONTR_CICLO_LGD (the current row's cycle ID)
 * instead of OP_LGD (the referenced contract ID). Since the reference
 * table keys on ID_CONTRATO, looking up "CIC_004" never matches any row,
 * and every unsecured contract with an OP_LGD reference gets flagged —
 * even if the OP_LGD contract IS actually secured.
 *****************************************************************************/

DATA work.ciclos;
    LENGTH ID_CONTR_CICLO_LGD ID_CONTRATO SEGMENTO COLATERAL_TIPO OP_LGD $20;
    LENGTH PD_ESTIMADA LGD_ESTIMADA EAD 8;
    LENGTH DPDS STAGE_IFRS9 CURE_FLAG PROVISION_PERIOD_MONTHS 8;
    INPUT ID_CONTR_CICLO_LGD $ ID_CONTRATO $ SEGMENTO $ COLATERAL_TIPO $
          PD_ESTIMADA LGD_ESTIMADA EAD
          DPDS STAGE_IFRS9 CURE_FLAG PROVISION_PERIOD_MONTHS
          OP_LGD $;
    DATALINES;
CIC_001 CONT_001 CORP HIPOTECA     0.010 0.40 100000  0 1 0 12 .
CIC_002 CONT_002 CORP NINGUNA      0.010 0.30 200000  0 1 0 12 .
CIC_003 CONT_003 CORP NINGUNA      0.020 0.45 150000  5 1 0 12 CONT_005
CIC_004 CONT_004 CORP NINGUNA      0.015 0.50 180000  0 1 0 12 CONT_002
CIC_005 CONT_005 CORP HIPOTECA     0.020 0.35 250000 20 1 0 12 .
;
RUN;

/* Reference table: which contracts have secured cycles */
DATA work.secured_cycles;
    LENGTH ID_CONTRATO $20 SECURED 8;
    INPUT ID_CONTRATO $ SECURED;
    DATALINES;
CONT_002 1
CONT_005 0
;
RUN;

/* Step: Check OP_LGD security
   BUG: _LOOKUP uses ID_CONTR_CICLO_LGD as the lookup key value
   instead of OP_LGD. It should look up the OP_LGD reference, but
   instead looks up the current row's own cycle ID.
   Result: CIC_004 gets flagged even though its OP_LGD (CONT_002)
   IS in the secured_cycles table (SECURED=1). */
DATA work.checked;
    SET work.ciclos;
    LENGTH OP_LGD_FLAG OP_LGD_IS_SECURED 8;

    IF OP_LGD NE '' AND OP_LGD NE '.' THEN DO;
        /* BUG: should be _LOOKUP('SECURED_CYCLES', 'ID_CONTRATO', OP_LGD, 'SECURED')
           Using ID_CONTR_CICLO_LGD (cycle ID) instead of OP_LGD (ref contract ID)
           means the lookup never finds a match. COALESCE defaults to 0. */
        OP_LGD_IS_SECURED = COALESCE(
            _LOOKUP('SECURED_CYCLES', 'ID_CONTRATO', ID_CONTR_CICLO_LGD, 'SECURED'),
            0);

        IF OP_LGD_IS_SECURED NE 1 AND COLATERAL_TIPO NE 'HIPOTECA' THEN
            OP_LGD_FLAG = 1;
        ELSE
            OP_LGD_FLAG = 0;
    END;
    ELSE OP_LGD_FLAG = 0;
RUN;

/* Expected correct lookup (using OP_LGD):
   CIC_001: OP_LGD=.        → FLAG=0 (no OP_LGD)
   CIC_002: OP_LGD=.        → FLAG=0 (no OP_LGD)
   CIC_003: OP_LGD=CONT_005 → lookup CONT_005 → SECURED=0
            self NINGUNA    → FLAG=1 (unsecured OP_LGD)
   CIC_004: OP_LGD=CONT_002 → lookup CONT_002 → SECURED=1
            self NINGUNA    → FLAG=0 (OP_LGD is secured)
   CIC_005: OP_LGD=.        → FLAG=0 (no OP_LGD)

   Actual (buggy lookup on ID_CONTR_CICLO_LGD + COALESCE to 0):
   CIC_001: OP_LGD=.            → FLAG=0 ✓
   CIC_002: OP_LGD=.            → FLAG=0 ✓
   CIC_003: lookup CIC_003      → not found → COALESCE(.,0)=0
            self NINGUNA        → FLAG=1 ✓ (correct, but by coincidence)
   CIC_004: lookup CIC_004      → not found → COALESCE(.,0)=0
            self NINGUNA        → FLAG=1 ✗ (WRONG! CONT_002 IS secured)
   CIC_005: OP_LGD=.            → FLAG=0 ✓
*/

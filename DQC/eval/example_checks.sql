-- Example DQC checks to exercise the harness coverage mode.
-- A realistic mix: some good coherence checks, one lazy single-column check
-- (fires on a decoy), and one overly-broad check (false positives on clean).

-- GOOD: adjudicacion valor>0 but flag inactive (catches D07)
SELECT ID_CONTR_CICLO_LGD FROM ciclos_calibrados WHERE ADJUDICACION_VALOR > 0 AND ADJUDICACION_FLAG = '0';

-- GOOD: closed cycle without terminacion (catches D11)
SELECT ID_CONTR_CICLO_LGD FROM ciclos_calibrados WHERE ESTADO_CICLO = 'CERRADO' AND (TERMINACION = '' OR TERMINACION IS NULL);

-- GOOD: stage 3 default with <30 DPD (catches D06)
SELECT ID_CONTR_CICLO_LGD FROM ciclos_calibrados WHERE STAGE_IFRS9 = 3 AND DPDS < 30;

-- GOOD: recovery exceeds 1.5x EAD (catches D10)
SELECT ID_CONTR_CICLO_LGD FROM ciclos_calibrados WHERE (RECUPERACION_ACUMULADA + COSTE_TOTAL_ACUMULADO) > 1.5 * EAD_TOTAL;

-- LAZY: single-column range check -> fires on decoy DA, scores r_coherence=0
SELECT ID_CONTR_CICLO_LGD FROM ciclos_calibrados WHERE EAD_TOTAL <= 0;

-- OVER-BROAD: tautology that fires on clean data (precision penalty)
SELECT ID_CONTR_CICLO_LGD FROM ciclos_calibrados WHERE PD_ESTIMADA >= 0 OR ADJUDICACION_FLAG IS NOT NULL;

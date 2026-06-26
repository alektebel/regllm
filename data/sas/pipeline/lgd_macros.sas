/*****************************************************************************
 * LGD Estimation Pipeline — Macro Library
 * =========================================
 * Regulatory framework: CRR Art. 154(3), 160(1), 161(1)(b), 181(1)(b)
 *                       EBA GL 2017/16, EBA GL 2019/03
 *                       Circular 6/2016 (Banco de España) Art. 8, 12, 15, 23
 *                       IFRS 9 B5.5.12
 *
 * Each macro encapsulates one pipeline step. Macros communicate via
 * work.* tables. Run order:
 *   1. lgd_init       — libnames, global parameters
 *   2. lgd_load       — raw cycles + contract enrichment
 *   3. lgd_gen_cashflows — synthetic recovery cashflows (demo) or load real
 *   4. lgd_discount   — NPV of recovery cashflows
 *   5. lgd_floors     — regulatory LGD floor application
 *   6. lgd_calculate  — LGD estimate with MoC
 *   7. lgd_ecl        — ECL = PD × LGD × EAD
 *   8. lgd_rwa        — RWA = EAD × LGD × 12.5 × K_IRB
 *   9. lgd_staging    — IFRS 9 staging with 30-DPD backstop
 *  10. lgd_validate   — flag non-conforming records
 *  11. lgd_report     — summary stats + export
 *****************************************************************************/


/* =========================================================================
 * 1. lgd_init — Global configuration
 * =========================================================================
 * Sets library paths, regulatory parameters, and output options.
 * Call this first. Parameters are captured as global macro variables.
 * ---------------------------------------------------------------------- */
%macro lgd_init(
    lib_in        = mylib,        /* Input library (raw data) */
    lib_out       = work,         /* Output library              */
    data_cycles   = ciclos_recuperacion,  /* Recovery cycles table  */
    data_contratos= catalogo_contratos,   /* Contract catalogue     */
    data_flujos   = flujos_recuperacion,  /* Monthly cashflows      */

    /* Regulatory LGD floors (segment × collateral) — Art. 15 */
    floor_hipoteca    = 0.30,
    floor_corp_none   = 0.45,
    floor_corp_fin    = 0.35,
    floor_retail_pers = 0.35,
    floor_retail_none = 0.45,
    floor_mortg_none  = 0.40,
    floor_mortg_hip   = 0.25,

    /* Stage 3 multiplier on LGD floor */
    stage3_lgd_mult = 1.20,

    /* PD floor — Art. 15 */
    pd_floor = 0.0003,

    /* MoC parameters — EBA GL 2017/16 §133 */
    moc_rate       = 0.10,     /* 10% of deviation from segment mean */
    moc_floor_rate = 0.05,     /* 5% flat floor when no deviation     */

    /* Cure rate adjustment */
    cure_rate = 0.15,

    /* Validation windows — EBA GL 2017/16 §6.3 */
    min_calib_years = 7,
    min_obs_years   = 5,

    /* IFRS 9 backstop DPD threshold */
    backstop_dpd = 30,

    /* Output paths */
    out_csv_no_conformes = /data/irb/output/no_conformes_lgd.csv,
    out_csv_summary      = /data/irb/output/resumen_lgd.csv
);

    /* Persist all parameters as global macro variables */
    %global _LGDLIB_IN _LGDLIB_OUT
            _LGD_DATA_CYCLES _LGD_DATA_CONTRATOS _LGD_DATA_FLUJOS
            _LGD_FLOOR_HIPOTECA _LGD_FLOOR_CORP_NONE _LGD_FLOOR_CORP_FIN
            _LGD_FLOOR_RETAIL_PERS _LGD_FLOOR_RETAIL_NONE
            _LGD_FLOOR_MORTG_NONE _LGD_FLOOR_MORTG_HIP
            _LGD_STAGE3_MULT _LGD_PD_FLOOR _LGD_MOC_RATE _LGD_MOC_FLOOR
            _LGD_CURE_RATE _LGD_MIN_CALIB _LGD_MIN_OBS
            _LGD_BACKSTOP_DPD
            _LGD_OUT_NC _LGD_OUT_SUM;

    /* Assign library */
    LIBNAME &lib_in. "/data/irb/calibracion" ACCESS=READONLY;
    %let _LGDLIB_IN    = &lib_in.;
    %let _LGDLIB_OUT   = &lib_out.;
    %let _LGD_DATA_CYCLES     = &data_cycles.;
    %let _LGD_DATA_CONTRATOS  = &data_contratos.;
    %let _LGD_DATA_FLUJOS     = &data_flujos.;

    %let _LGD_FLOOR_HIPOTECA    = &floor_hipoteca.;
    %let _LGD_FLOOR_CORP_NONE   = &floor_corp_none.;
    %let _LGD_FLOOR_CORP_FIN    = &floor_corp_fin.;
    %let _LGD_FLOOR_RETAIL_PERS = &floor_retail_pers.;
    %let _LGD_FLOOR_RETAIL_NONE = &floor_retail_none.;
    %let _LGD_FLOOR_MORTG_NONE  = &floor_mortg_none.;
    %let _LGD_FLOOR_MORTG_HIP   = &floor_mortg_hip.;
    %let _LGD_STAGE3_MULT       = &stage3_lgd_mult.;
    %let _LGD_PD_FLOOR          = &pd_floor.;
    %let _LGD_MOC_RATE          = &moc_rate.;
    %let _LGD_MOC_FLOOR         = &moc_floor_rate.;
    %let _LGD_CURE_RATE          = &cure_rate.;
    %let _LGD_MIN_CALIB         = &min_calib_years.;
    %let _LGD_MIN_OBS           = &min_obs_years.;
    %let _LGD_BACKSTOP_DPD      = &backstop_dpd.;
    %let _LGD_OUT_NC            = &out_csv_no_conformes.;
    %let _LGD_OUT_SUM           = &out_csv_summary.;

    %put NOTE: [lgd_init] LGD pipeline configured.;
    %put NOTE:   PD floor          = &_LGD_PD_FLOOR.;
    %put NOTE:   CORP floor        = &_LGD_FLOOR_CORP_NONE.;
    %put NOTE:   HIPOTECA floor    = &_LGD_FLOOR_HIPOTECA.;
    %put NOTE:   MoC rate          = &_LGD_MOC_RATE.;
    %put NOTE:   Cure rate         = &_LGD_CURE_RATE.;
    %put NOTE:   Min calib window  = &_LGD_MIN_CALIB. years;
%mend lgd_init;


/* =========================================================================
 * 2. lgd_load — Load and enrich raw recovery cycles
 * =========================================================================
 * Input:  &_LGDLIB_IN..&_LGD_DATA_CYCLES   (recovery cycles)
 *         &_LGDLIB_IN..&_LGD_DATA_CONTRATOS (contract catalogue)
 * Output: work.cycles_raw  (enriched with SW_FUSION, ID_FUSION_FINAL)
 * ---------------------------------------------------------------------- */
%macro lgd_load;

    %put NOTE: [lgd_load] Loading cycles from &_LGDLIB_IN..&_LGD_DATA_CYCLES.;

    /* Base: complete cycles with sufficient provision history */
    DATA work.cycles_raw;
        SET &_LGDLIB_IN..&_LGD_DATA_CYCLES.;
        WHERE PROVISION_PERIOD_MONTHS >= 9;
    RUN;

    /* Enrich with fusion flags from contract catalogue */
    PROC SQL;
        CREATE TABLE work.cycles_raw AS
        SELECT
            t1.*,
            COALESCE(t2.SW_FUSION, 0) AS SW_FUSION,
            t2.ID_FUSION_FINAL,
            t2.ENTIDAD_ORIGEN
        FROM work.cycles_raw AS t1
        LEFT JOIN &_LGDLIB_IN..&_LGD_DATA_CONTRATOS. AS t2
            ON t1.CONTRATO = t2.ID_CONTRATO;
    QUIT;

    %put NOTE: [lgd_load] Loaded work.cycles_raw.;

%mend lgd_load;


/* =========================================================================
 * 3. lgd_gen_cashflows — Generate synthetic recovery cashflows
 * =========================================================================
 * When real monthly cashflows are unavailable, this macro generates
 * synthetic cashflows consistent with LGD_ESTIMADA for demo purposes.
 *
 * In production, replace with direct load from &_LGD_DATA_FLUJOS.
 *
 * Output: work.recovery_cashflows (CICLO_ID, MES, FLUJO_NETO)
 * ---------------------------------------------------------------------- */
%macro lgd_gen_cashflows;

    %put NOTE: [lgd_gen_cashflows] Generating synthetic recovery cashflows.;

    /* Recovery period: assume 12-48 months depending on collateral type */
    DATA work.cashflow_params;
        SET work.cycles_raw;
        WHERE LGD_ESTIMADA > 0 AND EAD > 0;

        /* Expected recovery rate (1 - LGD) */
        RECOVERY_RATE = 1 - LGD_ESTIMADA;

        /* Recovery period in months: secured collateral is faster */
        SELECT (COLATERAL_TIPO);
            WHEN ('HIPOTECA') RECOV_MONTHS = 36;
            WHEN ('FINANCIERO', 'COLATERAL_FIN') RECOV_MONTHS = 12;
            WHEN ('PERSONAL')  RECOV_MONTHS = 18;
            OTHERWISE          RECOV_MONTHS = 24;
        END;

        /* Monthly discount rate from TIPO_LGD or default 5% p.a. */
        TASA_MENSUAL = (1 + 0.05) ** (1/12) - 1;

        KEEP CICLO_ID EAD RECOVERY_RATE RECOV_MONTHS TASA_MENSUAL
             COLATERAL_TIPO SEGMENTO;
    RUN;

    /* Expand to monthly cashflows */
    DATA work.recovery_cashflows (KEEP=CICLO_ID MES FLUJO_NETO TASA_MENSUAL);
        SET work.cashflow_params;
        DO MES = 1 TO RECOV_MONTHS;
            /* Recovery distribution: front-loaded for secured, uniform otherwise */
            IF COLATERAL_TIPO IN ('HIPOTECA') THEN DO;
                IF MES <= 6 THEN FACTOR = 1.5 / RECOV_MONTHS;      * front 6 months;
                ELSE               FACTOR = 0.8 / RECOV_MONTHS;
            END;
            ELSE IF COLATERAL_TIPO IN ('FINANCIERO', 'COLATERAL_FIN', 'PERSONAL') THEN DO;
                IF MES <= 3 THEN FACTOR = 2.0 / RECOV_MONTHS;      * front 3 months;
                ELSE               FACTOR = 0.7 / RECOV_MONTHS;
            END;
            ELSE DO;
                FACTOR = 1 / RECOV_MONTHS;                          * uniform;
            END;

            FLUJO_NETO = EAD * RECOVERY_RATE * FACTOR;
            OUTPUT;
        END;
    RUN;

    PROC SQL;
        CREATE TABLE work.cashflow_summary AS
        SELECT
            CICLO_ID,
            COUNT(*) AS N_FLUJOS,
            SUM(FLUJO_NETO) AS TOTAL_RECOVERY
        FROM work.recovery_cashflows
        GROUP BY CICLO_ID;
    QUIT;

    %put NOTE: [lgd_gen_cashflows] Generated cashflows. Output: work.cashflow_summary;

%mend lgd_gen_cashflows;


/* =========================================================================
 * 4. lgd_discount — Discount recovery cashflows to NPV
 * =========================================================================
 * Input:  work.recovery_cashflows
 * Output: work.cycle_npv  (CICLO_ID, NPV_RECOVERY, TASA_DESCUENTO_USADA)
 * ---------------------------------------------------------------------- */
%macro lgd_discount;

    %put NOTE: [lgd_discount] Computing discounted NPV of recoveries.;

    /* Discount each cashflow to present value */
    DATA work.cashflows_discounted;
        SET work.recovery_cashflows;
        PV = FLUJO_NETO / (1 + TASA_MENSUAL) ** MES;
    RUN;

    /* Aggregate to cycle level */
    PROC SQL;
        CREATE TABLE work.cycle_npv AS
        SELECT
            CICLO_ID,
            SUM(PV) AS NPV_RECOVERY,
            SUM(FLUJO_NETO) AS GROSS_RECOVERY,
            MEAN(TASA_MENSUAL) AS TASA_DESCUENTO_USADA
        FROM work.cashflows_discounted
        GROUP BY CICLO_ID;
    QUIT;

    %put NOTE: [lgd_discount] NPV computed. Output: work.cycle_npv;

%mend lgd_discount;


/* =========================================================================
 * 5. lgd_floors — Apply regulatory LGD floors
 * =========================================================================
 * Applies floors per (SEGMENTO, COLATERAL_TIPO) from Art. 15.
 * Tracks which floor was applied and flags affected rows.
 *
 * Input:  work.cycles_raw
 * Output: work.cycles_floored
 * ---------------------------------------------------------------------- */
%macro lgd_floors;

    %put NOTE: [lgd_floors] Applying regulatory LGD floors.;

    DATA work.cycles_floored;
        SET work.cycles_raw;

        LENGTH _FLOOR_APLICADO 8;
        _FLOOR_APLICADO = 0;

        /* Determine applicable floor based on SEGMENTO × COLATERAL_TIPO */
        SELECT;
            /* — HIPOTECA collateral across segments — */
            WHEN (COLATERAL_TIPO = 'HIPOTECA' AND SEGMENTO = 'MORTGAGE')
                _FLOOR_APLICADO = &_LGD_FLOOR_MORTG_HIP.;
            WHEN (COLATERAL_TIPO = 'HIPOTECA' AND SEGMENTO NOT IN ('MORTGAGE'))
                _FLOOR_APLICADO = &_LGD_FLOOR_HIPOTECA.;

            /* — CORP segment — */
            WHEN (SEGMENTO = 'CORP' AND COLATERAL_TIPO IN ('NINGUNA', ''))
                _FLOOR_APLICADO = &_LGD_FLOOR_CORP_NONE.;
            WHEN (SEGMENTO = 'CORP' AND COLATERAL_TIPO IN ('FINANCIERO', 'COLATERAL_FIN'))
                _FLOOR_APLICADO = &_LGD_FLOOR_CORP_FIN.;

            /* — RETAIL segment — */
            WHEN (SEGMENTO = 'RETAIL' AND COLATERAL_TIPO = 'PERSONAL')
                _FLOOR_APLICADO = &_LGD_FLOOR_RETAIL_PERS.;
            WHEN (SEGMENTO = 'RETAIL' AND COLATERAL_TIPO IN ('NINGUNA', ''))
                _FLOOR_APLICADO = &_LGD_FLOOR_RETAIL_NONE.;

            /* — MORTGAGE without hipoteca — */
            WHEN (SEGMENTO = 'MORTGAGE' AND COLATERAL_TIPO IN ('NINGUNA', ''))
                _FLOOR_APLICADO = &_LGD_FLOOR_MORTG_NONE.;

            OTHERWISE _FLOOR_APLICADO = 0;
        END;

        /* Stage 3 multiplier */
        IF STAGE_IFRS9 = 3 THEN
            _FLOOR_APLICADO = _FLOOR_APLICADO * &_LGD_STAGE3_MULT.;

        /* Apply floor: LGD_ESTIMADA = max(LGD_ESTIMADA, floor) */
        LGD_PRE_FLOOR = LGD_ESTIMADA;
        IF _FLOOR_APLICADO > 0 AND LGD_ESTIMADA < _FLOOR_APLICADO THEN DO;
            LGD_ESTIMADA = _FLOOR_APLICADO;
            LGD_FLOOR_APLICADO = _FLOOR_APLICADO;
            LGD_FLOOR_TRIGGERED = 1;
        END;
        ELSE DO;
            LGD_FLOOR_APLICADO = LGD_ESTIMADA;
            LGD_FLOOR_TRIGGERED = 0;
        END;

        DROP _FLOOR_APLICADO;
    RUN;

    %put NOTE: [lgd_floors] Floors applied.;

%mend lgd_floors;


/* =========================================================================
 * 6. lgd_calculate — Compute LGD estimate with MoC
 * =========================================================================
 * Computes LGD from discounted recovery cashflows (when available)
 * or uses the existing LGD_ESTIMADA. Applies Margin of Conservatism.
 *
 * Input:  work.cycles_floored, work.cycle_npv
 * Output: work.cycles_lgd
 * ---------------------------------------------------------------------- */
%macro lgd_calculate;

    %put NOTE: [lgd_calculate] Computing LGD estimate with MoC.;

    /* Merge cycles with NPV recovery if available */
    DATA work.cycles_with_npv;
        MERGE work.cycles_floored (IN=a)
              work.cycle_npv (IN=b);
        BY CICLO_ID;
        IF a;
    RUN;

    /* Compute segment-level LGD means for MoC */
    PROC SQL;
        CREATE TABLE work.segment_lgd_mean AS
        SELECT
            SEGMENTO,
            MEAN(LGD_ESTIMADA) AS LGD_MEDIA_SEGMENTO,
            STD(LGD_ESTIMADA)  AS LGD_STD_SEGMENTO
        FROM work.cycles_with_npv
        GROUP BY SEGMENTO;
    QUIT;

    /* Final LGD calculation with MoC */
    DATA work.cycles_lgd;
        MERGE work.cycles_with_npv (IN=a)
              work.segment_lgd_mean (IN=b);
        BY SEGMENTO;
        IF a;

        /* — Margin of Conservatism — */
        LGD_DESVIACION = LGD_ESTIMADA - LGD_MEDIA_SEGMENTO;
        IF LGD_DESVIACION > 0 THEN
            MoC = &_LGD_MOC_RATE. * LGD_DESVIACION;
        ELSE
            MoC = &_LGD_MOC_FLOOR. * LGD_ESTIMADA;

        LGD_CON_MOC = LGD_ESTIMADA + MoC;

        /* — Discounted recovery LGD (if NPV data exists) — */
        IF NPV_RECOVERY > 0 THEN DO;
            LGD_NPV = 1 - (NPV_RECOVERY / EAD);
            LGD_NPV = MAX(0, MIN(1, LGD_NPV));   * clamp to [0, 1];
        END;
        ELSE
            LGD_NPV = .;

    RUN;

    %put NOTE: [lgd_calculate] LGD with MoC computed.;

%mend lgd_calculate;


/* =========================================================================
 * 7. lgd_ecl — Compute ECL with cure adjustment
 * =========================================================================
 * ECL = PD × LGD × EAD, with cure-rate adjustment and PD floor.
 *
 * Input:  work.cycles_lgd
 * Output: work.cycles_ecl
 * ---------------------------------------------------------------------- */
%macro lgd_ecl;

    %put NOTE: [lgd_ecl] Computing ECL = PD × LGD × EAD.;

    DATA work.cycles_ecl;
        SET work.cycles_lgd;

        /* PD floor — Art. 15 §27 */
        IF PD_ESTIMADA < &_LGD_PD_FLOOR. THEN
            PD_ESTIMADA = &_LGD_PD_FLOOR.;

        /* Art. 15 sequence:
           1. LGD_ajustada = LGD_ESTIMADA × (1 + MoC)
           2. LGD_efectiva = max(LGD_ajustada, LGD_FLOOR_APLICADO)
           3. ECL = PD × LGD_efectiva × EAD */
        LGD_EFECTIVA = MAX(LGD_CON_MOC, LGD_FLOOR_APLICADO);

        /* ECL with floored LGD */
        ECL = PD_ESTIMADA * LGD_EFECTIVA * EAD;

        /* Cure rate adjustment — Art. 23 */
        IF CURE_FLAG = 1 THEN DO;
            ECL_AJUSTADO = ECL * (1 - &_LGD_CURE_RATE.);
            CURE_RATE_APLICADO = &_LGD_CURE_RATE.;
        END;
        ELSE DO;
            ECL_AJUSTADO = ECL;
            CURE_RATE_APLICADO = 0;
        END;

    RUN;

    %put NOTE: [lgd_ecl] ECL computed.;

%mend lgd_ecl;


/* =========================================================================
 * 8. lgd_rwa — Compute RWA
 * =========================================================================
 * RWA = EAD × LGD_CON_MOC × 12.5 × K_IRB
 * K_IRB = N( (N^-1(PD) + rho^0.5 × N^-1(0.999)) / sqrt(1 - rho) ) - PD × LGD
 *
 * Simplified: K_IRB = sqrt(PD) × 0.06 + PD × 0.5
 *
 * Input:  work.cycles_ecl
 * Output: work.cycles_rwa
 * ---------------------------------------------------------------------- */
%macro lgd_rwa;

    %put NOTE: [lgd_rwa] Computing RWA.;

    DATA work.cycles_rwa;
        SET work.cycles_ecl;

        /* Simplified IRB risk weight function */
        K_IRB = SQRT(PD_ESTIMADA) * 0.06 + PD_ESTIMADA * 0.5;
        RWA = EAD * LGD_EFECTIVA * 12.5 * K_IRB;

        /* Capital at 8% of RWA */
        CAPITAL_REGULATORIO = RWA * 0.08;

        /* Leverage ratio denominator (simplified) */
        EXPOSICION_TOTAL = EAD;
    RUN;

    %put NOTE: [lgd_rwa] RWA computed.;

%mend lgd_rwa;


/* =========================================================================
 * 9. lgd_staging — IFRS 9 staging
 * =========================================================================
 * Stage assignment with 30-DPD backstop (IFRS 9 B5.5.12).
 * Stage 3 receives LGD floor multiplier.
 *
 * Input:  work.cycles_rwa
 * Output: work.cycles_staged
 * ---------------------------------------------------------------------- */
%macro lgd_staging;

    %put NOTE: [lgd_staging] Applying IFRS 9 staging.;

    DATA work.cycles_staged;
        SET work.cycles_rwa;

        /* Backstop: 30 DPD forces Stage 2 */
        IF DPDS >= &_LGD_BACKSTOP_DPD. AND STAGE_IFRS9 = 1 THEN DO;
            STAGE_IFRS9 = 2;
            STAGE_RECLASIFICADO = 1;
        END;
        ELSE
            STAGE_RECLASIFICADO = 0;

        /* Stage 3: PD = 1 (unit probability of default) */
        IF STAGE_IFRS9 = 3 THEN DO;
            PD_STAGE3 = PD_ESTIMADA;
            PD_ESTIMADA = 1.0;
            /* LGD floor already multiplied in lgd_floors via STAGE3_MULT */
        END;

        /* Re-compute ECL with updated PD (Stage 3) */
        ECL_STAGE_AJUSTADO = PD_ESTIMADA * LGD_EFECTIVA * EAD;
    RUN;

    %put NOTE: [lgd_staging] Staging complete.;

%mend lgd_staging;


/* =========================================================================
 * 10. lgd_validate — Flag non-conforming records
 * =========================================================================
 * Flags records failing regulatory checks.
 *
 * Input:  work.cycles_staged
 * Output: work.non_conformes
 * ---------------------------------------------------------------------- */
%macro lgd_validate;

    %put NOTE: [lgd_validate] Validating pipeline outputs.;

    DATA work.non_conformes;
        SET work.cycles_staged;
        LENGTH MOTIVO $500;
        FLAG_NC = 0;
        MOTIVO = '';

        /* Calibration window < 7 years — EBA GL 2017/16 §6.3 */
        IF VENTANA_CALIBRACION_YEARS < &_LGD_MIN_CALIB. THEN DO;
            FLAG_NC = 1;
            MOTIVO = CATX(' | ', MOTIVO,
                'Ventana calibracion < &_LGD_MIN_CALIB. anos');
        END;

        /* Observation window < 5 years */
        IF VENTANA_OBSERVACION_YEARS < &_LGD_MIN_OBS. THEN DO;
            FLAG_NC = 1;
            MOTIVO = CATX(' | ', MOTIVO,
                'Ventana observacion < &_LGD_MIN_OBS. anos');
        END;

        /* LGD below regulatory floor */
        IF LGD_FLOOR_TRIGGERED = 1 THEN DO;
            FLAG_NC = 1;
            MOTIVO = CATX(' | ', MOTIVO,
                'LGD estimada inferior al suelo regulatorio ('
                || TRIM(LEFT(PUT(LGD_FLOOR_APLICADO, 8.4))) || ')');
        END;

        /* ECL missing (should not happen after pipeline) */
        IF ECL = . THEN DO;
            FLAG_NC = 1;
            MOTIVO = CATX(' | ', MOTIVO, 'ECL no calculado');
        END;

        /* Fusion contract with missing LGD */
        IF SW_FUSION = 1 AND LGD_ESTIMADA = . THEN DO;
            FLAG_NC = 1;
            MOTIVO = CATX(' | ', MOTIVO,
                'LGD no informada en contrato fusionado');
        END;

        /* Stage 3 without proper PD */
        IF STAGE_IFRS9 = 3 AND PD_ESTIMADA < 1.0 THEN DO;
            FLAG_NC = 1;
            MOTIVO = CATX(' | ', MOTIVO,
                'Stage 3 sin PD unitaria');
        END;

        /* MOC gap > 10% of floor */
        IF LGD_FLOOR_APLICADO > 0
           AND LGD_EFECTIVA < LGD_FLOOR_APLICADO
           AND (LGD_FLOOR_APLICADO - LGD_EFECTIVA) / LGD_FLOOR_APLICADO > 0.10
        THEN DO;
            FLAG_NC = 1;
            MOTIVO = CATX(' | ', MOTIVO,
                'Diferencia LGD+MoC vs suelo > 10%');
        END;

        /* Output only non-conforming records */
        IF FLAG_NC = 1;
    RUN;

    %put NOTE: [lgd_validate] Validated. Output: work.non_conformes;

%mend lgd_validate;


/* =========================================================================
 * 11. lgd_report — Summary statistics and export
 * =========================================================================
 * Produces segment-level summaries and exports non-conformes to CSV.
 *
 * Input:  work.cycles_staged, work.non_conformes
 * Output: printed tables + CSV export
 * ---------------------------------------------------------------------- */
%macro lgd_report;

    %put NOTE: [lgd_report] Generating summary reports.;

    /* — Segment-level summary — */
    PROC MEANS DATA=work.cycles_staged N MEAN STD MIN MAX P1 P5 P25 P50 P75 P95 P99;
        VAR LGD_ESTIMADA LGD_CON_MOC LGD_EFECTIVA PD_ESTIMADA
            EAD ECL ECL_AJUSTADO RWA CAPITAL_REGULATORIO MoC LGD_FLOOR_APLICADO;
        CLASS SEGMENTO;
        TITLE "LGD Pipeline — Summary Statistics by Segment";
    RUN;

    /* — Stage distribution — */
    PROC FREQ DATA=work.cycles_staged;
        TABLES STAGE_IFRS9 * SEGMENTO / NOROW NOCOL NOPERCENT;
        TITLE "IFRS 9 Stage Distribution by Segment";
    RUN;

    /* — Floor trigger analysis — */
    PROC FREQ DATA=work.cycles_staged;
        TABLES LGD_FLOOR_TRIGGERED * SEGMENTO / NOROW NOCOL NOPERCENT;
        TITLE "LGD Floor Triggered by Segment";
    RUN;

    /* — Calibration window quality — */
    PROC MEANS DATA=work.cycles_staged N MEAN STD MIN MAX;
        VAR VENTANA_CALIBRACION_YEARS VENTANA_OBSERVACION_YEARS;
        CLASS SEGMENTO;
        TITLE "Calibration and Observation Windows by Segment";
    RUN;

    /* — Export non-conformes — */
    %if %sysfunc(exist(work.non_conformes)) %then %do;
        PROC EXPORT DATA=work.non_conformes
            OUTFILE="&_LGD_OUT_NC."
            DBMS=CSV REPLACE;
        RUN;
        %put NOTE: [lgd_report] Non-conformes exported to &_LGD_OUT_NC.;
    %end;
    %else %do;
        %put NOTE: [lgd_report] No non-conforming records found.;
    %end;

    /* — Export segment summary — */
    PROC SQL;
        CREATE TABLE work.lgd_summary_export AS
        SELECT
            SEGMENTO,
            COUNT(*) AS N_CONTRATOS,
            SUM(EAD) AS EAD_TOTAL,
            MEAN(LGD_ESTIMADA) AS LGD_MEDIA,
            MEAN(LGD_CON_MOC) AS LGD_CON_MOC_MEDIA,
            MEAN(PD_ESTIMADA) AS PD_MEDIA,
            SUM(ECL) AS ECL_TOTAL,
            SUM(ECL_AJUSTADO) AS ECL_AJUSTADO_TOTAL,
            SUM(RWA) AS RWA_TOTAL,
            SUM(CAPITAL_REGULATORIO) AS CAPITAL_REGULATORIO_TOTAL,
            MEAN(VENTANA_CALIBRACION_YEARS) AS VENTANA_CALIB_MEDIA,
            MEAN(VENTANA_OBSERVACION_YEARS) AS VENTANA_OBSERV_MEDIA
        FROM work.cycles_staged
        GROUP BY SEGMENTO;
    QUIT;

    PROC EXPORT DATA=work.lgd_summary_export
        OUTFILE="&_LGD_OUT_SUM."
        DBMS=CSV REPLACE;
    RUN;
    %put NOTE: [lgd_report] Summary exported to &_LGD_OUT_SUM.;

%mend lgd_report;


/* =========================================================================
 * lgd_run_all — Execute the full pipeline end-to-end
 * =========================================================================
 * Convenience macro that calls all steps in order and reports timing.
 * ---------------------------------------------------------------------- */
%macro lgd_run_all;

    %local _start _step_start;
    %let _start = %sysfunc(datetime());

    %put NOTE: ============================================;
    %put NOTE: LGD Pipeline — Full Run Started;
    %put NOTE: ============================================;
    %put;

    %let _step_start = %sysfunc(datetime());
    %lgd_init
    %put NOTE: Timing: lgd_init = %sysfunc(datetime() - &_step_start., 8.2) sec;
    %put;

    %let _step_start = %sysfunc(datetime());
    %lgd_load
    %put NOTE: Timing: lgd_load = %sysfunc(datetime() - &_step_start., 8.2) sec;
    %put;

    %let _step_start = %sysfunc(datetime());
    %lgd_gen_cashflows
    %put NOTE: Timing: lgd_gen_cashflows = %sysfunc(datetime() - &_step_start., 8.2) sec;
    %put;

    %let _step_start = %sysfunc(datetime());
    %lgd_discount
    %put NOTE: Timing: lgd_discount = %sysfunc(datetime() - &_step_start., 8.2) sec;
    %put;

    %let _step_start = %sysfunc(datetime());
    %lgd_floors
    %put NOTE: Timing: lgd_floors = %sysfunc(datetime() - &_step_start., 8.2) sec;
    %put;

    %let _step_start = %sysfunc(datetime());
    %lgd_calculate
    %put NOTE: Timing: lgd_calculate = %sysfunc(datetime() - &_step_start., 8.2) sec;
    %put;

    %let _step_start = %sysfunc(datetime());
    %lgd_ecl
    %put NOTE: Timing: lgd_ecl = %sysfunc(datetime() - &_step_start., 8.2) sec;
    %put;

    %let _step_start = %sysfunc(datetime());
    %lgd_rwa
    %put NOTE: Timing: lgd_rwa = %sysfunc(datetime() - &_step_start., 8.2) sec;
    %put;

    %let _step_start = %sysfunc(datetime());
    %lgd_staging
    %put NOTE: Timing: lgd_staging = %sysfunc(datetime() - &_step_start., 8.2) sec;
    %put;

    %let _step_start = %sysfunc(datetime());
    %lgd_validate
    %put NOTE: Timing: lgd_validate = %sysfunc(datetime() - &_step_start., 8.2) sec;
    %put;

    %let _step_start = %sysfunc(datetime());
    %lgd_report
    %put NOTE: Timing: lgd_report = %sysfunc(datetime() - &_step_start., 8.2) sec;
    %put;

    %put NOTE: ============================================;
    %put NOTE: LGD Pipeline — Full Run Complete;
    %put NOTE: Total time: %sysfunc(datetime() - &_start., 8.2) sec;
    %put NOTE: ============================================;

%mend lgd_run_all;

/*****************************************************************************
 * LGD Pipeline — V3 Driver
 * ========================
 * Runs the full LGD estimation pipeline with V3 regulatory parameters.
 *
 * V3 changes from V2:
 *   1. CORP LGD floor raised from 0.45 → 0.50
 *   2. PD floor raised from 0.03% → 0.05%
 *   3. New titulizacion step producing OR_EAD_TIT
 *   4. Non-conformes threshold updated to new PD floor
 *****************************************************************************/

/* Include macro library */
%include "/data/sas/pipeline/lgd_macros.sas";

/* ── V3 Configuration ──────────────────────────────────────────────── */
%lgd_init(
    lib_in     = mylib,
    lib_out    = work,

    /* LGD floors — V3: CORP raised to 0.50 */
    floor_hipoteca    = 0.30,
    floor_corp_none   = 0.50,     /* V3: +5pp from 0.45 */
    floor_corp_fin    = 0.35,
    floor_retail_pers = 0.35,
    floor_retail_none = 0.45,
    floor_mortg_none  = 0.40,
    floor_mortg_hip   = 0.25,

    /* PD floor — V3: 0.05% (up from 0.03%) */
    pd_floor = 0.0005,

    /* MoC */
    moc_rate       = 0.10,
    moc_floor_rate = 0.05,

    /* Cure */
    cure_rate = 0.15,

    /* Validation windows */
    min_calib_years = 7,
    min_obs_years   = 5,

    /* Backstop */
    backstop_dpd = 30,

    /* Output paths */
    out_csv_no_conformes = /data/irb/output/v3_no_conformes_lgd.csv,
    out_csv_summary      = /data/irb/output/v3_resumen_lgd.csv
);

/* ── Pipeline Steps ─────────────────────────────────────────────────- */
%lgd_load;
%lgd_gen_cashflows;
%lgd_discount;
%lgd_floors;
%lgd_calculate;
%lgd_ecl;
%lgd_rwa;
%lgd_staging;

/* ── V3-Specific: Titulizacion step ────────────────────────────────── */
/* Computes OR_EAD_TIT from EAD and segment-level multipliers.
   New in V3 per Circular 4/2022 securitisation framework. */

%put NOTE: [V3] Computing OR_EAD_TIT (titulizacion, new in V3).;

DATA work.titulizado;
    SET work.cycles_staged;

    /* Regulatory multiplier for securitised tranches */
    SELECT (SEGMENTO);
        WHEN ('CORP')    OR_EAD_TIT = EAD * 2.0;
        WHEN ('RETAIL')  OR_EAD_TIT = EAD * 1.5;
        WHEN ('MORTGAGE') OR_EAD_TIT = EAD * 1.2;
        WHEN ('SME')     OR_EAD_TIT = EAD * 1.0;
        OTHERWISE        OR_EAD_TIT = EAD;
    END;
RUN;

%put NOTE: [V3] OR_EAD_TIT computed.;

/* ── Validation and reporting ──────────────────────────────────────── */
%lgd_validate;
%lgd_report;

%put NOTE: [V3] LGD pipeline complete (V3 parameters).;

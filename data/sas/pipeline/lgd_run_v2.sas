/*****************************************************************************
 * LGD Pipeline — V2 Driver
 * ========================
 * Runs the full LGD estimation pipeline with V2 regulatory parameters.
 *
 * V2 parameters:
 *   CORP LGD floor  : 0.45  (CRR Art. 161(1)(b))
 *   HIPOTECA floor  : 0.30  (CRR Art. 154(3))
 *   PD floor        : 0.03% (CRR Art. 160(1))
 *   MoC rate       : 10% of LGD deviation from segment mean
 *   Cure rate      : 15%
 *****************************************************************************/

/* Include macro library */
%include "/data/sas/pipeline/lgd_macros.sas";

/* ── V2 Configuration ──────────────────────────────────────────────── */
%lgd_init(
    lib_in     = mylib,
    lib_out    = work,

    /* LGD floors — V2 */
    floor_hipoteca    = 0.30,
    floor_corp_none   = 0.45,
    floor_corp_fin    = 0.35,
    floor_retail_pers = 0.35,
    floor_retail_none = 0.45,
    floor_mortg_none  = 0.40,
    floor_mortg_hip   = 0.25,

    /* PD floor — V2: 0.03% */
    pd_floor = 0.0003,

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
    out_csv_no_conformes = /data/irb/output/v2_no_conformes_lgd.csv,
    out_csv_summary      = /data/irb/output/v2_resumen_lgd.csv
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
%lgd_validate;
%lgd_report;

/* ── V2-Specific: no titulizacion step ───────────────────────────────- */
/* In V3, a titulizacion step is inserted here. In V2, the pipeline
   stops at reporting — no OR_EAD_TIT is produced. */

%put NOTE: [V2] LGD pipeline complete (V2 parameters).;

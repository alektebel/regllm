/*****************************************************************************
 * regllm_macros.sas
 * ==================
 * Libreria de 15 macros para el pipeline IRB extendido.
 * Incluye: init, floors, staging, ECL, IRB K, CCF, scoring, NOD, DPD.
 *****************************************************************************/

/* ── 1. %regllm_init ─────────────────────────────────────────────────── */
%macro regllm_init(lib=);
    %put NOTE: [regllm_init] Inicializando libreria &lib..;
    LIBNAME &lib CLEAR;
    LIBNAME &lib "data/sas/extended/data";
    %put NOTE: [regllm_init] Libreria &lib asignada.;
%mend regllm_init;

/* ── 2. %segment_pd_floor ───────────────────────────────────────────── */
%macro segment_pd_floor(segmento);
    %local floor;
    %if &segmento = CORP %then %let floor = 0.0005;
    %else %if &segmento = RETAIL %then %let floor = 0.0003;
    %else %if &segmento = MORTGAGE %then %let floor = 0.0005;
    %else %if &segmento = SME %then %let floor = 0.0005;
    %else %let floor = 0;
    &floor
%mend segment_pd_floor;

/* ── 3. %lgd_floor ──────────────────────────────────────────────────── */
%macro lgd_floor(segmento, colateral_tipo);
    %local floor;
    %if &segmento = CORP and &colateral_tipo = FINANCIERO %then %let floor = 0.35;
    %else %if &segmento = CORP and &colateral_tipo = NINGUNA %then %let floor = 0.50;
    %else %if &segmento = RETAIL and &colateral_tipo = PERSONAL %then %let floor = 0.35;
    %else %if &segmento = RETAIL and &colateral_tipo = NINGUNA %then %let floor = 0.45;
    %else %if &segmento = MORTGAGE and &colateral_tipo = HIPOTECA %then %let floor = 0.25;
    %else %if &segmento = MORTGAGE and &colateral_tipo = NINGUNA %then %let floor = 0.40;
    %else %let floor = 0;
    &floor
%mend lgd_floor;

/* ── 4. %determine_stage ────────────────────────────────────────────── */
%macro determine_stage(pd, dpd, forbearance, sicr);
    %local stage;
    %if &dpd > 90 or &forbearance = 1 %then %let stage = 3;
    %else %if &dpd > 30 or &sicr = 1 %then %let stage = 2;
    %else %let stage = 1;
    &stage
%mend determine_stage;

/* ── 5. %ecl_12m ────────────────────────────────────────────────────── */
%macro ecl_12m(pd, lgd, ead);
    %sysevalf(&pd * &lgd * &ead)
%mend ecl_12m;

/* ── 6. %ecl_lifetime ───────────────────────────────────────────────── */
%macro ecl_lifetime(pd, lgd, ead, years_to_maturity);
    %local years;
    %let years = %sysevalf(&years_to_maturity);
    %if &years > 5 %then %let years = 5;
    %sysevalf(&pd * &lgd * &ead * &years)
%mend ecl_lifetime;

/* ── 7. %irb_k ──────────────────────────────────────────────────────── */
%macro irb_k(pd, lgd, segmento, madurez=2.5);
    %local correl k;
    %if &segmento = CORP %then %let correl = 0.24;
    %else %if &segmento = RETAIL %then %let correl = 0.15;
    %else %if &segmento = MORTGAGE %then %let correl = 0.15;
    %else %if &segmento = SME %then %let correl = 0.20;
    %else %let correl = 0.24;
    %let k = %sysevalf(%sysfunc(sqrt(&pd)) * 0.06 + &pd * 0.5);
    %sysevalf(&k * (1 + (&madurez - 2.5) * (1 - 2 * &correl)))
%mend irb_k;

/* ── 8. %ead_ccf ────────────────────────────────────────────────────── */
%macro ead_ccf(ccf_base, usage_rate);
    %local adj;
    %if &usage_rate < 0.3 %then %let adj = 0.7;
    %else %if &usage_rate > 0.8 %then %let adj = 1.3;
    %else %let adj = 1.0;
    %sysevalf(&ccf_base * &adj)
%mend ead_ccf;

/* ── 9. %score_financial ────────────────────────────────────────────── */
%macro score_financial(ebitda, deuda_neta, activo, patrimonio);
    %local ratio_deuda ratio_lev ratio_ebitda score;
    %if &activo > 0 %then %let ratio_deuda = %sysevalf(&deuda_neta / &activo);
    %else %let ratio_deuda = 1;
    %if &patrimonio > 0 %then %let ratio_lev = %sysevalf(&deuda_neta / &patrimonio);
    %else %let ratio_lev = 5;
    %if &ebitda > 0 %then %let ratio_ebitda = %sysevalf(&ebitda / (&deuda_neta + 1));
    %else %let ratio_ebitda = 0;
    %let score = %sysevalf(
        max(0, min(100,
            (1 - &ratio_deuda) * 30 +
            (1 - min(&ratio_lev, 5) / 5) * 30 +
            min(&ratio_ebitda, 2) / 2 * 40
        ))
    );
    &score
%mend score_financial;

/* ── 10. %score_behavioral ───────────────────────────────────────────── */
%macro score_behavioral(dias_mora_avg, pago_prompto_pct, tenure_months);
    %local score;
    %let score = %sysevalf(
        max(0, min(100,
            (1 - min(&dias_mora_avg, 180) / 180) * 40 +
            &pago_prompto_pct * 30 +
            min(&tenure_months, 60) / 60 * 30
        ))
    );
    &score
%mend score_behavioral;

/* ── 11. %score_collateral ───────────────────────────────────────────── */
%macro score_collateral(valor_garantia, ead, colateral_tipo);
    %local ratio score_base;
    %if &ead > 0 %then %let ratio = %sysevalf(&valor_garantia / &ead);
    %else %let ratio = 0;
    %let score_base = %sysevalf(min(&ratio, 2) / 2 * 70);
    %if &colateral_tipo = HIPOTECA %then %let score_base = %sysevalf(&score_base + 20);
    %else %if &colateral_tipo = FINANCIERO %then %let score_base = %sysevalf(&score_base + 15);
    %else %if &colateral_tipo = PERSONAL %then %let score_base = %sysevalf(&score_base + 5);
    %let score_base = %sysevalf(min(100, max(0, &score_base)));
    &score_base
%mend score_collateral;

/* ── 12. %pd_from_score ──────────────────────────────────────────────── */
%macro pd_from_score(score, intercept=2.5, slope=-0.04);
    %local logit pd;
    %let logit = %sysevalf(&intercept + (&slope * &score));
    %let pd = %sysevalf(1 / (1 + %sysfunc(exp(-&logit))));
    &pd
%mend pd_from_score;

/* ── 13. %rating_from_pd ──────────────────────────────────────────────── */
%macro rating_from_pd(pd);
    %local rating;
    %if &pd <= 0.0003 %then %let rating = AAA;
    %else %if &pd <= 0.0010 %then %let rating = AA;
    %else %if &pd <= 0.0030 %then %let rating = A;
    %else %if &pd <= 0.0100 %then %let rating = BBB;
    %else %if &pd <= 0.0300 %then %let rating = BB;
    %else %if &pd <= 0.0700 %then %let rating = B;
    %else %if &pd <= 0.1500 %then %let rating = CCC;
    %else %let rating = D;
    &rating
%mend rating_from_pd;

/* ── 14. %detect_nod ─────────────────────────────────────────────────── */
%macro detect_nod(dpd, forbearance, insolvencia);
    %local nod_flag cure_status;
    %if &dpd > 90 or &forbearance = 1 or &insolvencia = 1 %then %let nod_flag = 1;
    %else %let nod_flag = 0;

    %if &nod_flag = 1 %then %do;
        %if &dpd > 365 %then %let cure_status = PERFORMING;
        %else %if &dpd > 180 %then %let cure_status = MONITORING;
        %else %if &dpd > 90 %then %let cure_status = PROBATION;
        %else %let cure_status = IN_DEFAULT;
    %end;
    %else %let cure_status = PERFORMING;

    nod_flag=&nod_flag cure_status=&cure_status
%mend detect_nod;

/* ── 15. %dpd_bucket ─────────────────────────────────────────────────── */
%macro dpd_bucket(dpd);
    %local bucket;
    %if &dpd = 0 %then %let bucket = 0;
    %else %if &dpd <= 29 %then %let bucket = 1-29;
    %else %if &dpd <= 59 %then %let bucket = 30-59;
    %else %if &dpd <= 89 %then %let bucket = 60-89;
    %else %if &dpd <= 179 %then %let bucket = 90-179;
    %else %let bucket = 180+;
    &bucket
%mend dpd_bucket;

%put NOTE: [regllm_macros] 15 macros cargadas correctamente.;

/*══════════════════════════════════════════════════════════════════════════
 * TABBERT EMBEDDING QUALITY ANALYSIS  —  SAS
 *
 * Reads the embedding CSV (sas_embeddings.csv) and produces diagnostics:
 *
 *   1. Dimension health     — degenerate / constant / extreme dims
 *   2. Outlier detection    — Euclidean distance from centroid
 *   3. Nearest-neighbor     — min distance to any other register
 *   4. PCA                  — eigenvalue spectrum, variance explained
 *   5. Group separation     — how well each metadata field is separated
 *                              by the embedding space (ANOVA-like ratio)
 *
 * Usage:
 *   sas scripts/analyze_sas_embeddings.sas \
 *       -log data/logs/embedding_analysis.log
 *
 * Input:
 *   data/embeddings/tabular/sas_embeddings.csv
 *
 * Output in data/embeddings/tabular/analysis/:
 *   dimension_stats.csv      — per-dimension min/max/mean/std
 *   outlier_registers.csv    — top 1 % farthest from centroid + metadata
 *   nn_outliers.csv          — registers with unusually large NN distance
 *   pca_eigenvalues.csv      — eigenvalues and variance % (top 20)
 *   group_separation.csv     — ANOVA-like F-ratio per metadata field
 *   analysis_summary.txt     — human-readable report
 *════════════════════════════════════════════════════════════════════════*/

%let ROOT       = ../../data/embeddings/tabular;
%let INPUT_CSV  = &ROOT./sas_embeddings.csv;
%let OUTDIR     = &ROOT./analysis;
%let OUTLIER_PCT= 1;        /* top N% farthest from centroid */
%let NN_SAMPLE  = 500;      /* query points for NN analysis */

title "TabBERT Embedding Quality Analysis";
footnote "Started: &sysdate &:&systime";

/* ── 0. Create output directory (SAS 9.4+ uses DLCREATEDIR) ──────────── */
options dlcreatedir;
libname out "&OUTDIR.";
options nodlcreatedir;

/* ── 1. Import embeddings ────────────────────────────────────────────── */
proc import datafile="&INPUT_CSV."
    out=embeddings dbms=csv replace;
    getnames=yes;
    guessingrows=10000;
run;

/* ── 2. Auto-detect embedding columns (EMB_*) and metadata ───────────── */
proc sql noprint;
    select name into :emb_cols separated by ' '
    from dictionary.columns
    where libname='WORK' and memname='EMBEDDINGS'
      and upcase(name) like 'EMB\_%' escape '\'
    order by input(scan(name, 2, '_'), best.);

    select name into :meta_cols separated by ' '
    from dictionary.columns
    where libname='WORK' and memname='EMBEDDINGS'
      and upcase(name) not like 'EMB\_%' escape '\'
      and upcase(name) ne 'ROW_ID';
quit;

%let n_emb = %sysfunc(countw(&emb_cols.));
%let n_meta = %sysfunc(countw(&meta_cols.));
%put NOTE: [TabBERT-QA] Found &n_emb. embedding dims, &n_meta. metadata cols;
%put NOTE: [TabBERT-QA] Metadata: &meta_cols.;

/* ── 3. Dimension health stats (IML avoids autoname complexity) ───────── */
proc iml;
    use embeddings;
    read all var {&emb_cols.} into E;
    close embeddings;

    n = nrow(E);
    p = ncol(E);

    create dim_stats var {"dim_no" "min_v" "max_v" "mean_v" "std_v" "range_v"};
    do i = 1 to p;
        dim_no = i;
        min_v = min(E[,i]);
        max_v = max(E[,i]);
        mean_v = mean(E[,i]);
        std_v = std(E[,i]);
        range_v = max_v - min_v;
        append;
    end;
    close dim_stats;
quit;

proc export data=dim_stats
    outfile="&OUTDIR./dimension_stats.csv"
    dbms=csv replace;
    putnames=yes;
run;

/* Flag degenerate dimensions */
data _null_;
    set dim_stats end=last;
    file "&OUTDIR./analysis_summary.txt";
    retain n_warn 0;
    if _n_ = 1 then do;
        put "═══════════════════════════════════════════════════════";
        put "  TABBERT EMBEDDING QUALITY ANALYSIS";
        put "  &sysdate &:&systime";
        put "═══════════════════════════════════════════════════════";
        put / "Dataset: &INPUT_CSV.";
        put "  Embedding dims: &n_emb.  Metadata cols: &n_meta.";
        put "═══════════════════════════════════════════════════════";
    end;
    length varname $32;
    varname = scan("&emb_cols.", dim_no);

    if std_v = 0 then do;
        if n_warn = 0 then
            put / "! WARNING — degenerate dimensions (std = 0):";
        put "  " varname " [constant: " min_v "]" ;
        n_warn + 1;
    end;
    else if range_v < 1e-10 then do;
        if n_warn = 0 then
            put / "! WARNING — near-constant dimensions (range < 1e-10):";
        put "  " varname " [range: " range_v "]" ;
        n_warn + 1;
    end;

    if last then do;
        if n_warn = 0 then
            put / "  All dimensions healthy — no degenerate dims found.";
    end;
run;

/* ── 4. Outlier detection: distance from centroid (PROC IML) ──────────── */
proc iml;
    use embeddings;
    read all var {&emb_cols.} into E;
    close embeddings;

    n = nrow(E);
    p = ncol(E);

    /* Centroid */
    centroid = mean(E);
    centered = E - repeat(centroid, n);

    /* Euclidean distance from centroid */
    dist = sqrt((centered # centered)[, +]);

    /* Flag top N% (ascending rank — largest dist gets highest rank) */
    outlier_pct = &OUTLIER_PCT. / 100;
    r = rank(dist);
    threshold_pos = n - ceil(outlier_pct * n) + 1;
    is_outlier = (r >= threshold_pos);
    outlier_rank = n - r + 1;   /* convert to descending order (1 = farthest) */

    /* Create output dataset (column name ROW_ID matches embeddings table) */
    ROW_ID = T(1:n);

    create _outliers var {"ROW_ID" "dist" "outlier_flag" "outlier_rank"};
    append;
    close _outliers;

    /* ── 5. Nearest-neighbor analysis (sampled) ───────────────────────── */
    call randseed(42);
    nn_sample = min(&NN_SAMPLE., n);
    query_idx = sample(1:n, nn_sample);

    min_dist = j(nn_sample, 1, .);

    do i = 1 to nn_sample;
        qi = query_idx[i];
        diff = E - repeat(E[qi,], n);
        d = sqrt((diff # diff)[, +]);
        d[qi] = .;          /* exclude self */
        min_dist[i] = min(d);
    end;

    create _nn_dist var {"query_idx" "min_dist"};
    append;
    close _nn_dist;

    /* Flag NN outliers: top 1 % with largest min-dist (ascending rank) */
    nn_r = rank(min_dist);
    nn_threshold = nn_sample - ceil(outlier_pct * nn_sample) + 1;
    nn_outlier = (nn_r >= nn_threshold);
    nn_outlier_rank = nn_sample - nn_r + 1;

    create _nn_outliers var {"query_idx" "min_dist" "nn_outlier_rank"};
    append;
    close _nn_outliers;

    /* ── 6. PCA (via eigenvalue decomposition of covariance) ──────────── */
    cov = cov(E);
    call eigen(evals, evecs, cov);

    /* Variance explained */
    total_var = sum(evals);
    pct_var = (evals / total_var) * 100;
    cum_var = cusum(pct_var);

    /* Write top 20 eigenvalues */
    k = min(20, p);
    comp = T(1:k);
    create _pca var {"comp" "evals" "pct_var" "cum_var"};
    do i = 1 to k;
        ev = evals[i];
        pct = pct_var[i];
        cum = cum_var[i];
        append;
    end;
    close _pca;

    /* ── 7. Summary to log ────────────────────────────────────────────── */
    print "═══════════════════════════════════════════════════════";
    print "  PCA: First component explains " (pct_var[1]) "% of variance";
    print "  Top 5 components: " (pct_var[1:5]);
    print "  Num components for 95 %: " (loc(cum_var >= 95)[1]);
    print "═══════════════════════════════════════════════════════";
    print "  Outliers: " sum(is_outlier) "of" n "(" outlier_pct*100 "%)";
    print "═══════════════════════════════════════════════════════";
quit;

/* Export outlier registers with metadata */
proc sql;
    create table outlier_regs as
        select o.*, e.*
        from _outliers o
        left join embeddings e on o.ROW_ID = e.ROW_ID
        where o.outlier_flag = 1
        order by o.outlier_rank;
    drop table _outliers;
quit;

proc export data=outlier_regs
    outfile="&OUTDIR./outlier_registers.csv"
    dbms=csv replace;
    putnames=yes;
run;

proc export data=_nn_outliers
    outfile="&OUTDIR./nn_outliers.csv"
    dbms=csv replace;
    putnames=yes;
run;

/* ── 8. Group separation analysis ───────────────────────────────────────
 * For each metadata field with 2-20 levels, compute the multivariate R²
 * (proportion of embedding variance explained by that field).
 *
 * R² = 1 - SS_within / SS_total   averaged across all embedding dims.
 * Higher R² → that metadata field is well-separated in embedding space.
 *─────────────────────────────────────────────────────────────────────────*/

/* Identify categorical metadata fields */
proc sql noprint;
    select name into :group_fields separated by ' '
    from dictionary.columns
    where libname='WORK' and memname='EMBEDDINGS'
      and upcase(name) not like 'EMB\_%' escape '\'
      and upcase(name) not in ('ROW_ID')
      and upcase(name) not like '%ID%'
      and upcase(name) not like '%FECHA%'
      and upcase(name) not like '%DATE%';
quit;
%let ngf = %sysfunc(countw(&group_fields.));
%let n_obs = 0;

/* Count total observations */
proc sql noprint;
    select count(*) into :n_obs from embeddings;
quit;

/* Pre-compute total corrected SS per dimension (once) */
proc means data=embeddings noprint;
    var &emb_cols.;
    output out=_total_css css= / autoname;
run;

/* Create empty result table */
data _group_sep;
    length field $64;
    stop;
run;

/* Macro: analyze one field at a time */
%macro _analyze_field(field);
    /* Within-group corrected SS per dimension, summed across groups */
    proc means data=embeddings noprint nway;
        class &field.;
        var &emb_cols.;
        output out=_wg_css css= / autoname;
    run;

    proc means data=_wg_css noprint;
        var &emb_cols._css;
        output out=_wg_sum sum= / autoname;
    run;

    /* Compute avg R² in IML */
    proc iml;
        use _total_css;
        read all var _num_ into total_css;
        close _total_css;
        use _wg_sum;
        read all var _num_ into wg_css;
        close _wg_sum;
        n_dim = ncol(total_css);
        sum_rsq = 0;
        do i = 1 to n_dim;
            if total_css[i] > 0 then
                sum_rsq = sum_rsq + (1 - wg_css[i] / total_css[i]);
        end;
        avg_rsq = sum_rsq / n_dim;
        call symputx("avg_rsq", avg_rsq);
    quit;

    /* Count groups */
    proc sql noprint;
        select count(distinct &field.) into :ng
        from embeddings
        where &field. is not null;
    quit;

    /* Build result row */
    data _row;
        length field $64;
        avg_rsq = &avg_rsq.;
        n_groups = &ng.;
        rsq = avg_rsq;
        df_num = max(1, n_groups - 1);
        df_den = max(1, &n_obs. - n_groups);
        f_ratio = (rsq / df_num) / ((1 - rsq) / df_den);
        field = "&field.";
        keep field n_groups rsq f_ratio;
    run;

    proc append base=_group_sep data=_row force; run;

    proc datasets lib=work nolist;
        delete _wg_css _wg_sum _row;
    run; quit;
%mend;

%let _i = 1;
%do %while(%scan(&group_fields., &_i.) ne );
    %let f = %scan(&group_fields., &_i.);
    %_analyze_field(&f.);
    %let _i = %eval(&_i. + 1);
%end;

proc sort data=_group_sep; by descending f_ratio; run;

proc export data=_group_sep
    outfile="&OUTDIR./group_separation.csv"
    dbms=csv replace;
    putnames=yes;
run;

/* ── 9. Export PCA eigenvalues ────────────────────────────────────────── */
proc export data=_pca
    outfile="&OUTDIR./pca_eigenvalues.csv"
    dbms=csv replace;
    putnames=yes;
run;

/* ── 10. Append key findings to summary ────────────────────────────────── */
proc sql;
    /* NN summary */
    select
        count(*)         as N_QUERIES,
        mean(min_dist)   as MEAN_NN_DIST,
        std(min_dist)    as STD_NN_DIST,
        max(min_dist)    as MAX_NN_DIST
    from _nn_dist;

    /* Outlier count */
    select count(*) as N_OUTLIERS from outlier_regs;

    /* Top separated field */
    select field, f_ratio, rsq as PCT_VAR_EXPLAINED
    from _group_sep(obs=1);
quit;

data _null_;
    file "&OUTDIR./analysis_summary.txt" mod;
    put / "═══════════════════════════════════════════════════════";
    put "  ANALYSIS COMPLETE";
    put "  Output directory: &OUTDIR.";
    put "═══════════════════════════════════════════════════════";
    put / "Output files:";
    put "  dimension_stats.csv     — per-dimension health check";
    put "  outlier_registers.csv   — top &OUTLIER_PCT.% extreme registers";
    put "  nn_outliers.csv         — registers with largest NN distance";
    put "  pca_eigenvalues.csv     — eigenvalue spectrum (top 20)";
    put "  group_separation.csv    — ANOVA-like F-ratio per metadata field";
    put "  analysis_summary.txt    — this file";
    put / "═══════════════════════════════════════════════════════";
run;

/* Cleanup */
proc datasets lib=work nolist;
    delete embeddings dim_stats _outliers _nn_dist _nn_outliers _pca _group_sep outlier_regs _total_css;
run;
quit;

title;
footnote;

%put NOTE: [TabBERT-QA] Analysis complete — outputs in &OUTDIR.;

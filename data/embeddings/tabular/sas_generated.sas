/* ── Auto-generated SAS embedding code ────────────────── */
%let N_FEATURES = 69;
%let N_CAT_COLS = 8;
%let N_NUM_COLS = 28;
%let EMBED_DIM = 50;

proc format;
  invalue _SEGMENTO_
    'CORP' = 1
    'MORTGAGE' = 2
    'RETAIL' = 3
    'SME' = 4
    other = 0;

  invalue _CALIBRATION_SEGMENT_
    'CORP_LARGE' = 1
    'CORP_MID' = 2
    'CORP_SMALL' = 3
    'MORTGAGE_LTV_HIGH' = 4
    'MORTGAGE_LTV_LOW' = 5
    'MORTGAGE_LTV_MED' = 6
    'RETAIL_OTHER' = 7
    'RETAIL_REVOLVING' = 8
    'SME_MEDIUM' = 9
    'SME_MICRO' = 10
    'SME_SMALL' = 11
    other = 0;

  invalue _PRODUCTO_
    'HIPOTECA' = 1
    'LINEA_CREDITO' = 2
    'PRESTAMO' = 3
    'TARJETA' = 4
    other = 0;

  invalue _TERMINACION_
    'DACCION_PAGO' = 1
    'FALLIDO' = 2
    'LIQUIDACION' = 3
    'PAGO_VOLUNTARIO' = 4
    'SUBASTA' = 5
    'VENTA_CARTERA' = 6
    other = 0;

  invalue _CAUSA_DEFAULT_
    '90_DIAS_VENCIDO' = 1
    'IMPROBABLE_PAGO' = 2
    'QUIEBRA' = 3
    other = 0;

  invalue _COLATERAL_TIPO_
    'FINANCIERO' = 1
    'GARANTIA_PERSONAL' = 2
    'HIPOTECA' = 3
    'NINGUNA' = 4
    'PRENDA' = 5
    other = 0;

  invalue _ADJUDICACION_TIPO_
    'DACION_PAGO' = 1
    'SUBASTA_INMOBILIARIA' = 2
    'SUBASTA_MUEBLE' = 3
    'VENTA_EXTRAJUDICIAL' = 4
    'VENTA_JUDICIAL' = 5
    other = 0;

  invalue _ESTADO_CICLO_
    'CERRADO' = 1
    'ESTIMACION' = 2
    'RECUPERACION' = 3
    other = 0;

run;

data _features;
    set cycles;

    array _f[69] _temporary_;
    do _i = 1 to 69; _f[_i] = 0; end;

    _code = input(SEGMENTO, _SEGMENTO_.);
    if _code > 0 then _f[1 + _code - 1] = 1;
    _code = input(CALIBRATION_SEGMENT, _CALIBRATION_SEGMENT_.);
    if _code > 0 then _f[5 + _code - 1] = 1;
    _code = input(PRODUCTO, _PRODUCTO_.);
    if _code > 0 then _f[16 + _code - 1] = 1;
    _code = input(TERMINACION, _TERMINACION_.);
    if _code > 0 then _f[20 + _code - 1] = 1;
    _code = input(CAUSA_DEFAULT, _CAUSA_DEFAULT_.);
    if _code > 0 then _f[26 + _code - 1] = 1;
    _code = input(COLATERAL_TIPO, _COLATERAL_TIPO_.);
    if _code > 0 then _f[29 + _code - 1] = 1;
    _code = input(ADJUDICACION_TIPO, _ADJUDICACION_TIPO_.);
    if _code > 0 then _f[34 + _code - 1] = 1;
    _code = input(ESTADO_CICLO, _ESTADO_CICLO_.);
    if _code > 0 then _f[39 + _code - 1] = 1;
    _f[42] = (input(OR_DISPTO, best32.) - 3089.92) / 9971251.9;
    _f[43] = (input(VALOR_COLATERAL_INICIAL, best32.) - 0) / 7331853.2;
    _f[44] = (input(TIPO_INTERES_ORIGINAL, best32.) - 0.01528) / 0.23438;
    _f[45] = (input(MES_CICLO, best32.) - 1) / 83;
    _f[46] = (input(PD_ESTIMADA, best32.) - 0.0003) / 0.9997;
    _f[47] = (input(LGD_ESTIMADA, best32.) - 0.05376373) / 0.94623627;
    _f[48] = (input(LGD_REALIZADA, best32.) - 0.05359046) / 0.94640954;
    _f[49] = (input(EAD, best32.) - 2806.16) / 8389751.4;
    _f[50] = (input(ECL, best32.) - 4.27) / 6121371;
    _f[51] = (input(PROVISION, best32.) - 4.27) / 6121371;
    _f[52] = (input(STAGE_IFRS9, best32.) - 1) / 2;
    _f[53] = (input(CURE_FLAG, best32.) - 0) / 1;
    _f[54] = (input(DPDS, best32.) - 30) / 2490;
    _f[55] = (input(RATING_GRADO, best32.) - 1) / 9;
    _f[56] = (input(SALDO_PENDIENTE, best32.) - 121.88) / 9974219.9;
    _f[57] = (input(TIPO_INTERES_ACTUAL, best32.) - 0.01517) / 0.23449;
    _f[58] = (input(INTERESES_ACUMULADOS, best32.) - 13.4) / 11085068;
    _f[59] = (input(FLUJO_RECUPERADO, best32.) - 0) / 2626696.1;
    _f[60] = (input(COSTE_RECUPERACION, best32.) - 0) / 123832.69;
    _f[61] = (input(FLUJO_NETO, best32.) - 0) / 2509986.2;
    _f[62] = (input(RECUPERACION_ACUMULADA, best32.) - 0) / 8241710.6;
    _f[63] = (input(COSTE_TOTAL_ACUMULADO, best32.) - 0) / 257770.19;
    _f[64] = (input(ADJUDICACION_FLAG, best32.) - 0) / 1;
    _f[65] = (input(ADJUDICACION_VALOR, best32.) - 0) / 2860157;
    _f[66] = (input(VALOR_COLATERAL, best32.) - 0) / 7305505.9;
    _f[67] = (input(LTV, best32.) - 0) / 3.188509;
    _f[68] = (input(TASA_DESCUENTO, best32.) - 0.015137) / 0.039708;
    _f[69] = (input(RWA, best32.) - 181.48) / 779649.15;

    array _out[69] F1-F69;
    do _i = 1 to 69; _out[_i] = _f[_i]; end;

    _ROWID_ = _N_;

    if mod(_N_, 500) = 0 then
        put 'NOTE: [TabBERT-SAS] Row ' _N_ ' built at ' datetime20.;
    keep F1-F69 _ROWID_;
run;

data _meta;
    set cycles;
    _ROWID_ = _N_;
    keep _ROWID_ _all_;
run;

proc iml;
    use proj_raw;
    read all var _num_ into W;
    close proj_raw;

    use _features;
    read all var _num_ into X_all;
    close _features;

    n = nrow(X_all);
    f_dim = ncol(X_all) - 1;
    X = X_all[, 1:f_dim];
    row_ids = X_all[, ncol(X_all)];
    e_dim = ncol(W);

    print "TabBERT-SAS: " n "cases, " f_dim "features -> " e_dim "embeddings";
    print '---';

    n_batches = ceil(n / 5000);
    do b = 1 to n_batches;
        start = (b-1) * 5000 + 1;
        end_  = min(b * 5000, n);
        X_batch  = X[start:end_, ];
        ids_batch = row_ids[start:end_, ];
        E_batch = X_batch * W;
        out_batch = ids_batch || E_batch;
        if b = 1 then E = out_batch;
        else E = E // out_batch;
        pct = round(end_ / n * 100, 1);
        print "Batch " b "of " n_batches ": " end_ "of " n " (" pct "%)";
    end;
    print '---';

    indices = 1:e_dim;
    emb_names = 'EMB_' + strip(char(indices));
    col_names = 'ROW_ID' || emb_names;

    create _embeddings from E [colname=col_names];
    append from E;
    close _embeddings;
    print 'TabBERT-SAS: Done';
quit;

proc sql;
    create table sas_embeddings_full as
        select e.*, m.*
        from _embeddings e
        left join _meta m on e.ROW_ID = m._ROWID_
        order by e.ROW_ID;
    drop table _features, _meta, _embeddings;
quit;

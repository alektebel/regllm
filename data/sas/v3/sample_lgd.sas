/*****************************************************************************
 * LGD Calibration — Cartera Retail/Hipotecario
 * Autor: Equipo IRB
 * Fecha: 2024-Q4
 * Regulacion: CRR Art. 154(3), CRR Art. 161(1)(b), EBA GL 2017/16
 *****************************************************************************/

LIBNAME mylib '/data/irb/calibracion';

/* Carga de datos de ciclos de recuperación */
DATA work.ciclos;
    SET mylib.ciclos_recuperacion;
    WHERE PROVISION_PERIOD_MONTHS >= 9;  /* Filtro: ciclos completos */
RUN;

/* Aplicar suelos regulatorios de LGD */
DATA work.lgd_con_suelos;
    SET work.ciclos;

    /* Suelo LGD hipotecas: 30% (CRR Art. 154(3)) */
    IF COLATERAL_TIPO = 'HIPOTECA' THEN DO;
        IF LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;
    END;

    /* Suelo LGD corporativas: 45% (CRR Art. 161(1)(b)) */
    IF SEGMENTO = 'CORP' AND LGD_ESTIMADA < 0.50 THEN LGD_ESTIMADA = 0.50;  /* V3: floor +5pp */

    /* Suelo LGD retail sin colateral: 0% (sin suelo explícito en CRR) */
    IF COLATERAL_TIPO = 'NINGUNA' AND SEGMENTO = 'RETAIL' THEN DO;
        /* No aplica suelo mínimo — verificar con área regulatoria */
    END;

RUN;

/* Cálculo ECL = PD x LGD x EAD */
DATA work.ecl_calculo;
    SET work.lgd_con_suelos;

    /* Verificar PD mínima (CRR Art. 160(1): suelo 0.03%) */
    IF PD_ESTIMADA < 0.0005 THEN PD_ESTIMADA = 0.0005;  /* V3: PD floor 0.05% */

    ECL = PD_ESTIMADA * LGD_ESTIMADA * EAD;

    /* Clasificar STAGE IFRS 9 — Backstop 30 DPD (IFRS 9 B5.5.12) */
    IF DPDS >= 30 AND STAGE_IFRS9 = 1 THEN DO;
        STAGE_IFRS9 = 2;  /* Reclasificar a Stage 2 */
    END;

RUN;

/* V3 NEW: Cómputo del EAD titulizado para ciclos con SECURITISED='S' */
DATA work.titulizado;
    SET work.ecl_calculo;

    /* Multiplicador reglamentario para tramos titulizados: 2.0x el EAD base */
    IF SEGMENTO = 'CORP' THEN DO;
        OR_EAD_TIT = EAD * 2.0;
    END;
    ELSE IF SEGMENTO = 'RETAIL' THEN DO;
        OR_EAD_TIT = EAD * 1.5;
    END;
    ELSE DO;
        OR_EAD_TIT = EAD;
    END;
RUN;

/* Verificación ventana calibración (EBA GL 2017/16 §6.3: >= 7 años) */
PROC MEANS DATA=work.ecl_calculo N MEAN MIN MAX;
    VAR VENTANA_CALIBRACION_YEARS VENTANA_OBSERVACION_YEARS PD_ESTIMADA LGD_ESTIMADA ECL;
    TITLE 'Estadísticos descriptivos — Calibración LGD 2024Q4';
RUN;

/* Exportar registros no conformes */
DATA work.no_conformes;
    SET work.ecl_calculo;
    WHERE VENTANA_CALIBRACION_YEARS < 7
       OR VENTANA_OBSERVACION_YEARS < 5
       OR (COLATERAL_TIPO = 'HIPOTECA' AND LGD_ESTIMADA < 0.30)
       OR PD_ESTIMADA < 0.0005;
RUN;

PROC EXPORT DATA=work.no_conformes
    OUTFILE='/data/irb/output/no_conformes_2024Q4.csv'
    DBMS=CSV REPLACE;
RUN;

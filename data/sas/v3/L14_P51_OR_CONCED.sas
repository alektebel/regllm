/*IDEALMENTE SE SUSTITUYE EN LA K*/


*Vamos a comparar el concedido que hemos usado hasta ahora (OR_CONCED_01 = OR_CONCED)
con una suma de campos de Basilea (OR_CONCED_02 = OR_DISPTO + OR_DISBLE)
y con OR_CONCED_03 = ME_CONCEDIDO_EU de la tabla &LIBREF_IT_CAL_PRO.BS_CAL_FR_EAD_CONCEDID, proporcionada por el equipo de EAD


-	Nombre del campo --> ¿?
ME_CONCEDIDO_EU
-	Nombre de la tabla --> &LIBREF_IT_CAL_PRO.BS_CAL_FR_EAD_CONCEDID
-	¿A qué nivel está la información? (Contrato, titularidad, mes…) --> (Contrato, mes)
-	Para llegar a tener esta información en la LGD_PRE, ¿sería viable incorporar este campo en la proxy mensual o en el catálogo? --> No sería necesario. Se propone que la LGD_PRE se alimente directamente de &LIBREF_IT_CAL_PRO.BS_CAL_FR_EAD_CONCEDID
-	¿Este campo sólo abarca Tarjetas y Créditos, o cubre todos los productos? --> Abarca Empresas y Particulares, pero ni Tarjetas ni Créditos
Están todos los productos, lo puedes confirmar con el campo ID_TIPO_PRODUCTO
-	¿Existe una tabla como &LIBREF_IT_CAL_PRO.BS_CAL_FR_EAD_CONCEDID, pero para Tarjetas y Créditos?
-	¿Qué diferencias tenéis identificadas entre este campo y el OR_CONCED de Basilea? --> Se contempla una comparación contra la suma (OR_DISPTO + OR_DISBLE) de Basilea. Si salen discrepancias, entonces se utiliza como concedido la suma (OR_DISPTO + OR_DISBLE) de Basilea. --> Entonces según esto, lo que realmente hacéis es usar directamente la suma (OR_DISPTO + OR_DISBLE) de Basilea?
El dispuesto+disponible a nivel mensual obligaría a usar Basilea_Mensual_Ini que es muy pesada, los cruces con la tabla de concedidos son mas rápidos, para las conciliaciones recomiendo usar las tablas trimestrales de Basilea fusionada.
-	El motivo por el que usáis este campo de &LIBREF_IT_CAL_PRO.BS_CAL_FR_EAD_CONCEDID  y no el OR_CONCED de Basilea, ¿es porque es más fiable? --> ¿?
No se exactamente cuál es la deficiencia que tiene el campo de concedido de Basilea, pero en los modelos de CCF se viene usando la tabla de concedidos indicada desde hace bastantes años. La mayoría de los casos coinciden con el dispuesto+disponible de Basilea.

;

PROC SQL;
   CREATE TABLE WORK.QUERY_FOR_BS_CAL_CT_CONTRAT_0000 AS 
   SELECT t1.ID_CONTRATO, 
          t1.ID_FCH_DATOS, 
          /* COUNT_of_ID_FCH_DATOS */
            (COUNT(t1.ID_FCH_DATOS)) AS COUNT_of_ID_FCH_DATOS, 
          /* MIN_of_ME_IMP_FORMALIZACION */
            (MIN(t1.ME_IMP_FORMALIZACION)) FORMAT=17.2 AS MIN_of_ME_IMP_FORMALIZACION, 
          /* MAX_of_ME_IMP_FORMALIZACION */
            (MAX(t1.ME_IMP_FORMALIZACION)) FORMAT=17.2 AS MAX_of_ME_IMP_FORMALIZACION
      FROM ORADMR.BS_CAL_CT_CONTRATOS t1
      GROUP BY t1.ID_CONTRATO,
               t1.ID_FCH_DATOS;
QUIT;

%put &SUFIJO_L_IN.;
proc sql;
create table &LIBWORK.L14_P51_VB_P02_01 as
select
	ID_CONTR_CICLO_LGD
	, FC_FECINICIC_LGD
	, FC_FECFINCIC_LGD
	, ID_FUSION_FINAL
	, SEGM_CAT_NDOD_PRI
	, ID_PRODUCTO_FIN AS ID_PRODUCTO_PRI
	, SEGM_CAT_NDOD
	, ID_PRODUCTO
	, ID_CONTRATO
	, ID_TITULARITAT
	, FEC_OBS
	, OR_CONCED as OR_CONCED_01
	, OR_DISPTO
	, OR_DISBLE
	, OR_DISPTO + OR_DISBLE as OR_CONCED_02


from &LIBWORK.L14_P48_FIN as t1
order by
	ID_CONTR_CICLO_LGD
	, FEC_OBS
; quit;

/*proc sql;*/
/*create table &LIBWORK.VB_P02_02 as*/
/*select*/
/*	ID_CONTR_CICLO_LGD*/
/*	, OR_CONCED_01 as OR_CONCED_PRI_01*/
/*	, OR_CONCED_02 as OR_CONCED_PRI_02*/
/*from &LIBWORK.VB_P02_01*/
/*where FEC_OBS = FC_FECINICIC_LGD*/
/*; quit;*/

*desfusionamos antes de cruzar con la tabla de EAD, ya que se cruza por contrato
excluir 0s a ver si sigue habiendo duplicados (anticipando respuesta Manuel, reclamar)
si sigue habiendo, agregar suma
;

data &LIBWORK.L14_P51_VB_P02_03_AUX_0_SFUS &LIBWORK.L14_P51_VB_P02_03_AUX_0_CFUS;
set  &LIBWORK.L14_P51_VB_P02_01 (drop= OR_CONCED_01 OR_CONCED_02 OR_DISPTO OR_DISBLE);
if ID_FUSION_FINAL = . then do;
	output &LIBWORK.L14_P51_VB_P02_03_AUX_0_SFUS;
end;
else do;
	output &LIBWORK.L14_P51_VB_P02_03_AUX_0_CFUS;
end;
run;
%put &T_D_CFUS.;
proc sql;
create table &LIBWORK.L14_P51_VB_P02_03_AUX_1_CFUS as
select
	t1.ID_CONTR_CICLO_LGD
	, t1.FC_FECINICIC_LGD
	, t1.FC_FECFINCIC_LGD
	, t1.ID_FUSION_FINAL
	, t1.SEGM_CAT_NDOD_PRI
	, t1.ID_PRODUCTO_PRI
	, t1.SEGM_CAT_NDOD
	, t1.ID_PRODUCTO
	, t1.FEC_OBS
	, t2.ID_CONTRATO
	, t2.ID_TITULARITAT
from &LIBWORK.L14_P51_VB_P02_03_AUX_0_CFUS as t1
inner join &T_D_CFUS. as t2
on  t1.ID_FUSION_FINAL = t2.ID_FUSION_FINAL
and t1.FEC_OBS = t2.FEC_OBS
; quit;

/*proc sql;*/
/*create table &LIBWORK.VB_P02_03_AUX_1 as*/
/*select * from &LIBWORK.VB_P02_03_AUX_0_SFUS union all*/
/*select * from &LIBWORK.VB_P02_03_AUX_1_CFUS*/
/*; quit;*/

data &LIBWORK.L14_P51_VB_P02_03_AUX_1;
set  &LIBWORK.L14_P51_VB_P02_03_AUX_0_SFUS &LIBWORK.L14_P51_VB_P02_03_AUX_1_CFUS;
run;

proc sql;
create table &LIBWORK.L14_P51_VB_P02_03_A as
select
	t1.*
	, ME_CONCEDIDO_EU as OR_CONCED_03
from &LIBWORK.L14_P51_VB_P02_03_AUX_1 as t1
left join &LIBREF_IT_CAL_PRO.BS_CAL_FR_EAD_CONCEDID  as t2  on  t1.ID_CONTRATO = t2.ID_CONTRATO and t1.FEC_OBS = t2.ID_FCH_DATOS
; quit;

proc sql;
create table &LIBWORK.L14_P51_VB_P02_03_A_ANA_01 as
select
	ID_CONTR_CICLO_LGD, FEC_OBS
	, count(*) as N_TOTAL
	, sum(case when round(OR_CONCED_03, 0.01) = 0 then 1 else 0 end) as N_CERO
	, sum(case when OR_CONCED_03 =  . then 1 else 0 end) as N_MISS
	, sum(case when mod(OR_CONCED_03, 1000) = 999 then 1 else 0 end) as N_999
	, sum(case when OR_CONCED_03 ne . and OR_CONCED_03 < 0 then 1 else 0 end) as N_NEG
	, min(OR_CONCED_03) as MIN_OR_CONCED_03
	, max(OR_CONCED_03) as MAX_OR_CONCED_03
	, sum(OR_CONCED_03) as SUM_OR_CONCED_03
from &LIBWORK.L14_P51_VB_P02_03_A
group by
	ID_CONTR_CICLO_LGD, FEC_OBS
; quit;

proc sql;
create table &LIBWORK.L14_P51_VB_P02_03_B as
select
	t1.*
	, case 
		when t3.CONTRATO ne . then t3.IMPORTE_CONCEDIDO_PR 
		when t4.CONTRATO ne . then t4.IMPORTE_CONCEDIDO_PR 
	end as OR_CONCED_04
	, MAX_of_ME_IMP_FORMALIZACION as OR_CONCED_05
	, case
		when t4.CONTRATO ne . and mod(t4.IMPORTE_CONCEDIDO_PR - t4.SALDO_AB_LIQ,10) 		= 0 then t4.IMPORTE_CONCEDIDO_PR - t4.SALDO_AB_LIQ
		when t4.CONTRATO ne . and mod(t4.IMPORTE_CONCEDIDO_PR + t4.importe_impago_pr,10) 	= 0 then t4.IMPORTE_CONCEDIDO_PR + t4.importe_impago_pr
		when t4.CONTRATO ne . and mod(t4.IMPORTE_CONCEDIDO_PR + t4.aplazado_posicion,10) 	= 0 then t4.IMPORTE_CONCEDIDO_PR + t4.aplazado_posicion
	end as OR_CONCED_06
from &LIBWORK.L14_P51_VB_P02_03_AUX_1 as t1
left join ORASCRCO.TB_INFCONTRATO_POLIZAS  as t3  on  t1.ID_CONTRATO =    t3.CONTRATO and t1.FEC_OBS = t3.MES_PROCESO
left join ORASCRCO.TB_INFCONTRATO_TARJETAS  as t4 on  t1.ID_CONTRATO =    t4.CONTRATO and t1.FEC_OBS = t4.MES_PROCESO
left join QUERY_FOR_BS_CAL_CT_CONTRAT_0000  as t5 on  t1.ID_CONTRATO = t5.ID_CONTRATO and t1.FEC_OBS = t5.ID_FCH_DATOS
; quit;

proc sql;
create table &LIBWORK.L14_P51_VB_P02_03_B_ANA_01 as
select
	ID_CONTR_CICLO_LGD, FEC_OBS
	, count(*) as N_TOTAL
	, min(OR_CONCED_04) as MIN_OR_CONCED_04
	, max(OR_CONCED_04) as MAX_OR_CONCED_04
	, min(OR_CONCED_05) as MIN_OR_CONCED_05
	, max(OR_CONCED_05) as MAX_OR_CONCED_05
	, sum(OR_CONCED_05) as SUM_OR_CONCED_05
	, min(OR_CONCED_06) as MIN_OR_CONCED_06
	, max(OR_CONCED_06) as MAX_OR_CONCED_06
	, sum(OR_CONCED_06) as SUM_OR_CONCED_06
from &LIBWORK.L14_P51_VB_P02_03_B
group by
	ID_CONTR_CICLO_LGD, FEC_OBS
; quit;

proc sql;
create table &LIBWORK.L14_P51_VB_P02_04 as
select
	t1.*
	, coalesce(SUM_OR_CONCED_03, 0) as OR_CONCED_03
	, coalesce(MAX_OR_CONCED_04, 0) as OR_CONCED_04
	, coalesce(SUM_OR_CONCED_05, 0) as OR_CONCED_05
	, coalesce(SUM_OR_CONCED_06, 0) as OR_CONCED_06
from &LIBWORK.L14_P51_VB_P02_01 as t1
left join &LIBWORK.L14_P51_VB_P02_03_A_ANA_01 as t2
on  t1.ID_CONTR_CICLO_LGD = t2.ID_CONTR_CICLO_LGD
and t1.FEC_OBS = t2.FEC_OBS
left join &LIBWORK.L14_P51_VB_P02_03_B_ANA_01 as t3
on  t1.ID_CONTR_CICLO_LGD = t3.ID_CONTR_CICLO_LGD
and t1.FEC_OBS = t3.FEC_OBS
order by ID_CONTR_CICLO_LGD, FEC_OBS
; quit;

DATA &LIBWORK.L14_P51_VB_P02_05 (DROP = OR_CONCED_01_TEMP 
OR_CONCED_02_TEMP 
OR_CONCED_03_TEMP 
OR_CONCED_04_TEMP
OR_CONCED_05_TEMP 
OR_CONCED_06_TEMP
);
set  &LIBWORK.L14_P51_VB_P02_04;
BY ID_CONTR_CICLO_LGD;
RETAIN OR_CONCED_01_TEMP 
OR_CONCED_02_TEMP 
OR_CONCED_03_TEMP 
OR_CONCED_04_TEMP
OR_CONCED_05_TEMP 
OR_CONCED_06_TEMP
;

IF FIRST.ID_CONTR_CICLO_LGD THEN DO;
	OR_CONCED_01_TEMP = OR_CONCED_01;
	OR_CONCED_01_PRI  = OR_CONCED_01_TEMP;
END;
ELSE IF OR_CONCED_01_TEMP NOT IN (., 0, 999, 9999, 99999, 999999) THEN DO;
	OR_CONCED_01_TEMP = OR_CONCED_01_TEMP;	
	OR_CONCED_01_PRI  = OR_CONCED_01_TEMP;
END;
ELSE IF OR_CONCED_01 NOT IN (., 0, 999, 9999, 99999, 999999) THEN DO;
	OR_CONCED_01_TEMP = OR_CONCED_01;
	OR_CONCED_01_PRI  = OR_CONCED_01_TEMP;
END;

IF FIRST.ID_CONTR_CICLO_LGD THEN DO;
	OR_CONCED_02_TEMP = OR_CONCED_02;
	OR_CONCED_02_PRI  = OR_CONCED_02_TEMP;
END;
ELSE IF OR_CONCED_02_TEMP NOT IN (., 0, 999, 9999, 99999, 999999) THEN DO;
	OR_CONCED_02_TEMP = OR_CONCED_02_TEMP;	
	OR_CONCED_02_PRI  = OR_CONCED_02_TEMP;
END;
ELSE IF OR_CONCED_02 NOT IN (., 0, 999, 9999, 99999, 999999) THEN DO;
	OR_CONCED_02_TEMP = OR_CONCED_02;
	OR_CONCED_02_PRI  = OR_CONCED_02_TEMP;
END;

IF FIRST.ID_CONTR_CICLO_LGD THEN DO;
	OR_CONCED_03_TEMP = OR_CONCED_03;
	OR_CONCED_03_PRI  = OR_CONCED_03_TEMP;
END;
ELSE IF OR_CONCED_03_TEMP NOT IN (., 0, 999, 9999, 99999, 999999) THEN DO;
	OR_CONCED_03_TEMP = OR_CONCED_03_TEMP;	
	OR_CONCED_03_PRI  = OR_CONCED_03_TEMP;
END;
ELSE IF OR_CONCED_03 NOT IN (., 0, 999, 9999, 99999, 999999) THEN DO;
	OR_CONCED_03_TEMP = OR_CONCED_03;
	OR_CONCED_03_PRI  = OR_CONCED_03_TEMP;
END;

IF FIRST.ID_CONTR_CICLO_LGD THEN DO;
	OR_CONCED_04_TEMP = OR_CONCED_04;
	OR_CONCED_04_PRI  = OR_CONCED_04_TEMP;
END;
ELSE IF OR_CONCED_04_TEMP NOT IN (., 0, 999, 9999, 99999, 999999) THEN DO;
	OR_CONCED_04_TEMP = OR_CONCED_04_TEMP;	
	OR_CONCED_04_PRI  = OR_CONCED_04_TEMP;
END;
ELSE IF OR_CONCED_04 NOT IN (., 0, 999, 9999, 99999, 999999) THEN DO;
	OR_CONCED_04_TEMP = OR_CONCED_04;
	OR_CONCED_04_PRI  = OR_CONCED_04_TEMP;
END;

IF FIRST.ID_CONTR_CICLO_LGD THEN DO;
	OR_CONCED_05_TEMP = OR_CONCED_05;
	OR_CONCED_05_PRI  = OR_CONCED_05_TEMP;
END;
ELSE IF OR_CONCED_05_TEMP NOT IN (., 0, 999, 9999, 99999, 999999) THEN DO;
	OR_CONCED_05_TEMP = OR_CONCED_05_TEMP;	
	OR_CONCED_05_PRI  = OR_CONCED_05_TEMP;
END;
ELSE IF OR_CONCED_05 NOT IN (., 0, 999, 9999, 99999, 999999) THEN DO;
	OR_CONCED_05_TEMP = OR_CONCED_05;
	OR_CONCED_05_PRI  = OR_CONCED_05_TEMP;
END;

IF FIRST.ID_CONTR_CICLO_LGD THEN DO;
	OR_CONCED_06_TEMP = OR_CONCED_06;
	OR_CONCED_06_PRI  = OR_CONCED_06_TEMP;
END;
ELSE IF OR_CONCED_06_TEMP NOT IN (., 0, 999, 9999, 99999, 999999) THEN DO;
	OR_CONCED_06_TEMP = OR_CONCED_06_TEMP;	
	OR_CONCED_06_PRI  = OR_CONCED_06_TEMP;
END;
ELSE IF OR_CONCED_06 NOT IN (., 0, 999, 9999, 99999, 999999) THEN DO;
	OR_CONCED_06_TEMP = OR_CONCED_06;
	OR_CONCED_06_PRI  = OR_CONCED_06_TEMP;
END;

run;

PROC SQL;
CREATE TABLE &LIBWORK.L14_P51_VB_P02_06 AS SELECT
	T1.*
FROM &LIBWORK.L14_P51_VB_P02_05 AS T1
ORDER BY T1.ID_CONTR_CICLO_LGD, T1.FEC_OBS DESC;
QUIT;

DATA &LIBWORK.L14_P51_VB_P02_07 (DROP = OR_CONCED_01_TEMP 
OR_CONCED_02_TEMP 
OR_CONCED_03_TEMP 
OR_CONCED_04_TEMP
OR_CONCED_05_TEMP 
OR_CONCED_06_TEMP
);
set  &LIBWORK.L14_P51_VB_P02_06;
BY ID_CONTR_CICLO_LGD;
RETAIN OR_CONCED_01_TEMP 
OR_CONCED_02_TEMP 
OR_CONCED_03_TEMP 
OR_CONCED_04_TEMP
OR_CONCED_05_TEMP 
OR_CONCED_06_TEMP
;

IF FIRST.ID_CONTR_CICLO_LGD THEN DO;
	OR_CONCED_01_TEMP = OR_CONCED_01_PRI;
END;
ELSE DO;	
	OR_CONCED_01_PRI  = OR_CONCED_01_TEMP;
END;

IF FIRST.ID_CONTR_CICLO_LGD THEN DO;
	OR_CONCED_02_TEMP = OR_CONCED_02_PRI;
END;
ELSE DO;	
	OR_CONCED_02_PRI  = OR_CONCED_02_TEMP;
END;

IF FIRST.ID_CONTR_CICLO_LGD THEN DO;
	OR_CONCED_03_TEMP = OR_CONCED_03_PRI;
END;
ELSE DO;	
	OR_CONCED_03_PRI  = OR_CONCED_03_TEMP;
END;

IF FIRST.ID_CONTR_CICLO_LGD THEN DO;
	OR_CONCED_04_TEMP = OR_CONCED_04_PRI;
END;
ELSE DO;	
	OR_CONCED_04_PRI  = OR_CONCED_04_TEMP;
END;

IF FIRST.ID_CONTR_CICLO_LGD THEN DO;
	OR_CONCED_05_TEMP = OR_CONCED_05_PRI;
END;
ELSE DO;	
	OR_CONCED_05_PRI  = OR_CONCED_05_TEMP;
END;

IF FIRST.ID_CONTR_CICLO_LGD THEN DO;
	OR_CONCED_06_TEMP = OR_CONCED_06_PRI;
END;
ELSE DO;	
	OR_CONCED_06_PRI  = OR_CONCED_06_TEMP;
END;

run;

PROC SQL;
CREATE TABLE &LIBWORK.L14_P51_VB_P02_08 AS 
SELECT DISTINCT
	T1.ID_CONTR_CICLO_LGD, T1.SEGM_CAT_NDOD_PRI
	, T1.ID_PRODUCTO_PRI
	, OR_CONCED_01_PRI
	, OR_CONCED_02_PRI
	, OR_CONCED_03_PRI
	, OR_CONCED_04_PRI
	, OR_CONCED_05_PRI
	, OR_CONCED_06_PRI
/*	, case when OR_CONCED_01_PRI ne OR_CONCED_02_PRI then 1 else 0 end as SW_VARIA_01_02*/
/*	, case when OR_CONCED_01_PRI ne OR_CONCED_03_PRI then 1 else 0 end as SW_VARIA_01_03*/
/*	, case when OR_CONCED_01_PRI ne OR_CONCED_04_PRI then 1 else 0 end as SW_VARIA_01_04*/
/*	, case when OR_CONCED_02_PRI ne OR_CONCED_03_PRI then 1 else 0 end as SW_VARIA_02_03*/
/*	, case when OR_CONCED_02_PRI ne OR_CONCED_04_PRI then 1 else 0 end as SW_VARIA_02_04*/
/*	, case when OR_CONCED_03_PRI ne OR_CONCED_04_PRI then 1 else 0 end as SW_VARIA_03_04*/
/*	, case*/
/*		when OR_CONCED_01_PRI = OR_CONCED_02_PRI and OR_CONCED_01_PRI = OR_CONCED_03_PRI then '00. Sin cambios'*/
/*		when OR_CONCED_02_PRI not in (., 0) then (case*/
/*			when OR_CONCED_01_PRI     in (., 0) and OR_CONCED_03_PRI     in (., 0) and OR_CONCED_04_PRI     in (., 0) then '01. DISPTO+DISBLE info; En el resto, todo missing'*/
/*			when OR_CONCED_01_PRI not in (., 0) and OR_CONCED_03_PRI     in (., 0) and OR_CONCED_04_PRI     in (., 0) then '02. DISPTO+DISBLE info; En el resto, sólo informado el OR_CONCED'*/
/*			when OR_CONCED_01_PRI     in (., 0) and OR_CONCED_03_PRI not in (., 0) and OR_CONCED_04_PRI     in (., 0) and OR_CONCED_02_PRI < OR_CONCED_03_PRI then '03A. DISPTO+DISBLE info; En el resto, sólo informado BS_CAL_FR...; DISPTO+DISBLE < tabla BS_CAL_FR'*/
/*			when OR_CONCED_01_PRI     in (., 0) and OR_CONCED_03_PRI not in (., 0) and OR_CONCED_04_PRI     in (., 0) and OR_CONCED_02_PRI = OR_CONCED_03_PRI then '03B. DISPTO+DISBLE info; En el resto, sólo informado BS_CAL_FR...; DISPTO+DISBLE = tabla BS_CAL_FR'*/
/*			when OR_CONCED_01_PRI     in (., 0) and OR_CONCED_03_PRI not in (., 0) and OR_CONCED_04_PRI     in (., 0) and OR_CONCED_02_PRI > OR_CONCED_03_PRI then '03C. DISPTO+DISBLE info; En el resto, sólo informado BS_CAL_FR...; DISPTO+DISBLE > tabla BS_CAL_FR'*/
/*			when OR_CONCED_01_PRI     in (., 0) and OR_CONCED_03_PRI     in (., 0) and OR_CONCED_04_PRI not in (., 0) and OR_CONCED_02_PRI < OR_CONCED_04_PRI then '04A. DISPTO+DISBLE info; En el resto, sólo informado   Scoring...; DISPTO+DISBLE < tabla   Scoring'*/
/*			when OR_CONCED_01_PRI     in (., 0) and OR_CONCED_03_PRI     in (., 0) and OR_CONCED_04_PRI not in (., 0) and OR_CONCED_02_PRI = OR_CONCED_04_PRI then '04B. DISPTO+DISBLE info; En el resto, sólo informado   Scoring...; DISPTO+DISBLE = tabla   Scoring'*/
/*			when OR_CONCED_01_PRI     in (., 0) and OR_CONCED_03_PRI     in (., 0) and OR_CONCED_04_PRI not in (., 0) and OR_CONCED_02_PRI > OR_CONCED_04_PRI then '04C. DISPTO+DISBLE info; En el resto, sólo informado   Scoring...; DISPTO+DISBLE > tabla   Scoring'*/
/*			when OR_CONCED_01_PRI     in (., 0) and OR_CONCED_03_PRI not in (., 0) and OR_CONCED_04_PRI not in (., 0) then '05. DISPTO+DISBLE info; En el resto, informados BS_CAL_FR y Scoring'*/
/*			else '98. Otros' end)*/
/*		else '99. Otros'*/
/*	end as CASUISTICA*/
	, case
		when ID_PRODUCTO_PRI in ('PR') and OR_CONCED_01_PRI not in (., 0, 999, 9999, 99999, 999999) then OR_CONCED_01_PRI
		when OR_CONCED_05_PRI not in (., 0, 999, 9999, 99999, 999999) then OR_CONCED_05_PRI
		when OR_CONCED_06_PRI not in (., 0, 999, 9999, 99999, 999999) then OR_CONCED_06_PRI
		when OR_CONCED_04_PRI not in (., 0, 999, 9999, 99999, 999999) then OR_CONCED_04_PRI
		when OR_CONCED_03_PRI not in (., 0, 999, 9999, 99999, 999999) then OR_CONCED_03_PRI
		when OR_CONCED_01_PRI not in (., 0, 999, 9999, 99999, 999999) then OR_CONCED_01_PRI
		when OR_CONCED_02_PRI not in (., 0, 999, 9999, 99999, 999999) then OR_CONCED_02_PRI
		else 0
	end as CAMP_RECUPERAT
	, case
		when ID_PRODUCTO_PRI in ('PR') and OR_CONCED_01_PRI not in (., 0, 999, 9999, 99999, 999999) then '01_PR'
		when OR_CONCED_05_PRI not in (., 0, 999, 9999, 99999, 999999) then '05'
		when OR_CONCED_06_PRI not in (., 0, 999, 9999, 99999, 999999) then '06'
		when OR_CONCED_04_PRI not in (., 0, 999, 9999, 99999, 999999) then '04'
		when OR_CONCED_03_PRI not in (., 0, 999, 9999, 99999, 999999) then '03'
		when OR_CONCED_01_PRI not in (., 0, 999, 9999, 99999, 999999) then '01'
		when OR_CONCED_02_PRI not in (., 0, 999, 9999, 99999, 999999) then '02'
		else '00'
	end as DESCR_CAMP_RECUPERAT
	, (OR_CONCED_02_PRI - (calculated CAMP_RECUPERAT))/(calculated CAMP_RECUPERAT) as INCR
	, MAX(CASE WHEN FC_FECINICIC_LGD = FEC_OBS THEN OR_DISPTO end) AS OR_DISPTO_INI
FROM &LIBWORK.L14_P51_VB_P02_07 AS T1
GROUP BY 
	ID_CONTR_CICLO_LGD
;
QUIT;

*selección final: escogemos la prelación;
PROC SQL;
CREATE TABLE &LIBWORK.L14_P51_VB_P02_FIN AS 
SELECT DISTINCT
	ID_CONTR_CICLO_LGD
	, CAMP_RECUPERAT as OR_CONCED_PRI
	, OR_DISPTO_INI/ CAMP_RECUPERAT AS RATIO_AMORT_CTO
	, DESCR_CAMP_RECUPERAT
from &LIBWORK.L14_P51_VB_P02_08
; quit;


/**/
/*proc sql;*/
/*drop table	&LIBWORK.L14_P51_VB_P02_01,*/
/*			&LIBWORK.L14_P51_VB_P02_03_AUX_0_SFUS,*/
/*			&LIBWORK.L14_P51_VB_P02_03_AUX_0_CFUS,*/
/*			&LIBWORK.L14_P51_VB_P02_03_AUX_1_CFUS,*/
/*			&LIBWORK.L14_P51_VB_P02_03_AUX_1,*/
/*			&LIBWORK.L14_P51_VB_P02_03,*/
/*			&LIBWORK.L14_P51_VB_P02_03_ANA_01,*/
/*			&LIBWORK.L14_P51_VB_P02_04,*/
/*			&LIBWORK.L14_P51_VB_P02_05,*/
/*			&LIBWORK.L14_P51_VB_P02_06,*/
/*			&LIBWORK.L14_P51_VB_P02_07,*/
/*			&LIBWORK.L14_P51_VB_P02_08*/
/*; quit;*/
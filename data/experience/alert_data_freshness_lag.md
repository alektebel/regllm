---
id: "alert_data_freshness_lag"
type: "insight"
priority: 0.6
tags: [alert, data, freshness, SAP, SAS, lag]
fields: [EAD, FECHA_DEV]
articles: []
source: "Alerta Data Freshness — extracto SAP desactualizado (2025-05-20)"
feedback: false
---

# [ALERTA] Extracto SAP desactualizado: +5 días sin refresco

Fuente: Data Freshness Monitor
Dataset: SAP.EXTRACT_LGD
Última actualización: 2025-05-15 03:00
Alerta: "Data freshness threshold exceeded: 5 days without refresh (threshold: 2 days)"

## Causa
El job de extracción SAP (ABAP program ZRISK_LGD_EXTRACT) falló silenciosamente el 2025-05-16. El error se registró en el log SAP pero no generó alerta al equipo de operaciones. SAS consumió datos del 15-May durante 5 días.

## Impacto
- Ejecuciones SAS entre 16-May y 20-May usaron datos de EAD con 1-5 días de desfase
- Para carteras estables el impacto es menor (~0.1% variación EAD)
- Pero si hubiera una operación corporativa grande en esos días, el impacto sería significativo

## Fix
1. Añadir webhook de error desde SAP ABAP al Data Freshness Monitor
2. En SAS, añadir check de frescura al inicio:
   ```sas
   %LET MAX_AGE_DAYS = 2;
   PROC SQL;
       SELECT MAX(FECHA_DEV) INTO :LAST_EXTRACT FROM mylib.ciclos_recuperacion;
       %IF %SYSEVALF(%SYSFUNC(INTCK(DAY, &LAST_EXTRACT, %SYSFUNC(TODAY())))) > &MAX_AGE_DAYS %THEN %DO;
           %PUT ERROR: Datos desactualizados (&LAST_EXTRACT). Pipeline detenido.;
           ABORT CANCEL;
       %END;
   QUIT;
   ```

## Lección
Los datos desactualizados son más peligrosos que los datos incorrectos: los incorrectos se detectan, los desactualizados no.

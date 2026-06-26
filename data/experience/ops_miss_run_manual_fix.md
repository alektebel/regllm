---
id: "ops_miss_run_manual_fix"
type: "insight"
priority: 0.6
tags: [operational, run, manual, fix, pipeline]
fields: []
articles: []
source: "Incidente PROD-2025-031 — ejecución manual fuera de secuencia"
feedback: false
---

# Ejecución manual de proj_02 fuera de secuencia causa datos inconsistentes

## Incidente
El 2025-03-15, el equipo de operaciones ejecutó manualmente `proj_02_enriquecimiento_ead.sas` para corregir un error de EAD sin volver a ejecutar `proj_01_carga_datos.sas` primero. El resultado fue que `proj_03_suelos_lgd.sas` encontró datos inconsistentes:
- Ciclos con EAD actualizada pero LGD_ESTIMADA de la ejecución anterior
- Discrepancias en tablas intermedias que asumían consistencia transaccional

## Causa raíz
El pipeline no tiene un mecanismo de versionado o transaccionalidad entre pasos. Cada script SAS asume que los datos de entrada están actualizados.

## Impacto
El reporting mensual se retrasó 3 días mientras se identificaban y corregían las discrepancias. Se ejecutó una rerun completa del pipeline.

## Lecciones
1. El pipeline SAS no es transaccional — las ejecuciones parciales corrompen datos
2. Si se ejecuta un paso manualmente, deben ejecutarse TODOS los pasos posteriores
3. Implementar un check de versión al inicio de cada paso:
   ```sas
   /* Verificar que el paso anterior se ejecutó */
   PROC SQL;
       SELECT MAX(FECHA_EJECUCION) INTO :LAST_RUN FROM mylib.pipeline_log
       WHERE PASO = 'proj_02' AND ESTADO = 'OK';
       SELECT MAX(FECHA_EJECUCION) INTO :CURRENT_RUN FROM mylib.pipeline_log
       WHERE PASO = 'proj_03';
       %IF &LAST_RUN > &CURRENT_RUN %THEN %DO;
           %PUT ERROR: proj_03 requiere re-ejecución tras proj_02 del &LAST_RUN;
       %END;
   QUIT;
   ```

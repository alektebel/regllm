---
id: "ops_data_freshness_alert"
type: "insight"
priority: 0.5
tags: [ops, data, freshness, ECL, reporting, alert]
fields: [ECL, FECHA_CIERRE, CICLO_MES]
articles: []
source: "Incidente datos desactualizados — reporting Q4-2025"
feedback:
  type: finding
  original: "Los datos de cierre siempre están actualizados"
  corrected: "Los datos de cierre tienen un lag de 2-3 días laborables. Si el reporting se ejecuta antes del D+3, usa datos parciales."
---

# Data freshness insuficiente en reporting ECL mensual

El proceso de reporting ECL mensual se ejecuta en D+1 laborable,
pero los datos de cierre contable (FECHA_CIERRE) no están completos
hasta D+3.

## Síntoma

| Mes | Fecha ejecución | Cobertura datos | ¿Completo? |
|---|---|---|---|
| Ene-25 | 02-Feb | 87% | NO |
| Feb-25 | 03-Mar | 92% | NO |
| Mar-25 | 01-Abr | 95% | NO |
| Abr-25 | 02-May | 100% | SÍ (ejecutado D+4 por festivo) |

## Impacto

ECL reportado con datos parciales. La diferencia entre ECL con datos
parciales y ECL completo es ~0.3-0.8% sistemáticamente.

## Causa raíz

El scheduler está configurado para D+1 porque "los datos de cierre
están disponibles". No hay verificación de completitud antes de ejecutar.

## Fix

```sas
/* Verificar completitud antes de ejecutar */
PROPRC SQL;
    SELECT COUNT(*) AS total, COUNT(FECHA_CIERRE) AS con_fecha
    INTO :total, :completos
    FROM ciclos_recuperacion
    WHERE CICLO_MES = "&MES";
QUIT;
%IF %SYSEVALF(&completos / &total) < 0.98 %THEN %DO;
    %PUT ERROR: Datos incompletos (&completos/&total). Abortando.;
    %ABORT;
%END;
```

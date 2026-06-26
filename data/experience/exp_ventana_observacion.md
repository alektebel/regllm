---
id: exp_ventana_observacion
type: insight
priority: 0.6
tags: [PD, ventana, observacion, calibracion]
fields: [VENTANA_OBSERVACION_YEARS, VENTANA_CALIBRACION_YEARS, PD_ESTIMADA]
articles: [eba_gl_2017_16]
source: "Análisis calibración PD"
feedback: false
---

# VENTANA_OBSERVACION_YEARS mínimos regulatorios EBA GL 2017/16

## Mínimos por segmento

| Segmento | Observación (años) | Calibración (años) |
|---|---|---|
| CORP | ≥5 | ≥7 |
| HIPOTECA | ≥7 | ≥7 |
| RETAIL | ≥5 | ≥7 |

## Pipeline SAS

En `lgd_macros.sas:60-62`:
```sas
min_calib_years = 7,
min_obs_years   = 5,
```

Non-conformes check en `proj_03_suelos_lgd.sas:78-81`:
```sas
IF VENTANA_OBSERVACION_YEARS < 5 THEN DO;
    FLAG_NC = 1;
    MOTIVO = CATX(' | ', MOTIVO, 'Ventana observación < 5 años');
END;
```

Chequea 5 años para todos los segmentos, pero hipotecas requieren 7.

## Relevancia regulatoria

- Art.8: cambios de fase NO son efectivos hasta cerrar ventana observación
- Art.23: requiere 2 ventanas de observación consecutivas en Stage≤2 para reclasificar Stage 2→1
- `VENTANA_CALIBRACION_YEARS` (mín 7) es distinta: para re-fitting modelos de rating

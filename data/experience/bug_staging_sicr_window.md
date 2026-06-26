---
id: "bug_staging_sicr_window"
type: "insight"
priority: 0.7
tags: [bug, staging, SICR, IFRS9, DPDS]
fields: [STAGE_IFRS9, DPDS, FASE_CRISIS, PROVISION_PERIOD_MONTHS]
articles: [ifrs9_5_5, circular_6_2016_art_8]
source: "Validación staging IFRS9 — ventana SICR incorrecta"
feedback: false
---

# Ventana SICR de 12 meses fija ignora segmentación EBA

El pipeline asigna `STAGE_IFRS9=2` con ventana SICR fija de 12 meses
para TODOS los segmentos. EBA GL 2017/16 §148 requiere segmentación:

| Segmento | Ventana SICR mínima | Riesgo |
|---|---|---|
| CORP | 12 meses | Adecuado |
| HIPOTECA | 24 meses | Infraestimado — ciclos entran tarde a Stage 2 |
| RETAIL | 6 meses | Sobreestimado — ciclos saltan rápido a Stage 2 |

## Impacto

- Hipotecas: ciclos con deterioro temprano (DPDS 13-24) clasificados como Stage 1
  cuando deberían ser Stage 2 → provision_period mal calculado
- Retail: ciclos estacionales entran a Stage 2 prematuramente → provisioning excesivo

## Causa raíz

`proj_01_staging_ifrs9.sas:22-25`:
```sas
IF DPDS >= 12 THEN STAGE_IFRS9 = 2;  /* SICR fijo */
IF DPDS >= 360 THEN STAGE_IFRS9 = 3;
```

No hay lógica por segmento. La EBA GL requiere SICR evaluation
que considere tipo de exposición.

## Fix

Parametrizar ventana SICR por segmento usando tabla de configuración:
```sas
SELECT ventana_sicr INTO :v_sicr FROM config_staging WHERE segmento = "&SEGMENTO";
```

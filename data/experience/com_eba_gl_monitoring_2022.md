---
id: "com_eba_gl_monitoring_2022"
type: "insight"
priority: 0.6
tags: [EBA, monitoring, GL, 2022, IRB, reporting]
fields: [PD_ESTIMADA, LGD_ESTIMADA, EAD, RWA]
articles: [eba_gl_2022_01]
source: "EBA/GL/2022/01 — Monitoring de modelos IRB"
feedback: false
---

# [EBA GL] EBA/GL/2022/01 — Monitoring de modelos IRB

EBA/GL/2022/01 (publicada enero 2022) actualiza los requisitos de
monitoreo continuo para modelos IRB. Sustituye EBA/GL/2017/16.

## Cambios clave respecto a 2017/16

| Aspecto | EBA GL 2017/16 | EBA GL 2022/01 |
|---|---|---|
| Frecuencia backtesting | Anual | Semestral |
| Segmentación | Por clase rating | Por clase + segmento + geografía |
| Benchmarks | Default rate LTP | Default rate LTP + externos |
| LGD backtesting | Por segmento | Por segmento + tipo colateral |
| EAD backtesting | No requerido | Requerido (conversión CCF) |
| Umbral materialidad | ±10% | ±5% |
| Reporting a supervisor | Anual | Trimestral |

## Implicaciones para el pipeline

1. Backtesting PD/LGD debe hacerse **cada 6 meses**, no anualmente
2. EAD requiere backtesting con CCF (Credit Conversion Factor)
3. Umbral de materialidad más estricto (±5% vs ±10%) — más falsos positivos
4. Reporte a supervisor trimestral — requiere automatización

## Estado actual

Pipeline configurado para backtesting anual. No hay proceso semestral
automatizado. El reporting trimestral se hace manualmente.

## Acción

Actualizar scheduler SAS para ejecutar backtesting cada 6 meses.
Implementar backtesting EAD con CCF.

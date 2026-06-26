---
id: "mv_lgd_calibration_drift"
type: "insight"
priority: 0.7
tags: [model validation, LGD, calibration, drift, backtesting]
fields: [LGD_ESTIMADA, LGD_REALIZADA]
articles: [eba_gl_2017_16]
source: "Backtesting anual LGD — calibración vs realizado 2024"
feedback: false
---

# Backtesting LGD 2024: calibración sobreestima LGD en CORP y subestima en HIPOTECA

Resultados del backtesting anual de calibración LGD (EBA GL 2017/16 §6.4):

| Segmento | LGD estimada media | LGD realizada media | Sesgo | ¿Significativo? |
|---|---|---|---|---|
| CORP | 0.52 | 0.41 | +0.11 | SÍ (p<0.01) |
| HIPOTECA | 0.28 | 0.35 | -0.07 | SÍ (p<0.05) |
| RETAIL | 0.38 | 0.36 | +0.02 | NO (p=0.32) |

## Interpretación
- **CORP**: La calibración sobreestima LGD en 11 puntos. Posible causa: la tasa de recuperación fija 0.4 es demasiado baja para CORP (recovery real ~0.52), lo que infla LGD_REALIZADA de referencia → calibración conservadora.
- **HIPOTECA**: Subestimación de 7 puntos. La tasa de recuperación 0.4 infraestima el recovery real hipotecario (~0.48).
- **RETAIL**: Calibración adecuada.

## Acción
Revisar la tasa de recuperación fija 0.4. Separar por segmento mejoraría la calibración. Impacto en ECL: sobrestimado en CORP (provisiones excesivas) e infraestimado en HIPOTECA.

## Relevancia
EBA GL 2017/16 §6.4 requiere backtesting anual y corrección de sesgos significativos.

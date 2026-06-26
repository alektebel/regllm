---
id: "mv_sharpe_ratio_validation"
type: "insight"
priority: 0.5
tags: [model validation, Sharpe, discriminación, PD]
fields: [PD_ESTIMADA, RATING_GRADO]
articles: [eba_gl_2017_16]
source: "Validación anual PD — estadístico Sharpe y poder discriminatorio"
feedback: false
---

# Sharpe Ratio del modelo PD: 0.52 — por debajo del umbral mínimo (0.60)

El estadístico de Sharpe (poder discriminatorio del modelo PD) se calculó en 0.52 para 2024, frente al mínimo regulatorio de 0.60 (EBA GL 2017/16 §5.5).

## Detalle por segmento

| Segmento | Sharpe 2023 | Sharpe 2024 | Cambio | ¿OK? |
|---|---|---|---|---|
| CORP | 0.61 | 0.58 | -0.03 | NO |
| HIPOTECA | 0.65 | 0.63 | -0.02 | SÍ |
| RETAIL | 0.58 | 0.49 | -0.09 | NO |

## Causa probable
RETAIL muestra la mayor caída. Posible causa: el modelo PD no captura el deterioro del segmento de consumo (tipos altos, inflación). Contratos con buen rating histórico empiezan a impagarse sin que el modelo lo anticipe.

## Acciones
- Recalibrar modelo PD retail con datos post-inflación
- Añadir variable macro (tipo de interés, IPC) como predictor
- Informar a BdE con plan de remediación en 6 meses.

## Nota
Este es un hallazgo de validación, no un bug del pipeline. Pero el agente debe conocerlo para contextualizar consultas sobre calidad del modelo.

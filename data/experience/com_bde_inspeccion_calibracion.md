---
id: "com_bde_inspeccion_calibracion"
type: "insight"
priority: 0.7
tags: [BdE, inspección, calibración, LGD, modelos]
fields: [LGD_ESTIMADA, LGD_REALIZADA, MoC]
articles: [eba_gl_2017_16]
source: "Carta inspección BdE sobre modelos — conclusiones preliminares"
feedback: {"type": "finding", "original": "Modelo LGD calibrado según guía interna del banco", "corrected": "BdE requiere recalibración LGD con ventana mínima 7 años (vs 5 actual) y MoC documentado por fuente de incertidumbre (categorías A/B/C)."}
---

# [BdE] Carta de inspección in situ — conclusiones preliminares modelos IRB

De: supervision.inspeccion@bde.es
Para: direccion.riesgos@banco.es
Asunto: Acta de inspección — Modelos IRB LGD/PD (ref. INS-2025-0089)

Durante la inspección in situ realizada del 10 al 28 de marzo de 2025, se han identificado las siguientes deficiencias preliminares:

## Deficiencia 1: Ventana de calibración LGD insuficiente
EBA GL 2017/16 §6.3 requiere mínimo **7 años** para calibración de LGD. El pipeline utiliza **5 años** (parámetro `min_calib_years=5` en `lgd_macros.sas:60`).

## Deficiencia 2: MoC no desglosado por categoría EBA
EBA GL 2017/16 §4.4 requiere que el MoC se desglose en:
- Categoría A: errores de estimación conocidos pero no cuantificables
- Categoría B: fuentes de incertidumbre potencial
- Categoría C: error general de estimación

El pipeline aplica MoC = 5% fijo sin desglose.

## Deficiencia 3: Backtesting sin segmentación
El backtesting LGD se realiza a nivel agregado. EBA GL 2017/16 §6.4 requiere backtesting por segmento de calibración.

## Plazo de respuesta
30 días hábiles para presentar plan de remediación.

## Acción
Las 3 deficiencias requieren cambios en el pipeline SAS. El agente debe conocerlas para contextualizar consultas sobre calibración LGD.

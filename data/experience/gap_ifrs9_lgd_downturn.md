---
id: "gap_ifrs9_lgd_downturn"
type: "insight"
priority: 0.6
tags: [gap, IFRS9, LGD, downturn, EBA, calibración]
fields: [LGD_ESTIMADA, LGD_DOWNTURN, ECL]
articles: [ifrs9_5_5_3, eba_gl_2017_16_sec6, circular_6_2016_art_15]
source: "Revisión metodología IFRS9 — LGD downturn vs TTC"
feedback: false
---

# LGD downturn no implementado — ECL no refleja escenario adverso

IFRS 9 §5.5.3 requiere que ECL refleje expectativas en escenarios
posibles, incluyendo downturn. El pipeline solo calcula LGD TTC
(through-the-cycle) sin ajuste downturn.

## Metodología requerida vs actual

| Aspecto | Requerido IFRS 9 | Actual pipeline |
|---|---|---|
| LGD base | TTC (media ciclo) | TTC |
| Ajuste downturn | Ponderado por escenario | NO implementado |
| Escenarios | Base, Upside, Downside | No existe |
| MoC | Conservadurismo + Downturn | Solo 5% fijo |

## Causa raíz

`proj_03_suelos_lgd.sas`:
```sas
/* No hay lógica downturn */
LGD_CON_MOC = LGD_ESTIMADA + (0.05 * LGD_ESTIMADA);  /* solo MoC fijo */
```

No hay ponderación por escenario económico ni variables macro
(PIB, desempleo, IPC) que ajusten LGD en periodo adverso.

## Impacto

- ECL en escenario downside estaría subestimado ~25-40% según simulaciones
- Riesgo de infra-provisionamiento en recesión
- Auditoría externa identificará como gap regulatorio

## Recomendación

Implementar modelo de ajuste downturn basado en:
```sas
LGD_DOWNTURN = LGD_TTC * F(desempleo, PIB, IPC);
/* MoD = MAX(LGD_TTC, LGD_DOWNTURN) */
```

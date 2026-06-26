---
id: exp_moc
type: insight
priority: 0.6
tags: [MoC, LGD, conservadurismo]
fields: [MoC, LGD_ESTIMADA, LGD_CON_MOC]
articles: []
source: "Análisis de pipeline LGD (2025-06)"
feedback: false
---

# MoC calculado como 5% de LGD_ESTIMADA en proj_03:38

El **Margin of Conservatism (MoC)** se calcula como el 5% de
`LGD_ESTIMADA`:

```sas
MoC = 0.05 * LGD_ESTIMADA;  /* línea 38 */
```

## Propósito

Añadir un margen de conservadurismo sobre la LGD estimada,
siguiendo prácticas prudenciales de la EBA.

## Relación con ECL

`MoC` se suma a `LGD_ESTIMADA` para formar `LGD_CON_MOC`:
```
LGD_CON_MOC = LGD_ESTIMADA + MoC
ECL = PD_ESTIMADA × LGD_CON_MOC × EAD
```

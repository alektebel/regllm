---
id: exp_hipoteca_floor
type: insight
priority: 0.6
tags: [LGD, hipoteca, mortgage, CRR, Art.154]
fields: [LGD_ESTIMADA, LGD_FLOOR]
articles: [CRR_154, art_15_dotaciones_minimas]
source: "Análisis regulatorio LGD floors"
feedback: false
---

# Hipoteca LGD floor 30% según CRR Art.154(3)

Para exposiciones hipotecarias, el suelo de LGD es **30%** según
el CRR Artículo 154(3).

## Implementación SAS

```sas
if COLATERAL_TIPO = "HIPOTECA" then LGD_FLOOR = 0.30;
```

## Aplicación

```sas
LGD_FLOOR_APLICADO = MAX(LGD_ESTIMADA, LGD_FLOOR);
```

Si una hipoteca tiene LGD_ESTIMADA calculada en 0.25, el floor
la eleva a 0.30.

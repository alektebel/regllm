---
id: exp_lgd_floor_corp
type: insight
priority: 0.7
tags: [LGD, floor, CORP]
fields: [LGD_ESTIMADA, LGD_FLOOR, LGD_FLOOR_APLICADO]
articles: [art_15_dotaciones_minimas, CRR_161]
source: "Análisis regulatorio LGD floors"
feedback: false
---

# LGD floor CORP aumentó de 45% a 50% en Circular 4/2022

El suelo de LGD para exposiciones corporativas (`SEGMENTO=CORP`)
sin colateral real (`COLATERAL_TIPO=NINGUNA`) pasó de **0.45 a 0.50**
según la modificación introducida por Circular 4/2022.

## Implementación

En `proj_03_suelos_lgd.sas:13-16`:

```sas
if COLATERAL_TIPO = "HIPOTECA" then LGD_FLOOR = 0.30;
else if SEGMENTO in ("CORP", "NINGUNA") then LGD_FLOOR = 0.50;
else LGD_FLOOR = 0;
```

## Base regulatoria

- **CRR Art. 161(1)(b)**: Suelo LGD 50% para exposiciones sin colateral
- **Circular 4/2022**: Modifica Circular 6/2016 actualizando el umbral

## LGD_FLOOR_APLICADO

`LGD_FLOOR_APLICADO = MAX(LGD_ESTIMADA, LGD_FLOOR)` (línea 27).

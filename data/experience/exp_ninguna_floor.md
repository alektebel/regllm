---
id: exp_ninguna_floor
type: insight
priority: 0.7
tags: [LGD, floor, NINGUNA, CORP]
fields: [LGD_FLOOR, LGD_FLOOR_APLICADO, LGD_ESTIMADA]
articles: [art_15_dotaciones_minimas, CRR_161, circular_4_2022]
source: "Análisis regulatorio LGD floors"
feedback: false
---

# LGD floor 50% para CORP sin colateral (NINGUNA) y lagunas Art.15

CORP con `COLATERAL_TIPO = NINGUNA` tiene LGD floor **0.50** según CRR Art.161(1)(b).

## Tabla completa Art.15 vs pipeline

| Segmento | Colateral | Floor Art.15 | Pipeline | Gap |
|---|---|---|---|---|
| CORP | NINGUNA | 0.50 | 0.50 | - |
| CORP | FINANCIERO | 0.35 | 0 (ELSE) | **SÍ** |
| RETAIL | PERSONAL | 0.35 | 0 (ELSE) | **SÍ** |
| RETAIL | NINGUNA | 0.45 | 0 (ELSE) | **SÍ** |
| HIPOTECA | HIPOTECA | 0.25 (Art) / 0.30 (Circular) | 0.30 | - |
| HIPOTECA | NINGUNA | 0.40 | 0 (ELSE) | **SÍ** |

Pipeline solo implementa HIPOTECA→0.30 y CORP+NINGUNA→0.50; el resto cae a ELSE=0.

## Implementación

En `proj_03_suelos_lgd.sas:17-21`:
```sas
ELSE IF SEGMENTO = 'CORP' AND COLATERAL_TIPO = 'NINGUNA' THEN DO;
    LGD_FLOOR = 0.50;
    IF LGD_ESTIMADA < 0.50 THEN LGD_ESTIMADA = 0.50;
END;
```

`lgd_macros.sas:289-290` también maneja string vacío:
`WHEN (SEGMENTO = 'CORP' AND COLATERAL_TIPO IN ('NINGUNA', ''))`

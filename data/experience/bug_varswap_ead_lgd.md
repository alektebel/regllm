---
id: "bug_varswap_ead_lgd"
type: "insight"
priority: 0.7
tags: [bug, varswap, LGD, EAD, floor, hipoteca]
fields: [LGD_ESTIMADA, EAD, LGD_FLOOR]
articles: [CRR_154]
source: "toy_lgd/01_easy_varswap — revisión condición floor hipoteca"
feedback: false
---

# Variable swap: EAD usado en lugar de LGD_ESTIMADA en condición de floor hipotecario

En `proj_03_suelos_lgd.sas` (o equivalente en versiones anteriores) la línea:
```sas
IF COLATERAL_TIPO = 'HIPOTECA' AND EAD < 0.30 THEN LGD_ESTIMADA = 0.30;
```
usa `EAD` (cientos de miles) en lugar de `LGD_ESTIMADA`. Como `EAD < 0.30` siempre es FALSO, el floor del 30% nunca se aplica.

## Causa raíz
Error de copy-paste o confusión de nombres (EAD y LGD son numéricos adyacentes en el esquema). SAS no genera advertencia.

## Fix
```sas
IF COLATERAL_TIPO = 'HIPOTECA' AND LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;
```

## Impacto
Hipotecas con LGD entre 0-30% no reciben floor regulatorio. Infracción CRR Art.154(3).

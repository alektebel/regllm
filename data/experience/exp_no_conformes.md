---
id: exp_no_conformes
type: insight
priority: 0.5
tags: [validación, floors, no_conformes]
fields: [PD_ESTIMADA, LGD_ESTIMADA]
articles: []
source: "Análisis de validación SAS (2025-06)"
feedback: false
---

# no_conformes: ciclos que no cumplen floors mínimos regulatorios

La tabla `no_conformes` contiene los ciclos que no pasan la
validación de floors mínimos de PD o LGD.

## Generación

Se genera en `proj_03_suelos_lgd.sas:45-50`:

```sas
data work.no_conformes;
  set work.lgd_final;
  if PD_ESTIMADA < 0.0005 or LGD_ESTIMADA < LGD_FLOOR then output;
run;
```

## Propósito

Identificar ciclos que requieren revisión manual porque sus
parámetros estimados están por debajo de los mínimos regulatorios.

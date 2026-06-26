---
id: "dq_ead_missing_not_distinct"
type: "insight"
priority: 0.5
tags: [data quality, EAD, missing, diagnóstico]
fields: [EAD, ECL, LGD_ESTIMADA, PD_ESTIMADA]
articles: []
source: "eval_dataset eval_039 — análisis ECL missing"
feedback: false
---

# EAD missing no se distingue de LGD missing como causa de ECL missing

Cuando `ECL` sale missing, el pipeline siempre reporta "ECL missing (posible campo no informado)" sin distinguir si la causa es:
- EAD missing (problema upstream en source data)
- LGD_ESTIMADA missing (bug SW_FUSION)
- PD_ESTIMADA missing (caso raro)

## Problema
Sin diagnóstico de causa raíz, un analista pierde horas rastreando. El mensaje de error es genérico.

## Mejora propuesta
```sas
IF ECL = . THEN DO;
    IF EAD = . THEN PUT "ECL missing por EAD no informado";
    ELSE IF LGD_ESTIMADA = . THEN PUT "ECL missing por LGD missing (posible fusion)";
    ELSE IF PD_ESTIMADA = . THEN PUT "ECL missing por PD missing";
    ELSE PUT "ECL missing por causa desconocida";
END;
```

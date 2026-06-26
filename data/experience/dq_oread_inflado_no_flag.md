---
id: "dq_oread_inflado_no_flag"
type: "insight"
priority: 0.6
tags: [data quality, OR_EAD, inflado, no_conformes, flag]
fields: [OR_EAD, NO_CONFORMES, ECL]
articles: []
source: "eval_dataset eval_049 — revisión flags no_conformes"
feedback: false
---

# OR_EAD inflado no tiene flag en tabla no_conformes

El bug de JOIN no único (ID_FUSION_FINAL) infla OR_EAD silenciosamente. La tabla `no_conformes` no tiene ninguna validación que detecte esto.

Ciclos con OR_EAD artificialmente alto pasan sin alerta hasta que un humano revisa ECLs fuera de rango.

## Check propuesto
```sas
IF OR_EAD > (SELECT PERCENTILE(0.99, OR_EAD) FROM mylib.cycles_hist)
    THEN FLAG_NC = 1; MOTIVO = 'OR_EAD potencialmente inflado';
```
O usar ranges esperados por segmento: CORP max ~500k, RETAIL max ~100k, etc.

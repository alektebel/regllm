---
id: "bug_case_sensitive"
type: "insight"
priority: 0.6
tags: [bug, case-sensitive, colateral, UPCASE]
fields: [COLATERAL_TIPO, LGD_FLOOR, LGD_ESTIMADA]
articles: []
source: "toy_lgd/07_harder_type_coerce — revisión comparación strings"
feedback: false
---

# Comparación case-sensitive en COLATERAL_TIPO

```sas
IF COLATERAL_TIPO = 'HIPOTECA' THEN LGD_FLOOR = 0.30;
```

SAS es case-sensitive. Datos fuente tienen casing inconsistente ('HIPOTECA', 'hipoteca', 'Hipoteca') por integración post-merger. Solo la variante mayúscula recibe floor.

## Causa raíz
Asunción de calidad de datos. Integración post-fusión introduce variaciones de casing. SAS no genera advertencia.

## Fix
```sas
IF UPCASE(COLATERAL_TIPO) = 'HIPOTECA' THEN LGD_FLOOR = 0.30;
```
O estandarizar en la capa de entrada.

## Impacto
Algunas hipotecas no reciben floor regulatorio. Inconsistencia silenciosa.

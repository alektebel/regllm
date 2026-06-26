---
id: "bug_op_lgd_lookup"
type: "insight"
priority: 0.6
tags: [bug, lookup, OP_LGD, cross-reference]
fields: [OP_LGD, OP_LGD_FLAG, OP_LGD_IS_SECURED, ID_CONTR_CICLO_LGD]
articles: []
source: "toy_lgd/09_op_lgd_lookup — revisión función _LOOKUP"
feedback: false
---

# Wrong lookup key en OP_LGD cross-reference

```sas
/* BUG: pasa ID_CONTR_CICLO_LGD (ID del ciclo actual) en vez de OP_LGD (ID del contrato referenciado) */
_LOOKUP('SECURED_CYCLES', 'ID_CONTRATO', ID_CONTR_CICLO_LGD, 'SECURED');
```

Todo contrato no garantizado con OP_LGD referenciando a otro contrato se marca incorrectamente como problema, incluso cuando el contrato referenciado SÍ está garantizado.

## Causa raíz
Variable incorrecta pasada a la función _LOOKUP. Posible error de refactor (renombrar variable sin actualizar todas las referencias).

## Fix
```sas
_LOOKUP('SECURED_CYCLES', 'ID_CONTRATO', OP_LGD, 'SECURED');
```

## Impacto
Falsos positivos en validación de garantías cruzadas. Ruido en informes de excepciones.

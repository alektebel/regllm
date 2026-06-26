---
id: "bug_filter_threshold"
type: "insight"
priority: 0.6
tags: [bug, filter, provision_period, threshold]
fields: [PROVISION_PERIOD_MONTHS]
articles: [art_12_periodos_dotacion]
source: "toy_lgd/04_medium_filter — revisión filtro WHERE"
feedback: false
---

# WHERE filter con umbral y operador incorrectos

```sas
WHERE PROVISION_PERIOD_MONTHS > 12;  /* BUG: debiera ser >= 9 */
```

## Problemas
1. Umbral incorrecto: 12 en vez de 9 (Circular 6/2016 Art.12 requiere mínimo 9 meses)
2. Operador incorrecto: `>` en vez de `>=` (excluye ciclos con exactamente 9, 10, 11 meses)

Caso real: CIC_004 con 9 meses de provision period fue excluido erróneamente.

## Causa raíz
Posible confusión con una versión anterior de la regulación. Umbral 12 podría venir de requerimiento CORP.

## Fix
```sas
WHERE PROVISION_PERIOD_MONTHS >= 9;
```

## Impacto
Pérdida silenciosa de filas válidas. Ciclos con 9-11 meses desaparecen del pipeline sin alerta.

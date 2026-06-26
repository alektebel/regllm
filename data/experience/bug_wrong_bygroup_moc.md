---
id: "bug_wrong_bygroup_moc"
type: "insight"
priority: 0.6
tags: [bug, MoC, BY-group, aggregation, segmento]
fields: [MoC, LGD_CON_MOC, SEGMENTO, COLATERAL_TIPO]
articles: []
source: "toy_lgd/06_hard_agg_level — revisión segmentación MoC"
feedback: false
---

# BY-group incorrecto: COLATERAL_TIPO en vez de SEGMENTO en cómputo de MoC

El MoC se calcula agrupando por `COLATERAL_TIPO` en lugar de `SEGMENTO`:
```sas
PROC MEANS DATA=...;
    VAR LGD_ESTIMADA;
    BY COLATERAL_TIPO;  /* BUG: debiera ser SEGMENTO */
    OUTPUT OUT=seg_means MEAN=SEG_LGD_MEAN;
RUN;
```

## Por qué es sutil
COLATERAL_TIPO y SEGMENTO están correlacionados (ej: HIPOTECA→MORTGAGE), por lo que la mayoría de filas obtienen el valor correcto por coincidencia. Pero un contrato CORP con colateral FINANCIERO se compara contra la media incorrecta.

## Fix
```sas
BY SEGMENTO;
```

## Impacto
MoC ligeramente incorrecto para combinaciones minoritarias (SEGMENTO×COLATERAL). ECL y RWA ligeramente desviados.

---
id: exp_oread_join_bug
type: insight
priority: 0.7
tags: [bug, EAD, OR_EAD_TIT, JOIN]
fields: [OR_EAD_TIT, OR_EAD, EAD]
articles: []
source: "Análisis de calidad EAD (2025-06)"
feedback: false
---

# OR_EAD_TIT inflado por JOIN no único en proj_02

Bug documentado: `OR_EAD_TIT` se infla porque el JOIN con la
tabla de garantías en `proj_02_enriquecimiento_ead.sas:28-39`
**no es 1:1**.

## Causa

Un contrato con N garantías produce N filas en el JOIN,
duplicando `OR_EAD_TIT` N veces:

```sas
proc sql;
  create table work.enriquecido as
  select a.*, b.VALOR_GARANTIA
  from work.contratos a
  left join work.garantias b
    on a.CICLO_ID = b.CICLO_ID;  /* ← 1:N */
quit;
```

## Impacto

- `OR_EAD_TIT` se multiplica por el número de garantías
- `EAD` (derivada) también se infla
- `ECL` (PD × LGD × EAD) se infla en cascada

## Solución propuesta

Agregar `DISTINCT` o hacer `SUM` agrupado por ciclo:
```sas
select a.*, sum(b.VALOR_GARANTIA) as VALOR_GARANTIA
from work.contratos a
left join work.garantias b ...
group by a.CICLO_ID;
```

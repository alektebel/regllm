---
id: "xs_bde_cir_6_2016_art_16"
type: "insight"
priority: 0.7
tags: [BdE, circular, regulation, Stage 3, LGD, multiplicador]
fields: [LGD_FLOOR_APLICADO, STAGE_IFRS9, LGD_CON_MOC, ECL]
articles: [circular_6_2016_art_16]
source: "Circular 6/2016 del BdE — Artículo 16: Multiplicador Stage 3"
feedback: false
---

# [REGLAMENTO] Art.16 Circular 6/2016 — Multiplicador 1.20 para Stage 3

La Circular 6/2016 del Banco de España establece en su Artículo 16:

## Texto oficial

> "Artículo 16. Suelo de pérdida en exposiciones dudosas.
>
> 1. Para las exposiciones clasificadas como dudosas (Stage 3 IFRS 9)
>    se aplicará un multiplicador de 1,20 sobre el suelo de pérdida
>    (LGD_FLOOR_APLICADO) establecido en el artículo anterior.
>
> 2. El multiplicador se aplicará después del suelo y antes del MoC:
>    LGD_PRE_MOC = LGD_FLOOR_APLICADO × 1,20
>
> 3. El MoC se calculará sobre LGD_PRE_MOC."

## Estado de implementación

**NO implementado** en `proj_03_suelos_lgd.sas`.
Ver feedback `exp_stage3_conflict` y gap documentado.

## Flujo correcto

```
LGD_ESTIMADA → MAX(LGD_ESTIMADA, LGD_FLOOR) → LGD_FLOOR_APLICADO
→ IF STAGE_IFRS9=3 THEN LGD_PRE_MOC = LGD_FLOOR_APLICADO × 1.20
→ LGD_CON_MOC = LGD_PRE_MOC + MoC
→ ECL = PD × LGD_CON_MOC × EAD
```

## Referencias

- Circular 6/2016, Artículo 15: Suelos mínimos
- Circular 6/2016, Artículo 16: Multiplicador Stage 3 (este)
- Circular 4/2022: Modifica floors en Art.15 pero no Art.16

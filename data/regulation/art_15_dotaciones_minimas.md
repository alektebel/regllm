# Artículo 15 — Dotaciones Mínimas por Segmento durante el Período de Dotación

**Norma:** Circular 6/2016 del Banco de España, modificada por Circular 4/2022  
**Epígrafe:** Tasas mínimas de pérdida esperada y suelos de parámetros durante el período de dotación regulatoria

## Ámbito de aplicación

Durante el período de dotación activo (`PROVISION_PERIOD_MONTHS` > 0), los parámetros de riesgo empleados en el cálculo de la pérdida esperada (`ECL`) deben respetar los suelos mínimos establecidos en el presente artículo. Estos suelos se registran en los campos `LGD_FLOOR_APLICADO` y `PD_ESTIMADA` del ciclo correspondiente.

## Suelos mínimos de LGD

El campo `LGD_ESTIMADA` no puede ser inferior a los valores mínimos de `LGD_FLOOR_APLICADO` definidos a continuación:

| Segmento | Colateral | LGD_FLOOR_APLICADO mínimo |
|---|---|---|
| CORP | NINGUNA | 0.50 |
| CORP | FINANCIERO / COLATERAL_FIN | 0.35 |
| RETAIL | PERSONAL | 0.35 |
| RETAIL | NINGUNA | 0.45 |
| MORTGAGE | HIPOTECA | 0.25 |
| MORTGAGE | NINGUNA | 0.40 |

El campo `LGD_FLOOR_APLICADO` debe reflejar el suelo efectivamente aplicado en el cálculo de `ECL`. Cuando `LGD_ESTIMADA` > `LGD_FLOOR_APLICADO`, se utiliza `LGD_ESTIMADA`; en caso contrario, se utiliza `LGD_FLOOR_APLICADO`.

## Suelo mínimo de PD

Durante cualquier período de dotación activo, `PD_ESTIMADA` no puede ser inferior al 0.05%, con independencia del `SEGMENTO` o del `RATING_GRADO`. Este suelo fue actualizado de 0.03% a 0.05% mediante la Circular 4/2022 y es de aplicación a todos los ciclos con `PROVISION_PERIOD_MONTHS` > 0 desde el 1 de enero de 2023.

## Interacción con el campo MOC

El ajuste por incertidumbre del modelo (`MOC`) se aplica sobre la `LGD_ESTIMADA` antes de comparar con `LGD_FLOOR_APLICADO`. La secuencia de cálculo es:

1. `LGD_ajustada` = `LGD_ESTIMADA` × (1 + `MOC`)
2. `LGD_efectiva` = max(`LGD_ajustada`, `LGD_FLOOR_APLICADO`)
3. `ECL` = `PD_ESTIMADA` × `LGD_efectiva` × `EAD`

Cuando `LGD_ajustada` < `LGD_FLOOR_APLICADO`, la diferencia debe reportarse en el log de validación y queda registrada en el campo `NO_CONFORMES` si supera el 10% del valor del suelo.

## Suelos durante STAGE_IFRS9 = 3

Cuando `STAGE_IFRS9` = 3, se aplican suelos adicionales de `PD_ESTIMADA` = 1.0 (probabilidad de default unitaria para Stage 3) y el `LGD_FLOOR_APLICADO` se incrementa en un factor de 1.20 respecto a los valores de la tabla anterior.

## Justificación

Los suelos establecidos en este artículo están calibrados para asegurar que la provisión regulatoria cubre, como mínimo, el percentil 75 de pérdidas históricas observadas en el sistema bancario español en períodos de contracción del ciclo crediticio. El suelo de `PD_ESTIMADA` al 0.05% previene la sub-dotación en carteras de alta calidad crediticia durante fases de expansión en las que el modelo de PD puede infraestimar el riesgo sistémico latente.

## Referencias cruzadas

- Artículo 8 — Clasificación de ciclos (`SEGMENTO`, `COLATERAL_TIPO`, `STAGE_IFRS9`)
- Artículo 12 — Períodos mínimos de dotación (`PROVISION_PERIOD_MONTHS`)
- Artículo 23 — Condiciones de liberación (`CURE_FLAG`, `LGD_FLOOR_APLICADO`)

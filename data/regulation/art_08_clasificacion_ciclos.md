# Artículo 8 — Clasificación de los Ciclos de Recuperación

**Norma:** Circular 6/2016 del Banco de España, modificada por Circular 4/2022  
**Epígrafe:** Criterios de clasificación del ciclo crediticio para la dotación de provisiones

## Ámbito de aplicación

El presente artículo define los criterios de clasificación de los ciclos de recuperación (`CICLO_ID`) en fases de expansión, contracción y crisis, a efectos del cálculo de la provisión regulatoria obligatoria.

La clasificación aplica a todas las exposiciones registradas en `mylib.ciclos_recuperacion` con independencia del `SEGMENTO` al que pertenezcan.

## Fases del ciclo crediticio

Los ciclos se clasifican en tres fases según el valor acumulado de `DPDS` (días en default) en la ventana de observación definida por `VENTANA_OBSERVACION_YEARS`:

| Fase | Condición sobre DPDS | Identificador regulatorio |
|---|---|---|
| Expansión | DPDS < 90 | FASE_EXPANSION |
| Contracción | 90 ≤ DPDS < 360 | FASE_CONTRACCION |
| Crisis | DPDS ≥ 360 | FASE_CRISIS |

## Criterios adicionales por segmento

Para el `SEGMENTO` CORP (corporativo), la clasificación se ajusta con el campo `RATING_GRADO`:

- `RATING_GRADO` ≤ 3 y `DPDS` ≥ 60 → anticipa la fase de contracción independientemente del umbral de DPDS.
- `RATING_GRADO` ≥ 8 → rebaja el umbral de crisis a `DPDS` ≥ 270.

Para el `SEGMENTO` MORTGAGE (hipotecario), el campo `COLATERAL_TIPO` HIPOTECA reduce el umbral de crisis a `DPDS` ≥ 480 debido a la mayor capacidad de recuperación colateralizada.

## Relación con STAGE_IFRS9

La fase del ciclo debe ser coherente con la clasificación IFRS9 del contrato:

- `STAGE_IFRS9` = 1 → compatible únicamente con FASE_EXPANSION.
- `STAGE_IFRS9` = 2 → compatible con FASE_EXPANSION o FASE_CONTRACCION.
- `STAGE_IFRS9` = 3 → requiere FASE_CONTRACCION o FASE_CRISIS obligatoriamente.

Cualquier inconsistencia entre `STAGE_IFRS9` y la fase calculada deberá ser reportada como excepción en el campo `NO_CONFORMES` del pipeline de cálculo.

## Ventana de calibración

La clasificación de fase utiliza la ventana definida por `VENTANA_CALIBRACION_YEARS`. Cambios de fase dentro de la ventana de calibración no son efectivos hasta el cierre del período de observación (`VENTANA_OBSERVACION_YEARS`).

## Referencias cruzadas

- Artículo 12 — Períodos de dotación por fase de ciclo (`PROVISION_PERIOD_MONTHS`)
- Artículo 15 — Dotaciones mínimas por segmento (`LGD_FLOOR_APLICADO`, `PD_ESTIMADA`)
- Artículo 23 — Condiciones de liberación de provisión (`CURE_FLAG`)

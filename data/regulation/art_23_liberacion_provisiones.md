# Artículo 23 — Condiciones para la Liberación de Provisiones

**Norma:** Circular 6/2016 del Banco de España, modificada por Circular 4/2022  
**Epígrafe:** Requisitos para la liberación o reducción de provisiones regulatorias al cierre del período de dotación

## Ámbito de aplicación

El presente artículo establece las condiciones bajo las cuales una entidad puede liberar o reducir las provisiones regulatorias asociadas a un ciclo de recuperación (`CICLO_ID`). La liberación parcial o total sólo procede cuando se cumplen simultáneamente todos los requisitos descritos en este artículo.

## Requisito 1 — Expiración del período de dotación mínimo

El valor de `PROVISION_PERIOD_MONTHS` acumulado para el ciclo debe ser igual o superior al mínimo establecido en el Artículo 12 para el `SEGMENTO` y fase de ciclo correspondiente.

El cómputo se realiza en meses naturales completos desde la fecha de entrada en default hasta la fecha de evaluación.

## Requisito 2 — Indicador de cura confirmado

El campo `CURE_FLAG` debe estar activo (`CURE_FLAG` = 1), lo que acredita que el acreditado ha regularizado su situación crediticia según los criterios definidos en la política de curas de la entidad.

La activación de `CURE_FLAG` requiere que el contrato haya permanecido fuera de default durante un mínimo de 90 días consecutivos (período de prueba). Durante este período el contrato conserva su clasificación en la fase de ciclo previa.

## Requisito 3 — Retorno a STAGE_IFRS9 ≤ 2

El campo `STAGE_IFRS9` debe haber retornado al valor 1 o 2 y mantenerse en dicho valor durante al menos dos ventanas de observación consecutivas (`VENTANA_OBSERVACION_YEARS`).

El retorno a `STAGE_IFRS9` = 1 directamente desde `STAGE_IFRS9` = 3 no está permitido salvo en casos de cancelación total del contrato. La secuencia obligatoria es: STAGE 3 → STAGE 2 (mínimo una ventana de observación) → STAGE 1.

## Requisito 4 — Suelos de parámetros cumplidos en origen

Los parámetros `LGD_FLOOR_APLICADO` y `PD_ESTIMADA` utilizados durante todo el período de dotación deben haber respetado los mínimos del Artículo 15. Si se detecta incumplimiento retroactivo, la liberación queda bloqueada hasta la corrección de los registros históricos en `mylib.ciclos_recuperacion`.

## Proceso de liberación

Cuando se cumplen los cuatro requisitos anteriores, la provisión puede liberarse de forma progresiva según el siguiente calendario:

| Meses adicionales tras cumplimiento | Liberación máxima acumulada |
|---|---|
| 0–6 meses | 25% de la provisión |
| 6–12 meses | 50% de la provisión |
| 12–18 meses | 75% de la provisión |
| > 18 meses | 100% de la provisión |

La provisión base se calcula como `ECL` = `PD_ESTIMADA` × `LGD_FLOOR_APLICADO` × `EAD` en la fecha de inicio del proceso de liberación.

## Excepciones y no conformidades

Los ciclos que no cumplan los requisitos anteriores al cierre del período de dotación se registran en el campo `NO_CONFORMES` del pipeline de cálculo (`lgd_pipeline`). Las no conformidades deben ser revisadas manualmente por el equipo de Riesgos antes del cierre contable trimestral.

El campo `OR_EAD_TIT` se utiliza en operaciones de titulización para ajustar la `EAD` efectiva aplicable al cálculo de provisión en el proceso de liberación para el segmento CORP.

## Justificación

Los requisitos de liberación establecidos en este artículo garantizan que la provisión regulatoria no se libera prematuramente en exposiciones que, aunque técnicamente curadas, pueden presentar recaída en default dentro del período de seguimiento post-cura. El escalonamiento en la liberación reduce el riesgo de volatilidad en los resultados contables.

## Referencias cruzadas

- Artículo 8 — Clasificación de ciclos (`DPDS`, `STAGE_IFRS9`, `VENTANA_OBSERVACION_YEARS`)
- Artículo 12 — Períodos mínimos de dotación (`PROVISION_PERIOD_MONTHS`)
- Artículo 15 — Dotaciones mínimas (`LGD_FLOOR_APLICADO`, `PD_ESTIMADA`, `ECL`, `EAD`)

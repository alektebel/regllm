# Artículo 12 — Períodos Mínimos de Dotación por Fase de Ciclo

**Norma:** Circular 6/2016 del Banco de España, modificada por Circular 4/2022  
**Epígrafe:** Duración mínima del período de dotación de provisiones en función del ciclo crediticio

## Ámbito de aplicación

El presente artículo establece la duración mínima del período de dotación de provisiones regulatorias (`PROVISION_PERIOD_MONTHS`) para cada `SEGMENTO` de exposición y fase de ciclo crediticio definida en el Artículo 8.

El campo `PROVISION_PERIOD_MONTHS` registrado en `mylib.ciclos_recuperacion` debe respetar en todo momento los mínimos establecidos en este artículo. Su incumplimiento activa el indicador de excepción del pipeline (`NO_CONFORMES`).

## Períodos mínimos por segmento

### Segmento CORP (corporativo)

| Fase del ciclo | PROVISION_PERIOD_MONTHS mínimo |
|---|---|
| FASE_EXPANSION | 12 meses |
| FASE_CONTRACCION | 18 meses |
| FASE_CRISIS | 24 meses |

Adicionalmente, cuando `COLATERAL_TIPO` = NINGUNA y el segmento es CORP, se aplica un recargo de +6 meses sobre los mínimos anteriores, debido a la mayor incertidumbre en la recuperación sin garantía real.

### Segmento RETAIL (minorista)

| Fase del ciclo | PROVISION_PERIOD_MONTHS mínimo |
|---|---|
| FASE_EXPANSION | 24 meses |
| FASE_CONTRACCION | 30 meses |
| FASE_CRISIS | 36 meses |

### Segmento MORTGAGE (hipotecario)

| Fase del ciclo | PROVISION_PERIOD_MONTHS mínimo |
|---|---|
| FASE_EXPANSION | 36 meses |
| FASE_CONTRACCION | 42 meses |
| FASE_CRISIS | 48 meses |

La mayor duración para el segmento MORTGAGE se justifica por los plazos procesales asociados a la ejecución hipotecaría, que en la práctica difieren el momento de recuperación efectiva.

## Interacción con STAGE_IFRS9

Cuando `STAGE_IFRS9` = 3, el período de dotación mínimo se amplía en 6 meses adicionales sobre los valores indicados en las tablas anteriores, con independencia del segmento. Esta ampliación es obligatoria y no admite excepción individual.

Cuando `STAGE_IFRS9` retorna a 2 desde 3, el período de dotación debe mantenerse al menos 12 meses adicionales antes de poder reducirse al mínimo correspondiente a FASE_CONTRACCION. Esto se controla mediante el campo `CURE_FLAG`.

## Cómputo del período de dotación

El período de dotación (`PROVISION_PERIOD_MONTHS`) se computa desde la fecha de primera clasificación en default hasta la fecha de cierre del ciclo o la fecha de evaluación, la que ocurra antes. El campo EAD se utiliza como base de exposición para el cálculo de la provisión efectiva durante el período.

## Justificación regulatoria

Los períodos mínimos establecidos en este artículo derivan de los percentiles históricos de recuperación observados en el sistema financiero español durante el período 2008–2015, ajustados por un factor de prudencia del 15% para reflejar la incertidumbre del modelo (MOC).

## Referencias cruzadas

- Artículo 8 — Clasificación de ciclos y fases (`DPDS`, `STAGE_IFRS9`, `SEGMENTO`)
- Artículo 15 — Dotaciones mínimas (`LGD_FLOOR_APLICADO`, `PD_ESTIMADA`)
- Artículo 23 — Condiciones de liberación (`CURE_FLAG`, `PROVISION_PERIOD_MONTHS`)

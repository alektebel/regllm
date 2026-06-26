---
id: "perf_sas_macro_overhead"
type: "insight"
priority: 0.4
tags: [performance, SAS, macro, optimization]
fields: []
articles: []
source: "Optimización pipeline SAS — runtime analysis Q1-2025"
feedback: false
---

# Macro LGD_DQ_CALL llamada 47 veces por ciclo — overhead 12min en ejecución mensual

Análisis de logging reveló que la macro `%LGD_DQ_CALL` se ejecuta 47 veces por ciclo de validación (una por cada check DQC). Cada llamada abre/cierra la tabla `mylib.cycles_check`:

```
NOTE: Table MYLIB.CYCLES_CHECK opened at line 123.
NOTE: Table MYLIB.CYCLES_CHECK closed at line 125.
... (repetido 47 veces por ciclo, × ~50k ciclos = 2.35M aperturas/cierres)
```

## Impacto
- Tiempo de ejecución: ~18 min → ~6 min después de optimizar
- 12 minutos/mes perdidos en overhead I/O
- Proporcional al número de ciclos (empeora con crecimiento de cartera)

## Fix
```sas
/* Antes: 47 llamadas separadas */
%LGD_DQ_CALL(CHECK=DQ01);
%LGD_DQ_CALL(CHECK=DQ02);
/* ... 45 más ... */

/* Después: 1 llamada batch */
%LGD_DQ_BATCH();
```
La macro batch ejecuta todos los checks en una sola pasada de la tabla.

## Lección
El overhead de I/O en SAS es significativo para procesos batch con muchas tablas pequeñas. Siempre agrupar operaciones sobre la misma tabla.

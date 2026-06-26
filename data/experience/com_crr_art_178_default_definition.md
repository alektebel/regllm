---
id: "com_crr_art_178_default_definition"
type: "insight"
priority: 0.6
tags: [CRR, default, definition, PD, DPDS, days past due]
fields: [DPDS, DEFAULT_FLAG, PD_ESTIMADA]
articles: [crr_art_178]
source: "Revisión definición de default CRR Art.178 vs pipeline"
feedback: false
---

# [CRR Art.178] Definición de default no alineada con pipeline

CRR Art. 178(1) define default como:

> "Se considerará que existe incumplimiento cuando se haya superado
> cualquiera de los siguientes umbrales:
> (a) 90 días consecutivos de mora (DPDS > 90)
> (b) La entidad considere improbable que el deudor pague"

## Pipeline actual

```sas
IF DPDS >= 90 THEN DEFAULT_FLAG = 1;  /* solo criterio (a) */
```

## Gaps identificados

| Criterio CRR Art.178 | Implementado | Notas |
|---|---|---|
| (a) 90 días mora | SÍ | DPDS >= 90 |
| (b) Improbable pago | NO | No hay flag de "improbable" |
| (c) Quiebra/suspensión | NO | No hay flag concursal |
| (d) Reestructuración forzosa | NO | Dato no disponible en pipeline |
| (e) Default de garantizador | NO | No hay flag de garantizador |

## Impacto

- DEFAULT_FLAG infraestima defaults reales (falta criterio (b)-(e))
- PD estimada basada en default histórico usa definición incompleta
- Calibración PD es optimista (menos defaults de los que debería)

## Nota

El criterio (b) requiere juicio de experto — no es automatizable
al 100%. Se necesita un proceso semiautomático con revisión del
equipo de riesgos.

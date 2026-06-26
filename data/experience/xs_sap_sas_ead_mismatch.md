---
id: "xs_sap_sas_ead_mismatch"
type: "insight"
priority: 0.7
tags: [cross-system, SAP, SAS, EAD, reconciliation]
fields: [EAD, EAD_TOTAL, OR_EAD]
articles: []
source: "Concilación mensual SAP→SAS Q2-2025 — diferencias EAD"
feedback: false
---

# Diferencias sistemáticas EAD entre SAP (origen) y SAS pipeline

La conciliación mensual EAD entre SAP (sistema origen) y la tabla `mylib.ciclos_recuperacion` en SAS muestra una diferencia sistemática de ~2.3%:

| Mes | SAP SUM(EAD) | SAS SUM(EAD) | Dif | % |
|---|---|---|---|---|
| Ene-25 | 1,234M | 1,206M | -28M | -2.3% |
| Feb-25 | 1,241M | 1,213M | -28M | -2.3% |
| Mar-25 | 1,228M | 1,200M | -28M | -2.3% |

## Causa raíz
El extracto SAP exporta contratos con `STATUS ≠ 'CANCELED'`, pero el job de importación SAS filtra adicionalmente `WHERE FECHA_ALTA >= '2020-01-01'`. Contratos anteriores a 2020 (aprox. 28M€ en EAD) se pierden silenciosamente.

## Impacto
ECL y RWA infraestimados en ~2.3%. El gap es constante pero no se había detectado porque ambas cifras se movían juntas.

## Fix
Eliminar el filtro de fecha o justificarlo con documentación. Si es intencional, documentar la exclusión.

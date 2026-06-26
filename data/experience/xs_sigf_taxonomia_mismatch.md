---
id: "xs_sigf_taxonomia_mismatch"
type: "insight"
priority: 0.6
tags: [cross-system, SIGF, taxonomía, COREP, FINREP]
fields: [SEGMENTO, COLATERAL_TIPO]
articles: [EBA_ITS_2024_02]
source: "Validación COREP Q2-2025 — mapeo taxonomía SIGF vs pipeline"
feedback: false
---

# Mapeo taxonomía SIGF vs pipeline: diferencias en clasificación COREP

La validación de envío COREP Q2-2025 reveló diferencias en la clasificación de exposiciones IRB entre la taxonomía SIGF (oficial) y el pipeline SAS:

| Categoría | SIGF | Pipeline | Diferencia |
|---|---|---|---|
| CORP→SME | 847M | 798M | -49M (empresas >250 empleados mal clasificadas) |
| RETAIL→OTHER | 234M | 289M | +55M (CLUDs pequeños clasificados como RETAIL) |
| MORTGAGE | 1,234M | 1,234M | 0 (correcto) |

## Causa raíz
El pipeline clasifica SME usando una regla simplificada (importe < 1M€), mientras que SIGF usa la definición completa de PYME de la Comisión Europea (<250 empleados, <50M€ facturación, <43M€ balance).

## Impacto
COREP enviado con clasificación incorrecta en 2 de 6 categorías. Riesgo de requerimiento de aclaración por parte del BdE.

## Fix
Usar la tabla de empleados/facturación de SAP para clasificar SME según definición UE, no por importe.

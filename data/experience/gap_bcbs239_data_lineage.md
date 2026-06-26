---
id: "gap_bcbs239_data_lineage"
type: "insight"
priority: 0.7
tags: [gap, BCBS239, lineage, data governance, ECL, RWA]
fields: [ECL, RWA, PD_ESTIMADA, LGD_ESTIMADA, EAD]
articles: [bcbs_239_principle_2]
source: "Auto-evaluación BCBS 239 — trazabilidad de datos 2025"
feedback: false
---

# BCBS 239 Principio 2: Data lineage y trazabilidad incompleta

BCBS 239 (Principio 2) exige que las entidades financieras tengan
data lineage completo y preciso para todos los datos utilizados en
reportes regulatorios (COREP, FINREP).

## Hallazgos

| Dimensión | Estado | Evidencia |
|---|---|---|
| Lineage ECL → PD × LGD × EAD | Parcial | Trazado hasta proj_03 pero no upstream |
| Lineage LGD_ESTIMADA → SAS steps | Completo | Documentado en proj_03_suelos_lgd.sas |
| Lineage PD → ratings → calibración | Incompleto | Origen ratings no documentado |
| Lineage EAD → SAP → SAS | Parcial | Gap SAP→SAS documentado (ver xs_sap_sas_ead_mismatch) |
| Diccionario de datos | Desactualizado | Columns sin descripción ni fuente |
| Transformaciones | No documentadas | JOINs, filtros, agregaciones sin metadatos |

## Riesgos

1. Auditoría no puede verificar integridad del cálculo de ECL
2. Error en upstream no es trazable a su impacto downstream
3. Nuevos miembros del equipo tardan meses en entender el pipeline

## Recomendaciones

- Implementar diccionario de datos con origen de cada campo
- Documentar cada transformación SAS con metadatos (source, target, regla)
- Vincular cada columna COREP/FINREP a su cálculo y fuente de datos
- Automatizar lineage con herramientas de parsing SAS

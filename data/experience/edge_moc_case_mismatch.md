---
id: "edge_moc_case_mismatch"
type: "quirk"
priority: 0.3
tags: [quirk, MoC, case, mapping, CSV]
fields: [MoC, MOC]
articles: []
severity: "low"
source: "audit/mapping.json — revisión naming columnas"
feedback: false
---

# MoC: nombre con mixed case inconsistente entre CSV y SAS

- En CSV de salida: `MoC` (mixed case)
- En código SAS: `MOC` (mayúscula)
- En mapping.json: referenciado como 'MoC' con nota "case mismatch"

## Impacto
Consultas DQC que usan `MOC` no encuentran la columna en el CSV. Queries deben hacer `UPPER(columna)` o el pipeline debe unificar el naming.

## Fix recomendado
Unificar a `MOC` (mayúscula, consistente con el resto de columnas SAS) y añadir alias en la exportación CSV.

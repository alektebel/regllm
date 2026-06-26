---
id: "com_bde_carta_circular_4_2022"
type: "insight"
priority: 0.7
tags: [BdE, circular, PD, floor, retail]
fields: [PD_ESTIMADA]
articles: [circular_4_2022]
source: "Boletín Oficial del Estado — Circular 4/2022 de 28 de diciembre"
feedback: false
---

# [BOE] Circular 4/2022 — modificación PD floors retail

Disposición publicada en BOE núm. 312, de 29 de diciembre de 2022.

## Extracto relevante (Artículo Único, apartado 3)

"Se modifica el artículo 15 de la Circular 6/2016, que queda redactado como sigue:

Artículo 15. Suelos mínimos.
[...]
3. El suelo de probabilidad de incumplimiento (PD) para exposiciones minoristas será del 0,03 por ciento. Para el resto de exposiciones, el suelo será del 0,05 por ciento."

## Implicaciones
- **Antes de Circular 4/2022**: PD floor ERA 0.05% para todos los segmentos
- **Después**: PD floor retail = 0.03%, CORP/soberano/otros = 0.05%
- **Pipeline SAS** actualmente usa 0.05% uniforme → INCORRECTO para retail

## Estado
Este hallazgo fue identificado por auditoría externa (ver email_auditor_pd_floor). La circular entró en vigor el 1 de enero de 2023. El pipeline lleva 2.5 años con el valor incorrecto.

## Prioridad
ALTA — corrección regulatoria mandatory, no opcional.

---
id: "email_auditor_pd_floor"
type: "insight"
priority: 0.8
tags: [email, auditor, PD, floor, retail, circular]
fields: [PD_ESTIMADA]
articles: [circular_4_2022]
source: "Email de auditoría externa 2025-05-20 — PD floor retail"
feedback: {"type": "finding", "original": "PD floor es 0.05% uniforme para todos los segmentos", "corrected": "Hallazgo de auditoría: PD floor retail debe ser 0.03% (Circular 4/2022), no 0.05%. Pipeline usa 0.05% incorrecto para retail."}
---

# [EMAIL] Hallazgo auditoría externa — PD floor retail incorrecto

De: auditor.externo@big4.com
Para: cumply.regulatorio@banco.es
CC: direccion.riesgos@banco.es
Asunto: HALLAZGO - PD floor retail Circular 4/2022

Estimados,

Como parte de la revisión independiente de cumplimiento regulatorio IRB correspondiente al ejercicio 2025, hemos identificado el siguiente hallazgo:

## Hallazgo AUD-2025-047: PD floor para retail no conforme con Circular 4/2022

**Evidencia:** En el pipeline SAS LGD, línea 42 de proj_03_suelos_lgd.sas:
```sas
IF PD_ESTIMADA < 0.0005 THEN PD_ESTIMADA = 0.0005;
```

**Requisito:** Circular 4/2022, modificación del Artículo X, establece:
- PD floor retail (PERSONAL, REVOLVING): **0.03% (0.0003)**
- PD floor corporate/sovereign: **0.05% (0.0005)**

El pipeline aplica 0.05% a TODOS los segmentos, sobrestimando PD_ESTIMADA para retail en 67%.

**Impacto:** ECL para cartera retail sobrestimado en ~0.02% × EAD. Para una cartera retail de 500M€, el exceso de provisiones es ~100k€.

**Severidad:** Media

**Plazo de corrección:** Próximo ciclo de reporting (Q3-2025)

Atentamente,
Equipo de Auditoría Regulatoria

## Acción
Corregir pipeline para usar 0.0003 en retail. Documentar como feedback.

---
id: "email_business_cure_rate"
type: "insight"
priority: 0.7
tags: [email, business, cure, LGD, hipoteca]
fields: [LGD_ESTIMADA, CURE_FLAG]
articles: []
source: "Email de equipo de negocio 2025-03-14 — cure rate CORP"
feedback: {"type": "correction", "original": "Cure rate ×0.95 aplica a todos los segmentos", "corrected": "Cure rate ×0.95 aplica SOLO a HIPOTECA. CORP y otros segmentos NO tienen ajuste por cure."}
---

# [EMAIL] Corrección: cure rate aplica solo a HIPOTECA, no a CORP

De: equipo.negocio@banco.es
Para: validacion.regulatoria@banco.es
Asunto: RE: Aplicación del cure rate LGD

Buenas,

Revisando la documentación del cure rate re-fitted 2018-2024, confirmamos que el ajuste ×0.95 se aplica EXCLUSIVAMENTE a contratos con COLATERAL_TIPO = 'HIPOTECA'.

La nota inicial que decía que aplicaba a todos los segmentos era incorrecta. Para CORP, NINGUNA, y otros segmentos no hay ajuste por cure rate en este momento. El floor del 30% sigue aplicando normalmente.

Quedo atenta a cualquier duda.

Saludos,
María García
Equipo de Negocio - Riesgos

## Acción requerida
Corregir insight automático que asumió aplicación global. El changelog 2025-q1-lgd-cure.md especifica HIPOTECA pero la redacción es ambigua.

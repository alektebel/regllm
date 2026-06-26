---
id: "email_eba_qna_pd_floor"
type: "insight"
priority: 0.7
tags: [email, EBA, Q&A, PD, floor, retail, CORP]
fields: [PD_ESTIMADA, PD_FLOOR]
articles: [circular_4_2022, eba_qa_2023_12, crr_art_161]
source: "EBA Q&A 2023-12 — PD floor aplicabilidad retail vs CORP"
feedback:
  type: clarification
  original: "PD floor aplica igual a todos los segmentos"
  corrected: "EBA Q&A 2023-12 confirma: PD floor 0.03% retail (Art.154 CRR), 0.05% CORP (Art.161 CRR). Son artículos separados con fundamentos distintos."
---

# [EBA Q&A 2023-12] PD floor: confirmación segmentación

Respuesta de EBA a consulta Q&A 2023-12 (publicada enero 2024):

## Pregunta

"¿El PD floor de 0.03% para retail (Circular 4/2022) aplica también
a exposiciones corporativas con metodología IRB?"

## Respuesta EBA

**No.** El fundamento legal es distinto:

| Segmento | PD floor | Artículo CRR | Fundamento |
|---|---|---|---|
| Retail | 0.03% | Art. 154(3) | Specific IRB retail |
| CORP | 0.05% | Art. 161(1)(b) | Minimum IRB floor |
| Soberano | 0.05% | Art. 161(1)(a) | Sovereign IRB floor |
| Hipoteca minorista | 0.03% | Art. 154(3) | Retail IRB (incluye mortgages) |

La Circular 4/2022 solo modificó el Art. 15 de Circular 6/2016 para
retail. CORP y soberano mantienen el 0.05% del CRR Art. 161.

## Implicación

El pipeline debe mantener dos floors diferenciados:
- `IF SEGMENTO IN ("RETAIL", "PERSONAL", "REVOLVING") THEN PD_FLOOR = 0.0003;`
- `ELSE PD_FLOOR = 0.0005;`

Aplicar 0.05% a retail es un hallazgo de auditoría confirmado.

---
id: "bug_ead_crm_double_count"
type: "insight"
priority: 0.7
tags: [bug, EAD, CRM, double-count, garantías]
fields: [EAD_TOTAL, OR_EAD, EAD_CRM_AJUSTADA, ECL]
articles: []
source: "Validación EAD post-CRM — Q1-2025"
feedback: false
---

# Ajuste CRM duplica reducción de EAD cuando hay múltiples garantías del mismo tipo

Cuando un contrato tiene N garantías del mismo tipo (ej: 2 avales), el ajuste CRM aplica
la reducción N veces en lugar de 1, subestimando EAD_TOTAL:

```
EAD_TOTAL = OR_EAD - SUM(CRM_HAIRCUT_i × EAD_i)  i=1..N  ← reduce N veces
```

## Causa raíz

El CRM loop itera sobre todas las garantías pero no agrupa por tipo.
`proj_02_enriquecimiento_ead.sas:71-78`:
```sas
DO i = 1 TO num_garantias;
    EAD_TOTAL = EAD_TOTAL - (CRM_HAIRCUT(i) * OR_EAD_GAR(i));
END;  /* descuenta cada garantía individualmente */
```

Para garantías del mismo colateral, el descuento debería aplicarse una sola vez.

## Impacto

Contratos con 2+ garantías del mismo tipo tienen EAD_TOTAL artificialmente baja.
Esto reduce ECL y RWA. Estimación: ~1.5% de EAD total afectado.

## Fix

```sas
/* Agrupar por tipo de garantía antes de aplicar haircut */
PROC SUMMARY DATA=garantias NWAY;
    CLASS ID_CONTRATO COLATERAL_TIPO;
    VAR OR_EAD_GAR;
    OUTPUT OUT=gar_agg SUM=OR_EAD_GAR_AGG;
RUN;
/* Aplicar haircut una vez por tipo */
```

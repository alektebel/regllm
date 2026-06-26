---
id: "gap_segment_ventana_check"
type: "insight"
priority: 0.5
tags: [gap, ventana_observacion, segmento, EBA]
fields: [VENTANA_OBSERVACION_YEARS, NO_CONFORMES, SEGMENTO]
articles: [eba_gl_2017_16]
source: "proj_03_suelos_lgd.sas:78-81 — revisión no_conformes"
feedback: false
---

# VENTANA_OBSERVACION check agnóstico a segmento — gap regulatorio

```sas
IF VENTANA_OBSERVACION_YEARS < 5 THEN DO;
    FLAG_NC = 1;
    MOTIVO = 'Ventana observación < 5 años';
END;
```

EBA GL 2017/16 §6.3 requiere:
- CORP/RETAIL: ≥5 años ✓
- HIPOTECA: ≥7 años ✗ (el chequeo no discrimina)

Hipotecas con 5-6 años pasan sin ser marcadas como no-conformes.

## Fix
```sas
IF (SEGMENTO = 'HIPOTECA' AND VENTANA_OBSERVACION_YEARS < 7)
    OR VENTANA_OBSERVACION_YEARS < 5 THEN DO;
    FLAG_NC = 1;
END;
```

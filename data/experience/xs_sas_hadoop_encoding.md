---
id: "xs_sas_hadoop_encoding"
type: "quirk"
priority: 0.5
tags: [cross-system, encoding, Hadoop, SAS, UTF-8]
fields: [COLATERAL_TIPO, SEGMENTO]
articles: []
severity: "medium"
source: "Incidente integración SAS→Hadoop — caracteres especiales"
feedback: false
---

# Problema de encoding SAS→Hadoop: caracteres latinos corrompidos

El pipeline exporta particiones a Hadoop en LATIN1, pero el consumidor Hadoop espera UTF-8. Caracteres como 'ó', 'é', 'ñ', 'í' en campos como `SEGMENTO` ('Hipoteca' vs 'HipotÃ©ca') causaban fallos en JOINs con tablas Hadoop.

## Síntoma
```sql
SELECT * FROM contracts_hadoop c
JOIN sas_export s ON c.contrato_id = s.ID_CONTRATO
WHERE c.segmento = 'CORP'  -- funciona
  AND c.segmento LIKE '%Ã©%' -- hay filas con encoding corrupto
```

## Causa raíz
Exportación SAS usa `ENCODING=LATIN1` por defecto. Hadoop asume UTF-8. Los caracteres >0x7F se corrompen.

## Fix
```sas
FILENAME exportado PIPE 'iconv -f LATIN1 -t UTF-8' ENCODING=UTF-8;
```
O configurar `%LET SAS_ENCODING=UTF-8;` antes de la exportación.

## Lección
Problemas de encoding son silenciosos — no generan error, solo datos incorrectos. Cualquier interfaz entre sistemas con distinto encoding debe validarse con un contrato específico conocido (ej: 'VALLE NORTE S.A.' con ñ).

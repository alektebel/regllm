---
id: exp_colateral_quirk
type: quirk
priority: 0.5
severity: medium
tags: [colateral, renaming, V2, V3, shim]
fields: [COLATERAL_TIPO, COLATERAL_FIN]
articles: []
source: "Detección automática en changelog analysis"
---

# COLATERAL_FIN aparece como COLATERAL_TIPO vacío en salida V2→V3

## Descripción

Los colaterales tipo `FINANCIERO` se renombraron a `COLATERAL_FIN`
en V3 (ver changelog 2025-q2-collateral-codes.md), pero **no hay
shim retroactivo**.

## Impacto

En salidas V2, `COLATERAL_TIPO` aparece vacío (`''`) para estos
contratos, lo que provoca que caigan en el `else LGD_FLOOR = 0`
del cálculo de floors.

## Severidad

Media — afecta solo a contratos con colateral financiero en
pipeline V2, pero el impacto es subestimación de LGD floor.

## Solución propuesta

Aplicar shim retroactivo antes de calcular LGD floor:
```sas
if COLATERAL_TIPO = "" and COLATERAL_FIN = 1 then
    COLATERAL_TIPO = "FINANCIERO";
```

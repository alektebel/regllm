# Table cycles_v[23]

Each row is one *recovery cycle* identified by `CICLO_ID`. Pairs across
versions share the same primary key.

## Key fields

- `CICLO_ID` (PK) — e.g. `CIC_00031`
- `SEGMENTO` — `CORP` | `RETAIL`
- `COLATERAL_TIPO` — `HIPOTECA` | `PERSONAL` | `NINGUNA` | `FINANCIERO`
- Numeric: `PD_ESTIMADA`, `LGD_ESTIMADA`, `EAD`, `DPDS`,
  `STAGE_IFRS9`, `RATING_GRADO`, `PROVISION_PERIOD_MONTHS`,
  `VENTANA_OBSERVACION_YEARS`, `VENTANA_CALIBRACION_YEARS`,
  `LGD_FLOOR_APLICADO`, `MOC`, `CURE_FLAG`, `RWA`, `ECL`
- V3-only: `OR_EAD_TIT` (computed)

# Q2 2025 — Collateral-type code migration

**Effective date:** 2025-04-01  
**Affected versions:** V2 → V3  
**Owner:** Data Governance

## Summary

The collateral-type vocabulary has been harmonised across IRB, IFRS 9
and Pillar 3 reporting. One legacy code has been **renamed** in the
warehouse:

| V2 (legacy) | V3 (new)        | Notes                              |
|-------------|-----------------|------------------------------------|
| FINANCIERO  | COLATERAL_FIN   | rename FINANCIERO to COLATERAL_FIN |
| HIPOTECA    | HIPOTECA        | unchanged                          |
| PERSONAL    | PERSONAL        | unchanged                          |
| NINGUNA     | NINGUNA         | unchanged                          |

## Affected fields

- `COLATERAL_TIPO` — value `'FINANCIERO'` becomes `'COLATERAL_FIN'`.

## Rationale

The unified taxonomy is required by the FINREP 2025 reporting template
(EBA ITS 2024/02) and aligns the source-of-truth field naming across
data marts. The behavioural change is purely a **string label**; no
numeric parameter, threshold or floor is altered.

The change does **not** affect:

- `LGD_ESTIMADA`
- `LGD_FLOOR_APLICADO` (the SAS pipeline matches on `'FINANCIERO'` for
  no-floor logic; this branch will need a separate remediation if the
  pipeline is run on V3 without an upstream code-mapping shim).

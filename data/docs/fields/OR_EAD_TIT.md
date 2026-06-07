# Field OR_EAD_TIT

`OR_EAD_TIT` is the **EAD titulizado** — Exposure At Default for tranches
sold into a securitisation programme. It is derived from the base `EAD`
by applying a regulatory multiplier that depends on the customer
`SEGMENTO`.

## Definition

| Segmento | Multiplier | Source |
|----------|-----------:|--------|
| CORP     | 2.0        | EBA GL 2020/06 §3.4 |
| RETAIL   | 1.5        | EBA GL 2020/06 §3.7 |
| Other    | 1.0        | (fall-through, no multiplier) |

## Pipeline

This field is produced by the SAS data step `work.titulizado`, which
exists in V3 of the calibration pipeline only. In V2 the field is not
computed at all — downstream consumers fell back to the raw `EAD`.

## Common pitfalls

- Comparing V2 vs V3 directly is misleading because the field did not
  exist in V2. The agent should treat a V2 row's missing `OR_EAD_TIT`
  as "not applicable" and explain that the V3 multiplier was added by
  release 2025-Q1.

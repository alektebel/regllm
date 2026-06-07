# Q1 2025 — LGD update for residential mortgages

**Effective date:** 2025-02-15  
**Affected versions:** V2 → V3  
**Owner:** Recovery Modelling

## Summary

The cure-rate component of the residential-mortgage LGD model was
re-fitted on the extended 2018–2024 vintage. Higher post-default cures
in the new sample reduce best-estimate LGD for `COLATERAL_TIPO =
'HIPOTECA'` by ~5 %.

## Affected fields

- `LGD_ESTIMADA` — multiplied by **0.95** when `COLATERAL_TIPO = 'HIPOTECA'`.
  Floor at 0.30 (CRR Art. 154(3)) is **still enforced** by the SAS
  pipeline, so cycles with collateral-LGD already at the floor are
  unaffected.

## Rationale

The reduction is justified by the cure-rate evidence summarised in the
internal note `RM-2025-002`. The downturn-LGD margin remains unchanged
because the change is structural rather than cyclical.

Because the SAS pipeline applies the **regulatory LGD floor** *after*
the model estimate, only cycles whose recalibrated LGD remains above
30 % will see a downstream impact on `ECL`.

## Out of scope

- `LGD_ESTIMADA` for non-mortgage collateral types (`PERSONAL`,
  `FINANCIERO`, `NINGUNA`).
- `LGD_FLOOR_APLICADO` — the floor logic is unchanged.

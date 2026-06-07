# Q1 2025 — PD master scale recalibration

**Effective date:** 2025-01-31  
**Affected versions:** V2 → V3  
**Owner:** IRB Calibration Team

## Summary

Following the annual back-testing exercise, the PD master scale was
recalibrated for the upper rating grades. The new estimates raise the
through-the-cycle PD for low-default segments by ~15 % to align with
the EBA Default Definition GL guidance and the latest economic-cycle
adjustments.

## Affected fields

- `PD_ESTIMADA` — multiplied by **1.15** for `RATING_GRADO ∈ {1, 2}`.
- `MOC` — recomputed with the new tail-risk add-on (no field rename).

## Rationale

The change is justified by the increase in observed default rates over
the most recent observation window (2023–2024). The EBA template
requires the PD for top-tier obligors to remain conservative; back-test
binomial p-values for grades 1 and 2 fell below the 5 % threshold, so
the floor was lifted accordingly.

The recalibrated PD enters the standard ECL formula:

    ECL = PD_ESTIMADA × LGD_ESTIMADA × EAD

so any V2 → V3 movement in `ECL` for cycles with rating 1 or 2 is
expected and traceable to this section.

## Out of scope

This change does **not** modify:

- `LGD_ESTIMADA` (covered separately in `2025-q1-lgd-cure.md`)
- `EAD` (no scheduled adjustments)
- `STAGE_IFRS9` classification

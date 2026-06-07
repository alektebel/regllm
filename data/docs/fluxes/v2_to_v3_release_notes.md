# Release notes: V2 → V3 (2025-Q1)

- **PD master-scale recalibration**: floor lifted to 0.05 %; ratings
  1–2 multiplied by 1.15. See `data/changelog/2025-q1-pd-recalibration.md`.
- **CORP LGD floor**: tightened from 45 % to 50 %.
- **Securitisation EAD**: introduced `OR_EAD_TIT` derived from `EAD` × a
  segment-dependent multiplier (CORP × 2.0, RETAIL × 1.5, otherwise × 1.0).
- **Collateral codes**: `FINANCIERO` renamed to `COLATERAL_FIN` per
  `data/changelog/2025-q2-collateral-codes.md`.

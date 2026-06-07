# Field EAD

`EAD` (Exposure At Default) is the on-balance exposure used in IRB ECL
formulas: `ECL = PD × LGD × EAD`. It feeds both `ECL` and `RWA`.

## Source table

`mylib.ciclos_recuperacion` — column `EAD`, type NUMERIC (€).

## Notes

- V3 of the database refreshed `EAD` for retail mortgages with revised
  amortisation profiles (see `data/changelog/2025-q1-pd-recalibration.md`).
- The field is *input* to the SAS pipeline; it is never overwritten.

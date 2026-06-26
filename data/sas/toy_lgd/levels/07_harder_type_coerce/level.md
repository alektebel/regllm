# Level 07 — Harder: Hidden Type Coercion

## Scenario

The pipeline checks collateral types with string comparisons.
SAS is case-sensitive for character comparisons unless explicitly
handled. The COLATERAL_TIPO comes from a source that uses mixed
case, but the code assumes a specific case.

## Expected correct output

All HIPOTECA rows should have the floor applied regardless of
the case used in the source data: `HIPOTECA`, `Hipoteca`, `hipoteca`.

## Buggy output

Some HIPOTECA rows (with different case) don't match the comparison
and the floor is not applied.

## Tables needed

- CICLOS only

## Hint

Look at the COLATERAL_TIPO values in the input data. Are they all
the same case? Does the comparison handle case variation?

# Field: PROVISION_PERIOD_MONTHS

**Table:** `mylib.ciclos_recuperacion`  
**Type:** INTEGER  
**Regulatory source:** Circular 6/2016 BdE, Artículo 12

## Definition

Number of whole calendar months elapsed since the cycle first entered default status, up to the evaluation date or cycle closure date (whichever is earlier).

This field is the primary control variable for regulatory provision requirements. It determines whether the minimum provision period mandated by the regulation has been satisfied for a given cycle.

## Regulatory minimum thresholds

The minimum value of `PROVISION_PERIOD_MONTHS` required before provision release is permitted (see `CURE_FLAG`) depends on `SEGMENTO` and the credit cycle phase (derived from `DPDS` per Article 8):

| SEGMENTO | Expansion phase | Contraction phase | Crisis phase |
|---|---|---|---|
| CORP | 12 | 18 | 24 |
| RETAIL | 24 | 30 | 36 |
| MORTGAGE | 36 | 42 | 48 |

Additional +6 months surcharge applies for CORP with `COLATERAL_TIPO` = NINGUNA.  
Additional +6 months surcharge applies when `STAGE_IFRS9` = 3 in any segment.

## Pipeline usage

`PROVISION_PERIOD_MONTHS` is read by the `lgd_pipeline` at the stage `lgd_con_suelos` to:
1. Verify minimum period compliance before allowing `LGD_FLOOR_APLICADO` reduction.
2. Flag non-compliant cycles in `NO_CONFORMES`.
3. Gate the provision release schedule defined in Article 23.

## Related fields

| Field | Relationship |
|---|---|
| `CURE_FLAG` | Set to 1 only after PROVISION_PERIOD_MONTHS ≥ regulatory minimum |
| `STAGE_IFRS9` | Affects minimum PROVISION_PERIOD_MONTHS; STAGE 3 adds +6 months |
| `DPDS` | Determines cycle phase which sets the base minimum |
| `SEGMENTO` | Determines which minimum threshold row applies |
| `COLATERAL_TIPO` | NINGUNA in CORP adds +6 months surcharge |
| `NO_CONFORMES` | Set when PROVISION_PERIOD_MONTHS < regulatory minimum at evaluation date |
| `ECL` | Provision amount computed over the active PROVISION_PERIOD_MONTHS |

# Level 06 — Solution

## Bug

Two related issues at the same point:

1. `GROUP BY COLATERAL_TIPO` instead of `GROUP BY SEGMENTO`
2. `MERGE BY COLATERAL_TIPO` instead of `MERGE BY SEGMENTO`

## Why it fails

The MoC is supposed to measure how much each cycle's LGD deviates
from its **segment** (business line) mean. Using COLATERAL_TIPO
instead means a CORP contract with NINGUNA collateral gets compared
to the NINGUNA mean (which is just CORP values anyway — correct by
accident). But a CORP contract with FINANCIERO collateral (CIC_009)
gets compared to the FINANCIERO mean of 0.40 instead of the CORP mean.

## Fix

```sas
PROC SQL;
    CREATE TABLE work.segment_means AS
    SELECT
        SEGMENTO,
        MEAN(LGD_ESTIMADA) AS LGD_MEDIA_SEGMENTO
    FROM work.floored
    GROUP BY SEGMENTO;
QUIT;

DATA work.ecl;
    MERGE work.floored (IN=a) work.segment_means (IN=b);
    BY SEGMENTO;
    ...
```

## Root cause

COLATERAL_TIPO and SEGMENTO are correlated (certain segments tend to
have certain collateral types), so the bug is subtle — it only
manifests when a segment contains multiple collateral types with
different LGD profiles. Most rows get the "right" mean by accident,
making the bug hard to spot without understanding the domain.

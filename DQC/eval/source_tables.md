# Source Tables — upstream of `CICLOS_CALIBRADOS`

The final reporting table is derived from three source tables. The eval
harness materialises them (`generate_db.build_clean_conn`) so checks can be
written **across tables** — referential integrity and reconciliation ("the
same attribute reported in two places must agree").

These tables are documented here, separately from `data_dictionary.md`, so
the field × article coverage matrix stays scoped to the reporting table.
Cross-table defects (D48–D53) live in `defect_catalog.py`.

---

## `CONTRATOS` — one row per contract (master)

| Field | Type | Description | Reconciles with |
|---|---|---|---|
| `ID_CONTRATO` | TEXT (PK) | Contract identifier. | `ciclos_calibrados.ID_CONTRATO` (referential parent — D48) |
| `ID_CLIENTE` | TEXT | Counterparty id. | `ciclos_calibrados.ID_CLIENTE` (D51) |
| `TIPO_PERSONA` | TEXT | `J` / `F`. | — |
| `SEGMENTO` | TEXT | Master exposure segment. | `ciclos_calibrados.SEGMENTO` (D50) |
| `PRODUCTO` | TEXT | Product family. | — |
| `ENTIDAD_ORIGEN` | TEXT | Originating entity. | — |
| `FECHA_ALTA_CONTRATO` | TEXT (date) | Origination date. | `ciclos_calibrados.FECHA_ALTA_CONTRATO` (D53) |

## `BASILEA_MENSUAL` — monthly exposure snapshots (authoritative EAD source)

| Field | Type | Description | Reconciles with |
|---|---|---|---|
| `ID_CONTRATO` | TEXT | Contract id (join key). | — |
| `MES_CICLO` | INTEGER | YYYYMM snapshot period (join key). | `ciclos_calibrados.MES_CICLO` |
| `OR_EAD` | REAL | Authoritative original EAD. | `ciclos_calibrados.OR_EAD` (reconciliation — D49) |
| `OR_DISPTO` | REAL | Drawn amount. | `ciclos_calibrados.OR_DISPTO` |
| `OR_DISBLE` | REAL | Undrawn amount. | `ciclos_calibrados.OR_DISBLE` |

Join key: `(ID_CONTRATO, MES_CICLO)`.

## `COLATERALES` — one row per collateral (secured exposures only)

| Field | Type | Description | Reconciles with |
|---|---|---|---|
| `ID_COLATERAL` | TEXT (PK) | Collateral id (`COL_<contrato>`). | — |
| `ID_CONTRATO` | TEXT | Contract id (join key). | — |
| `COLATERAL_TIPO` | TEXT | `HIPOTECA` / `PRENDA`. | `ciclos_calibrados.COLATERAL_TIPO` |
| `VALOR_COLATERAL_INICIAL` | REAL | Valuation at origination. | `ciclos_calibrados.VALOR_COLATERAL_INICIAL` (reconciliation — D52) |
| `HAIRCUT` | REAL | Regulatory haircut. | — |
| `FECHA_VALORACION` | TEXT (date) | Latest valuation date. | — |

## `EVOLUCION_MENSUAL` — monthly evolution panel (one row per cycle-month)

The temporal history of each cycle: how it evolves month by month after
default. This is the table that forces **panel / temporal** checks —
invariants that only exist *across months* and cannot be seen in a single
snapshot. Compliant with the PD/LGD guidelines by construction.

| Field | Type | Description | Temporal invariant |
|---|---|---|---|
| `ID_CONTR_CICLO_LGD` | TEXT | Cycle key (join back to the reporting table). | — |
| `ID_CONTRATO` | TEXT | Contract id. | — |
| `SEQ` | INTEGER | 1-based month index within the cycle. | contiguous |
| `MES_CICLO` | INTEGER | YYYYMM of this observation. | months contiguous, unique per cycle (D59, D60) |
| `DPDS` | INTEGER | Days past due this month. | non-decreasing unless `CURE_FLAG=1` (D63) |
| `STAGE_IFRS9` | INTEGER | IFRS-9 stage this month. | never improves unless `CURE_FLAG=1` (D62) |
| `SALDO_PENDIENTE` | REAL | Outstanding balance. | non-increasing |
| `RECUPERACION_ACUMULADA` | REAL | Cumulative recoveries to date. | non-decreasing (D58) |
| `COSTE_TOTAL_ACUMULADO` | REAL | Cumulative recovery costs. | non-decreasing |
| `PD_ESTIMADA` | REAL | PD this month. | `= 1.0` while `STAGE=3` — CRR Art. 178 (D61) |
| `CURE_FLAG` | INTEGER | 1 on the month the cycle cures. | gates every stage/DPD improvement |
| `ESTADO_CICLO` | TEXT | `ESTIMACION` / `CERRADO`. | — |

Panel defects (D58–D63) plant a fresh dirty *series* (all months share a
dirty cycle key) with exactly one temporal invariant broken; the oracle
self-joins on `SEQ = SEQ-1` (or groups by month) and returns the offending
cycle key, so attribution works the same as single-row defects.

---

## Cross-table check families the harness can now exercise

| Family | Meaning | Example defect | SQL shape |
|---|---|---|---|
| **Referential integrity** | every child has a parent | D48 (orphan cycle) | `LEFT JOIN … WHERE parent.key IS NULL` |
| **Reconciliation** | same attribute, two tables, must agree | D49 (EAD vs BASILEA), D50 (segment vs CONTRATOS), D51 (client), D52 (collateral value), D53 (origination date) | `JOIN … WHERE a.x <> b.x` |
| **Date interrelation** | multiple dates in a lifecycle order | D54–D57 (alta ≤ default ≤ adjudication ≤ close; sale ≥ adjudication; date/period agreement) | `WHERE date_a > date_b` |
| **Panel / temporal** | invariants across months of one cycle | D58–D63 (recovery monotone, contiguous months, unique month, PD=1 in default, stage/DPD only improve on cure) | `JOIN … ON SEQ = SEQ-1` / `GROUP BY … HAVING` |
| **Weird cross-domain** | business-rule combinations that make no sense | D64 (collateral on a credit card), D65 (mortgage loan with no mortgage), D66 (foreclosure on an unsecured exposure) | `WHERE PRODUCTO = … AND COLATERAL_TIPO …` |

These are the checks a single-table snapshot schema **cannot** force a model
to write — they need a JOIN, multi-date reasoning, a window over months, or
cross-domain business knowledge, and a lazy `col IS NULL` generator scores
zero on all of them.

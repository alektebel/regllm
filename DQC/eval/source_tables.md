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

---

## Cross-table check families the harness can now exercise

| Family | Meaning | Example defect | SQL shape |
|---|---|---|---|
| **Referential integrity** | every child has a parent | D48 (orphan cycle) | `LEFT JOIN … WHERE parent.key IS NULL` |
| **Reconciliation** | same attribute, two tables, must agree | D49 (EAD vs BASILEA), D50 (segment vs CONTRATOS), D51 (client), D52 (collateral value), D53 (origination date) | `JOIN … WHERE a.x <> b.x` |
| **Date interrelation** | multiple dates in a lifecycle order | D54–D57 (alta ≤ default ≤ adjudication ≤ close; sale ≥ adjudication; date/period agreement) | `WHERE date_a > date_b` |

These are the checks a single-table schema **cannot** force a model to
write — they need a JOIN or multi-date reasoning, and a lazy `col IS NULL`
generator scores zero on them.

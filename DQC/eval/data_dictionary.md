# Data Dictionary — `CICLOS_CALIBRADOS`

> The three **source tables** this reporting table is derived from
> (`CONTRATOS`, `BASILEA_MENSUAL`, `COLATERALES`) — and the cross-table
> reconciliation checks between them — are documented in
> [`source_tables.md`](source_tables.md).

Final output table of the 7-layer PD & LGD estimation pipeline
(`sas/ciclos_calibrados_pipeline.sas`). Every field below has a lineage of
**≥ 7 hops** from the raw sources (`CICLOS`, `CONTRATOS`, `BASILEA_MENSUAL`,
`COLATERALES`).

**Conventions**
- *Layer* = the SAS DATA/SQL step that produces (or last mutates) the field.
  `L0`=source, `L1`=staging, `L2`=fusion, `L3`=BASILEA join, `L4`=PD,
  `L5`=LGD, `L6`=EAD, `L7`=ECL/RWA/Stage (final).
- *Type* is the SQLite affinity used by `generate_db.py` (REAL/INTEGER/TEXT).
- *Null* = whether the clean DB may contain NULL for the column.
- *Reg ref* = the regulatory anchor cited by the DQC system prompt.

---

## 1. Identity & segmentation

| Field | Type | Layer | Null | Description | Reg ref |
|---|---|---|---|---|---|
| `ID_CONTR_CICLO_LGD` | TEXT (PK) | L0 | no | Recovery-cycle key `{ID_CONTRATO}_{YYYYMM}`. Primary key of the table. | — |
| `ID_CONTRATO` | TEXT | L0 | no | Contract identifier (FK to `CONTRATOS`). | — |
| `ID_CLIENTE` | TEXT | L0 | no | Counterparty identifier (FK to customer master). | BCBS 239 P3 |
| `TIPO_PERSONA` | TEXT | L0 | no | Counterparty legal nature: `J` (corporate/SME) / `F` (retail). | CRR Art. 147 |
| `MES_DEFAULT` | INTEGER | L0 | no | YYYYMM month the default event opened the cycle (= `MES_CICLO` at origination). | CRR Art. 178 |
| `MES_CIERRE_CICLO` | INTEGER | L0 | yes | YYYYMM closure month; only for `ESTADO_CICLO='CERRADO'`; must be `>= MES_DEFAULT`. | EBA GL 2017/16 §101 |
| `FECHA_ALTA_CONTRATO` | TEXT (date) | L0 | no | Contract origination date (ISO). Must equal `contratos.FECHA_ALTA_CONTRATO` and precede `FECHA_DEFAULT`. | BCBS 239 P3 |
| `FECHA_DEFAULT` | TEXT (date) | L0 | no | Default event date (ISO); its month must equal `MES_DEFAULT`. Opens the cycle. | CRR Art. 178 |
| `FECHA_CIERRE_CICLO` | TEXT (date) | L0 | yes | Cycle closure date (ISO); `>= FECHA_DEFAULT`; NULL when open. Ordering checked by D55. | — |
| `FECHA_ADJUDICACION` | TEXT (date) | L0 | yes | Foreclosure/adjudication date (ISO); must fall within `[FECHA_DEFAULT, FECHA_CIERRE_CICLO]`. Ordering checked by D55. | — |
| `FECHA_VENTA_COLATERAL` | TEXT (date) | L0 | yes | Collateral sale date (ISO); `>= FECHA_ADJUDICACION`. Ordering checked by D56. | — |
| `DIVISA` | TEXT | L0 | no | Exposure currency: `EUR`/`USD`/`GBP` (ISO 4217). | BCBS 239 P3 |
| `TIPO_CAMBIO` | REAL | L0 | no | FX rate to EUR at the cycle month (`1.0` for EUR). | — |
| `ID_FUSION_FINAL` | TEXT | L2 | yes | Fusion-group id; populated only when `SW_FUSION=1`. Non-unique in BASILEA (duplication hazard). | — |
| `SW_FUSION` | INTEGER | L2 | no | Fusion flag (0/1). Drives whether BASILEA lookup keys on `ID_FUSION_FINAL` instead of `ID_CONTRATO`. | — |
| `SEGMENTO` | TEXT | L0 | no | Exposure segment: `CORP`, `SME`, `RETAIL_HIP`, `RETAIL_CONS`. Determines PD floor, LGD floor, maturity. | CRR Art. 147 |
| `CALIBRATION_SEGMENT` | TEXT | L0 | no | Finer calibration bucket (e.g. `CORP_MED`, `RETAIL_HIP`). | EBA GL 2017/16 §73 |
| `PRODUCTO` | TEXT | L0 | no | Product family: `PRESTAMO`, `LINEA`, `TARJETA`. | — |
| `MES_CICLO` | INTEGER | L0 | no | Cycle period in YYYYMM form; the BASILEA join key (`= ID_FCH_DATOS`). | — |
| `ENTIDAD_ORIGEN` | TEXT | L0 | no | Originating entity (`BANCO_A`/`BANCO_B`/`BANCO_C`) — multi-entity lineage after fusions. | — |

## 2. Exposure (from BASILEA)

| Field | Type | Layer | Null | Description | Reg ref |
|---|---|---|---|---|---|
| `OR_EAD` | REAL | L3 | yes | Original EAD from BASILEA_MENSUAL (deduplicated for fusion groups). | CRR Art. 166 |
| `OR_DISPTO` | REAL | L3 | yes | Drawn amount (on-balance). Becomes `EAD_BALANCE`. | CRR Art. 166 |
| `OR_DISBLE` | REAL | L3 | yes | Undrawn amount (off-balance). Multiplied by CCF. | CRR Art. 166.8 |
| `SALDO_PENDIENTE` | REAL | L0 | yes | Outstanding balance; fallback for `EAD_BALANCE` when `OR_DISPTO` missing. | — |
| `EAD` | REAL | L6 | no | Canonical EAD alias = `EAD_TOTAL` (kept for `mylib.ciclos_recuperacion` compat). | CRR Art. 166 |
| `EAD_BALANCE` | REAL | L6 | no | On-balance EAD = `COALESCE(OR_DISPTO, SALDO_PENDIENTE)`. | CRR Art. 166 |
| `CCF_ESTIMADO` | REAL | L6 | no | Credit Conversion Factor applied to undrawn (portfolio value 0.75). | CRR Art. 182 |
| `EAD_FUERA_BALANCE` | REAL | L6 | no | Off-balance EAD = `CCF_ESTIMADO * OR_DISBLE`. | CRR Art. 166.8 |
| `EAD_TOTAL` | REAL | L6 | no | `EAD_BALANCE + EAD_FUERA_BALANCE`. Invariant; the ECL/RWA exposure base. | CRR Art. 166 |
| `EAD_TOTAL_EUR` | REAL | L6 | no | EUR-converted exposure = `EAD_TOTAL * TIPO_CAMBIO`. Invariant vs FX. | BCBS 239 P3 |

## 3. Collateral

| Field | Type | Layer | Null | Description | Reg ref |
|---|---|---|---|---|---|
| `COLATERAL_TIPO` | TEXT | L0 | no | `HIPOTECA` / `PRENDA` / `NINGUNA`. Drives `LGD_SUELO` (30% / 10% / 0%). | CRR Art. 161.1 |
| `VALOR_COLATERAL_INICIAL` | REAL | L3 | yes | Collateral valuation at origination. | — |
| `VALOR_COLATERAL` | REAL | L3 | yes | Current (haircut-adjusted) collateral value. | CRR Part 3 T. II Ch.4 |
| `HAIRCUT` | REAL | L3 | yes | Regulatory haircut applied to collateral (e.g. 0.04 for mortgage). | CRR Art. 161.4 |
| `MES_VALORACION_COLATERAL` | INTEGER | L3 | yes | YYYYMM of the latest collateral valuation; secured exposures must be revalued within 36 months of `MES_CICLO`. NULL when unsecured. | CRR Art. 208.3 |
| `LTV` | REAL | L6 | yes | Loan-to-value = `EAD_BALANCE / VALOR_COLATERAL` (0 when unsecured). | — |

## 4. PD (Probability of Default)

| Field | Type | Layer | Null | Description | Reg ref |
|---|---|---|---|---|---|
| `RATING_GRADO` | INTEGER | L0 | no | Internal masterscale grade (1=best). | EBA GL 2017/16 §73 |
| `PD_ESTIMADA` | REAL | L0 | yes | Raw model PD in [0,1] (per grade × segment). | CRR Art. 159 |
| `PD_SUELO` | REAL | L4 | no | Regulatory PD floor: 0.0005 for RETAIL_HIP, else 0.0003. | CRR Art. 160.1 |
| `PD_FINAL` | REAL | L4 | no | `MAX(PD_ESTIMADA, PD_SUELO)` — the corrective floor. Drives ECL & K_IRB. | CRR Art. 160.1 |
| `PD_DOWNTURN` | REAL | L4 | no | `MIN(1, 1.5 * PD_ESTIMADA)` — stressed PD for downturn sensitivity. | CRR Art. 181.1.b |

## 5. LGD (Loss Given Default)

| Field | Type | Layer | Null | Description | Reg ref |
|---|---|---|---|---|---|
| `LGD_ESTIMADA` | REAL | L5 | no | Model LGD (missing imputed to 0.45 segment central tendency). | CRR Art. 159 |
| `LGD_REALIZADA` | REAL | L0 | yes | Observed ex-post LGD = `1 - (recup - costes)/EAD`; for backtesting. | EBA GL 2017/16 §135 |
| `LGD_SUELO` | REAL | L5 | no | Collateral-driven floor: HIPOTECA 0.30, CORP 0.45, else 0. | CRR Art. 161.1 |
| `MOC_CAT_A` | REAL | L5 | no | MoC category A — identified data & methodological deficiencies. | EBA GL 2017/16 §43-44 |
| `MOC_CAT_B` | REAL | L5 | no | MoC category B — representativeness of historical data. | EBA GL 2017/16 §43-44 |
| `MOC_CAT_C` | REAL | L5 | no | MoC category C — general estimation error. | EBA GL 2017/16 §43-44 |
| `MOC` | REAL | L5 | no | Margin of Conservatism = `MOC_CAT_A + MOC_CAT_B + MOC_CAT_C` (= `0.05 * LGD_ESTIMADA` in the toy calibration). Invariant vs its categories. | EBA GL 2017/16 §43-44, §50 |
| `LGD_CON_MOC` | REAL | L5 | no | `LGD_ESTIMADA + MOC` (post-conservatism LGD). | EBA GL 2017/16 §50 |
| `LGD_FINAL` | REAL | L5 | no | `MAX(LGD_CON_MOC, LGD_SUELO)` — drives ECL & RWA. | CRR Art. 161.1 |
| `LGD_DOWNTURN` | REAL | L5 | no | Downturn LGD = `MIN(1, 1.15 * LGD_ESTIMADA)`; must never undercut the long-run average LGD. | CRR Art. 181.1(b) / EBA GL 2017/16 §345 |

## 5b. Calibration governance

| Field | Type | Layer | Null | Description | Reg ref |
|---|---|---|---|---|---|
| `VENTANA_OBSERVACION_YEARS` | INTEGER | L0 | no | Length (years) of the historical observation window backing the estimate. | EBA GL 2017/16 §6.3.2.1 |
| `FLAG_NC` | INTEGER | L0 | no | Non-conformity flag: must be 1 whenever `VENTANA_OBSERVACION_YEARS < 5`. | EBA GL 2017/16 §6.3.2.1 |

## 6. ECL / RWA / Stage

| Field | Type | Layer | Null | Description | Reg ref |
|---|---|---|---|---|---|
| `K_IRB` | REAL | L7 | no | Capital ratio = `SQRT(PD_FINAL)*0.06 + PD_FINAL*0.5`. | CRR Art. 153 |
| `M_VENCIMIENTO` | REAL | L7 | no | Effective maturity: 2.5 retail, `[1,5]` corp. | CRR Art. 162 |
| `RWA` | REAL | L7 | no | Risk-weighted assets = `EAD_TOTAL * LGD_FINAL * 12.5 * K_IRB`. | CRR Art. 153 |
| `ECL` | REAL | L7 | no | **Expected credit loss = `PD_FINAL * LGD_FINAL * EAD_TOTAL`** (canonical). | IFRS 9 / CRR Art. 158 |
| `PROVISION` | REAL | L7 | no | Accounting provision = `ECL`. | IFRS 9 |
| `STAGE_IFRS9` | INTEGER | L7 | no | IFRS-9 stage 1/2/3. Backstop: `DPDS≥30 & stage=1 → stage 2`. | IFRS 9 B5.5.12 |

## 7. Lifecycle & recovery (operational coherence fields)

| Field | Type | Layer | Null | Description | Reg ref |
|---|---|---|---|---|---|
| `DPDS` | INTEGER | L0 | no | Days past due at observation. `≥30` triggers stage 2; `≥90` default. | CRR Art. 178.1(b) |
| `CURE_FLAG` | INTEGER | L0 | no | 1 if the cycle cured (recovered from default). | — |
| `ESTADO_CICLO` | TEXT | L0 | no | `ESTIMACION` / `CERRADO`. Closed cycles require a `TERMINACION`. | — |
| `TERMINACION` | TEXT | L0 | yes | Exit cause when `ESTADO_CICLO='CERRADO'` (`CURA`/`FALLIDO`/…). Coherence-required. | — |
| `CAUSA_DEFAULT` | TEXT | L0 | no | Default cause (`90_DIAS_VENCIDO`/`IMPROBABLE_PAGO`/…). | CRR Art. 178 |
| `ADJUDICACION_FLAG` | TEXT | L0 | no | `'1'`/`'0'` — whether a foreclosure/adjudicación occurred. | — |
| `ADJUDICACION_TIPO` | TEXT | L0 | yes | Foreclosure type (`SUBASTA`/`DACCION`/…). Required iff value > 0. | — |
| `ADJUDICACION_VALOR` | REAL | L0 | no | Foreclosure value. `>0` requires flag `'1'` and a type. | — |
| `RECUPERACION_ACUMULADA` | REAL | L0 | no | Cumulative recovered cash. With costs, must not exceed 1.5 × EAD. | EBA GL 2017/16 §135 |
| `COSTE_TOTAL_ACUMULADO` | REAL | L0 | no | Cumulative recovery costs. | — |
| `TASA_DESCUENTO` | REAL | L0 | no | Discount rate applied to recovery cashflows. | EBA GL 2019/03 §85 |
| `TIPO_INTERES_ORIGINAL` | REAL | L0 | yes | Original contractual interest rate. | — |
| `TIPO_INTERES_ACTUAL` | REAL | L0 | yes | Current contractual interest rate. | — |
| `INTERESES_ACUMULADOS` | REAL | L0 | yes | Accrued interest capitalised into exposure. | — |

---

## Field-to-layer lineage map

```
L0 raw sources ── CICLOS, CONTRATOS, BASILEA_MENSUAL, COLATERALES
   │
L1 staging      ── type coercion + COALESCE fixes  (PD/LGD/EAD coerced)
   │
L2 fusion       ── MERGE CONTRATOS  → SW_FUSION, ID_FUSION_FINAL
   │
L3 BASILEA join ── OR_EAD, OR_DISPTO, OR_DISBLE, VALOR_COLATERAL  (dedup fix)
   │
L4 PD           ── PD_SUELO, PD_FINAL, PD_DOWNTURN
   │
L5 LGD          ── LGD_SUELO, MOC, LGD_CON_MOC, LGD_FINAL
   │
L6 EAD/CCF      ── EAD_BALANCE, EAD_FUERA_BALANCE, EAD_TOTAL, EAD
   │
L7 ECL/RWA      ── K_IRB, M_VENCIMIENTO, RWA, ECL, PROVISION, STAGE_IFRS9
```

The deepest fields (`ECL`, `RWA`) read inputs from **every** upstream layer
(L0 keys → L1 coercion → L2 fusion → L3 exposure → L4 PD → L5 LGD → L6 EAD → L7
formula), satisfying the **≥ 7-layer depth** requirement.

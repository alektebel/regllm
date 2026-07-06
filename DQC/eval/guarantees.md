# Stress Database & Eval Harness — why they guarantee good DQCs

This document explains the *rationale*: what the stress database and the eval
harness are, and why scoring well on them is a meaningful guarantee that a DQC
is actually good. For the operational how-to, see `README.md`.

---

## 1. The problem we are solving

A **DQC** (data-quality check) is a SQL predicate the DQC agent emits to detect
incoherent rows in the IRB recovery-cycles warehouse. The hard question is
not "does this SQL look reasonable?" — it is:

> *If this DQC fires, is there really a data-quality defect? And if a real
> defect exists, will this DQC actually catch it?*

On real production data you cannot answer that, because you do not know which
rows are genuinely wrong — there is no ground truth. So the only way to
evaluate a DQC rigorously is to **manufacture a database where the truth is
known by construction**: clean when it should be clean, and dirty in precise,
pre-specified ways. That is what the stress database is.

---

## 2. The stress database (`CICLOS_CALIBRADOS`)

### What it is
A complete PD & LGD estimation built from the BASILEA table in **7 layers**
(`sas/ciclos_calibrados_pipeline.sas`):

```
L0 BASILEA/CONTRATOS/CICLOS/COLATERALES
   → L1 staging coercion
   → L2 fusion enrichment
   → L3 BASILEA OR_EAD join (dedup fix)
   → L4 PD calibration + floors
   → L5 LGD downturn + floors + MoC
   → L6 EAD / CCF
   → L7 ECL + RWA + IFRS-9 stage
```

Every field in the final table has a lineage of **>= 7 hops**, and the fields
are densely interrelated (ECL reads PD, LGD, EAD; RWA reads EAD, LGD, K_IRB;
K_IRB reads PD; etc.). This density is deliberate: it creates many genuine
cross-field invariants a DQC could verify.

### Why it is a *stress* database
Three properties make it a stress test rather than a toy:

1. **Corrective fixes at every layer.** The pipeline applies the real fixes a
   production calibration would (regulatory floors, COALESCE imputation,
   fusion de-duplication, type coercion, caps). So the *clean* state is the
   correct end-state of a non-trivial pipeline — not a trivially clean table.
2. **Planted defects, not random noise.** 26 specific incoherences (D01–D25
   + 2 decoys) are documented in the SAS and materialized by `generate_db.py`.
   Each is a precise, named violation of one invariant.
3. **Many fields, different fixes, complex interrelations** — exactly the
   situation in which shallow checks fail.

### Clean vs trap — the construction that gives ground truth
- **Clean DB**: base attributes are drawn consistently and **every computed
  field is derived by re-applying the pipeline formulas**, so *every* catalog
  oracle returns 0 rows by construction. The self-test proves this (26/26).
- **Trap DB** (one per defect): the clean DB **plus one appended row** mutated
  by the defect's `mutate(row)` to break exactly that invariant. A fresh PK
  means any non-empty query result is unambiguously attributable to the
  planted defect.

This is **mutation testing**: the truth is manufactured, so "did the DQC catch
the defect?" is a deterministic fact, not an opinion.

---

## 3. The eval harness

### The defect catalog (`defect_catalog.py`)
26 defects across **8 data-quality dimensions** (DAMA / ISO 8000 / BCBS 239):

| dimension | defects | example invariant |
|---|---|---|
| consistency | D01–D09, D11–D13, D15, D16 | `ECL = PD_FINAL * LGD_FINAL * EAD_TOTAL` |
| validity | D19, D20 (+DA, DB decoys) | `SEGMENTO IN (...)` |
| conformity | D14, D24 | `SEGMENTO='RETAIL_HIP' ⇒ COLATERAL_TIPO='HIPOTECA'` |
| plausibility | D10, D25 | `RATING>=14 ⇒ PD>=0.01` (monotonicity) |
| completeness | D18 | `PD_ESTIMADA IS NOT NULL` on active cycles |
| timeliness | D21 | `MES_CICLO <= reference period` |
| uniqueness | D17 | no surviving BASILEA duplicates |
| accuracy | D22 | `EAD_BALANCE ≈ OR_DISPTO` (vs source) |

Each defect is an `(oracle_sql, mutate)` pair — the oracle is the reference
check, the mutator plants it.

### The 6-component verifiable reward (no LLM judge, no human)

| component | weight | what it proves |
|---|---:|---|
| `r_parse` | .10 | the SQL parses (`EXPLAIN` succeeds) |
| `r_template` | .10 | has SELECT/FROM/WHERE + the right table |
| `r_coherence` | .15 | WHERE combines >= 2 columns — the defining property of a coherence DQC |
| `r_clean_zero` | .15 | returns 0 rows on clean data (**no false positives**) |
| `r_catches` | .20 | returns >= 1 row on the mutated trap (**sensitivity**) |
| `r_specificity` | .30 | does NOT fire on unrelated traps (**anti-gaming**) |

### Aggregate metrics
- **Useful precision** = fraction of checks that are clean *and* catch >= 1
  coherence defect. A useless `WHERE 1=0` is excluded.
- **Specific recall** = coherence defects caught via *plurality-attributed*
  checks. A single broad OR-query is credited to only one dimension, so it
  cannot fake coverage of dimensions whose traps it fires on only by cascade.
- **Per-dimension recall** — any dimension below 50% is flagged `DEFICIENT`,
  which is the signal that the agent/model writes no adequate checks there.
- **`over_broad`** count flags checks firing on >= 3 dimensions.

---

## 4. Why this *guarantees* good DQCs

Define a **good DQC** as one that (a) targets a real cross-field invariant,
(b) fires when that invariant is broken, (c) stays silent on clean data,
(d) is narrow and attributable rather than a catch-all, and (e) is grounded
in a regulatory/business rule. The harness measures (a)–(d) directly and the
catalog encodes (e). Concretely:

1. **Ground truth is known by construction.** Clean = 0 violations for every
   oracle (self-tested); trap = exactly one planted violation. "Caught" and
   "clean" are unambiguous facts, not judgments. The metric cannot be fooled
   about whether a defect exists.

2. **Sensitivity is proven, not claimed (`r_catches`).** A DQC scores 1 only
   if it returns the mutated violating row. A check that *sounds* right but
   does not fire on the planted defect scores 0. This forces checks to be
   functionally correct on the invariant they claim to verify.

3. **No false positives (`r_clean_zero`).** A check that fires on clean data
   scores 0. Tautologies and over-broad predicates are penalized.

4. **Anti-gaming (`r_specificity` + plurality attribution).** Without this,
   one degenerate OR-query would rack up high recall across dimensions it has
   no logic for. We verified this empirically: a hand-crafted über-query that
   used to report **58% recall, 100% precision, and false "accuracy 100% /
   uniqueness 100%"** drops, after the fix, to **0% accuracy, 0% uniqueness,
   50% precision**, and is flagged `over_broad`. The guarantee is that high
   recall now requires many *narrow, specific* checks — which is what "good"
   means.

5. **Useless-clean inflation is blocked (useful precision).** `WHERE 1=0` is
   clean but catches nothing; it no longer counts toward precision.

6. **Breadth is enforced (dimension coverage).** A model that only emits
   consistency checks will score 0% on completeness/timeliness/accuracy/
   uniqueness/conformity and be flagged deficient. High overall recall
   requires the agent to cover *all* DQ dimensions.

7. **Reproducible and CI-able.** Deterministic seed, fully programmatic, no
   LLM judge. Scores are comparable across model versions and over time, so
   "did the new LoRA regress timeliness checks?" is an answerable question.

In short: **a DQC that scores high has provably demonstrated, on a database
whose truth is known, that it catches a specific defect and stays silent
otherwise.** That is exactly the content of "this DQC is good".

---

## 5. What it does *not* guarantee (honest limits)

The guarantee is bounded, and overstating it would be misleading:

- **Catalog-bounded.** Only the 26 modeled defect types are tested. A novel
  failure mode not in the catalog is not covered. Breadth of the catalog is
  the ceiling on what "good" can mean here.
- **Synthetic distributions.** The clean DB uses simplified generators
  (constant CCF, MoC = 5%, banded PD). Real warehouse noise and edge cases
  differ. A check that passes here may still misbehave on production data.
- **Single-table denormalization.** Genuine cross-table agent SQL (`JOIN
  contratos`) cannot be scored — only `ciclos_calibrados` exists. D14/D17/D24
  are cross-table *concepts* modeled as in-row symptoms until a multi-table
  mode is added.
- **Unavoidable formula coupling.** Mutating a shared input (e.g. EAD)
  legitimately breaks several formula oracles at once, so perfect per-defect
  isolation is impossible; specificity mitigates but cannot fully solve this.
- **Not a production sign-off.** A high reward is *necessary evidence* that a
  DQC works; it does not replace the human validation gate or backtesting on
  the real warehouse.

So the precise claim is: **scoring high on this harness is a verifiable
guarantee that a DQC detects its target defect with no false positives on the
stress database — and the per-dimension report tells you exactly which DQ
areas the model still cannot handle.**

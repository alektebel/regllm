# DQC Eval Harness

Stress-test the **DQC agent** (`api/routers/dqc.py` + `training/dq/`) against a
complete 7-layer PD & LGD estimation database built from the BASILEA table, and
**detect where the model is deficient** — broken down by industry data-quality
dimensions (DAMA / ISO 8000 / BCBS 239) *and by regulatory article*.

```
SAS pipeline (7 layers) ──► CICLOS_CALIBRADOS schema ──► data_dictionary.md
        │                          │                          │
        └── corrective fixes       │            coverage_matrix.py
                                   │        (field × article certification)
                     defect_catalog.py (48 ground-truth defects,
                      8 DQ dims, regulation_ref per defect)
                                   │
        generate_db.py ──► clean SQLite + k-row traps + MIXED DB
                                   │
                             eval_harness.py
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
         --selftest          --sql FILE           --agent URL
       (oracle audit +   (coverage on the      (targeted per-defect,
        overlap matrix)     mixed DB)          k-row catch fraction)
                                   │
        per-dimension + per-article recall  →  DEFICIENT / UNCOVERED flags
```

## Files

| Path | Role |
|---|---|
| `sas/ciclos_calibrados_pipeline.sas` | 7-layer PD+LGD pipeline (L0 sources → L7 ECL/RWA), corrective fixes, documents planted defects D01–D25. |
| `data_dictionary.md` | Final-table schema (66 fields) + per-field description, lineage layer, regulatory ref. **The single source of truth for the coverage matrix.** |
| `defect_catalog.py` | 48 ground-truth defects across 8 DQ dimensions, each with an `oracle_sql`, a `mutate(row)` and a `regulation_ref`. |
| `generate_db.py` | Clean SQLite (all invariants hold by construction), k-row traps per defect, and a **mixed DB** with every defect planted at once. |
| `eval_harness.py` | Self-test, mixed-DB coverage, agent-targeted scoring, per-dimension + per-article deficiency report, `--fail-under` CI gate. |
| `coverage_matrix.py` | Field × article coverage certification + GL applicability skeleton generator. |
| `../coverage/applicability.yaml` | EBA GL/2017/16 section → field applicability map (human-reviewed). |
| `example_checks.sql` | A sample check file to exercise coverage mode. |
| `data/` | Generated `clean.db`, `trap_<id>.db`, `mixed.db` (via `generate_db.py`). |

## The database

7 lineage layers (`L0` BASILEA/CONTRATOS/CICLOS/COLATERALES → `L7` ECL/RWA/
stage) so the deepest fields read inputs from every upstream layer. The schema
carries realistic production context: counterparty identity, cycle dates
(default / closure / collateral valuation in YYYYMM), multi-currency exposure
with EUR conversion, MoC broken into EBA GL §43-44 categories A/B/C, downturn
PD/LGD, and calibration-governance fields (observation window +
non-conformity flag). Every derived field satisfies its documented formula in
the clean DB **by construction**, so any oracle firing on `clean.db` is a bug.

## Trap protocol (mutation testing, hardened)

- **k planted rows per defect** (`TRAP_K = 3` default): each planted row is
  derived from a different random base, so sensitivity is sampled at k points
  of the violation space. `r_catches` is the *fraction* of planted rows a
  check detects — a check that memorises one row no longer scores full recall.
- **True PK duplicates**: the uniqueness defect (D33) is planted by
  re-inserting verbatim copies of existing rows, not by proxy symptoms.
- **Mixed DB**: `--sql` coverage plants *all 48 defects simultaneously*
  (production-like) and attributes hits by planted PK. This also yields the
  **confusion matrix**: a check firing on more than 3 distinct defects is
  flagged *overbroad* (tautology / reward-hacking smell), and checks fishing
  for trap PKs directly are caught the same way.
- **Oracle overlap report**: nested invariants legitimately overlap (a floor
  breach implies the max() was skipped); the self-test prints which oracles
  fire on other defects' planted rows so the overlap is visible, not silent.
- **Decoys**: single-column range checks (DA, DB) that must score
  `r_coherence = 0`; firing on them over-claims coherence.

## Run

```bash
# 1. Validate every oracle (0 rows on clean, ALL k rows on its trap)
python DQC/eval/eval_harness.py --selftest

# 2. Score a hand-written set of checks on the mixed DB
python DQC/eval/eval_harness.py --sql DQC/eval/example_checks.sql

# 3. Targeted eval against a running DQC agent (POST /dqc/generate)
python DQC/eval/eval_harness.py --agent http://localhost:8000/api

# 4. CI gate: fail if coherence-defect recall drops below 80%
python DQC/eval/eval_harness.py --sql checks.sql --fail-under 0.8

# 5. Field × article coverage certification (0 TODO cells required in CI)
python DQC/eval/coverage_matrix.py --fail-under 1.0

# 6. Regenerate the GL applicability skeleton for human review
python DQC/eval/coverage_matrix.py --emit-applicability

# 7. Materialise the databases to disk for ad-hoc inspection
python DQC/eval/generate_db.py --rows 2000

# JSON output for any mode
python DQC/eval/eval_harness.py --sql checks.sql --json
```

## Scoring (verifiable, no LLM judge)

Each candidate SQL gets the 5-component reward (mirrors
`training/dq/dq_reward.py`, weights sum to 1):

| Component | Weight | Meaning |
|---|---:|---|
| `r_parse` | 0.15 | SQLite `EXPLAIN` succeeds |
| `r_template` | 0.20 | SELECT + FROM + WHERE + correct table |
| `r_coherence` | 0.20 | WHERE references ≥ 2 columns (or a JOIN) |
| `r_clean_zero` | 0.20 | returns 0 rows on the clean DB (specificity) |
| `r_catches` | 0.25 | **fraction of the k planted rows caught** (sensitivity) |

Aggregate metrics:
- **Precision** = fraction of supplied checks that are valid + clean.
- **Recall** = fraction of coherence (non-decoy) defects caught by some clean check.
- **Per-dimension recall** — any dimension below 50% is flagged `← DEFICIENT`.
- **Per-article recall** — every defect carries a `regulation_ref`; articles
  with zero caught defects are flagged `← UNCOVERED`.
- **Decoy over-claim** and **overbroad checks** are counted as quality smells.

Attribution note: hits are attributed via the `ID_CONTR_CICLO_LGD` column, so
checks should project the primary key (the system prompt and all oracles do).

## Certifying "100% of the articles are checked"

Coverage is a **computable statement**, not an observation of LLM output:

1. `data_dictionary.md` attaches regulatory references to every field —
   the applicability map at field level.
2. `DQC/coverage/applicability.yaml` maps every GL/2017/16 *section* (from
   the 221 ingested paragraphs) to fields, or marks it not-applicable with a
   reason — each entry requires human sign-off (`review: approved`).
3. `coverage_matrix.py` cross-checks: every applicable (field × article) cell
   must be **covered** by a defect whose oracle demonstrably fires on a
   planted violation (`covered`), or at least whose field is exercised
   (`partial`). `--fail-under 1.0` gates CI on zero `todo` cells.
4. `eval_harness.py --sql/--agent` then measures how much of that certified
   surface the *generated* checks actually catch, per dimension and per
   article.

So "all articles checked" = applicability.yaml fully approved ∧ matrix has no
TODO ∧ harness recall = 100% on the target check set — three artifacts, all
versioned, all machine-verifiable.

## Design notes

- The clean DB is generated by drawing consistent base attributes and
  re-applying the pipeline formulas, so *every* oracle returns 0 rows by
  construction. Each defect's `mutate(row)` breaks exactly one invariant per
  planted row (fresh PK), so a non-empty result is unambiguously caused by
  that defect — the same mutation-testing contract as the RL reward.
- The harness is **self-contained**: `--selftest` and `--sql` need no LLM. The
  agent path is optional and degrades to the local scorer.
- Table aliases the agent emits (`mylib.ciclos_recuperacion`,
  `recuperatory_cycles`) are normalised to `ciclos_calibrados` before scoring.
- `tests/test_dqc_eval.py` regression-tests the harness itself (the catalog
  shape, trap protocol, attribution, matrix parsing) — the certifier is code
  too.

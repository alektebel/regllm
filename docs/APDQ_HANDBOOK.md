# APDQ Handbook

**Auditor-Parity Data Quality for Bank Reporting Databases — from the
proposal to the regulation to the running code.**

This is the consolidated reference. Each section summarizes and then
links the deeper artifact (all in this repository); the handbook is the
place to start reading and the map of everything else.

- Date: 2026-07-12 · Status: MVP implemented and certified (see §8)
- Companion documents:
  [`DATA_QUALITY_INDUSTRY_SOTA_2026.md`](DATA_QUALITY_INDUSTRY_SOTA_2026.md) ·
  [`DATA_QUALITY_SPAIN_NICHE.md`](DATA_QUALITY_SPAIN_NICHE.md) ·
  [`AUDITOR_PARITY_STANDARD.md`](AUDITOR_PARITY_STANDARD.md) ·
  [`MVP_ROADMAP.md`](MVP_ROADMAP.md) ·
  [`../apdq/README.md`](../apdq/README.md) ·
  [`BRANCHES.md`](BRANCHES.md)

---

## Table of contents

1. [The proposal](#1-the-proposal)
2. [Market context and economics](#2-market-context-and-economics)
3. [The regulatory foundations, in technical detail](#3-the-regulatory-foundations-in-technical-detail)
4. [The standard: auditor parity](#4-the-standard-auditor-parity)
5. [The defect taxonomy (normative)](#5-the-defect-taxonomy-normative)
6. [Implementation reference](#6-implementation-reference)
7. [The certification protocol](#7-the-certification-protocol)
8. [Worked proof: the ciclos_full certification](#8-worked-proof-the-ciclos_full-certification)
9. [Where the LLM sits](#9-where-the-llm-sits)
10. [Verification of the implementation itself](#10-verification-of-the-implementation-itself)
11. [Roadmap and declared limits](#11-roadmap-and-declared-limits)
12. [Glossary](#12-glossary)

---

## 1. The proposal

**Problem.** Banks compute regulatory numbers (provisions, capital,
risk parameters) from reporting databases. Wrong data means wrong
reported numbers, supervisory findings, capital add-ons and fines.
Banks defend against this with hand-written data-quality checks — and
nobody can answer the supervisor's two questions: *"how do you know
these checks work?"* and *"how do you know you have enough of them?"*

**Proposal.** An open standard plus a reference tool, APDQ
(Auditor-Parity Data Quality), that lets **any bank** certify — by
execution, not assertion — that its data-quality test suite detects
**every issue any auditor could detect from the data**. Three ideas
make this possible:

1. **Auditor parity** (§4): the achievable guarantee is not "catch
   everything wrong" but "catch everything detectable from the data,
   its documentation, and the regulation" — which is all an auditor
   has too. The remainder is declared, like an audit opinion's scope
   limitation.
2. **Proof by mutation** (§7): every check is validated by planting
   the defect it claims to catch into a synthetic twin of the schema
   and demanding it fires — and stays silent on clean data.
3. **Completeness by construction** (§4.2): "do we have enough
   checks?" becomes a computable walk over the schema's lineage graph,
   not an opinion.

**Deliverable per engagement**: a versioned, third-party-verifiable
**audit pack** — the artifact a bank hands to the ECB inspector, the
external auditor and internal validation, regenerable on every
regulation amendment or schema change.

**Why this repo**: the DQC eval harness (`DQC/eval/`) already invented
the certification mechanics (mutation testing against a
clean-by-construction database, verifiable scoring, no LLM judge) for
one schema; APDQ (`apdq/`) generalizes them to any schema via a
binding manifest. The SAS AST engine (`src/`) supplies the two assets
no competitor has: formula transcription out of legacy SAS estates and
the class-12 (semantic drift) oracle.

Full strategic write-up: [`AUDITOR_PARITY_STANDARD.md`](AUDITOR_PARITY_STANDARD.md).

## 2. Market context and economics

Condensed from [`DATA_QUALITY_INDUSTRY_SOTA_2026.md`](DATA_QUALITY_INDUSTRY_SOTA_2026.md)
(global) and [`DATA_QUALITY_SPAIN_NICHE.md`](DATA_QUALITY_SPAIN_NICHE.md)
(the target niche):

- The market splits into managed ML observability ($50K–$200K+/yr:
  Monte Carlo, Anomalo), open-source in-pipeline testing (dbt tests,
  Great Expectations, Soda — free plus engineering), and governance
  suites ($90K–$300K/yr: Collibra, Ataccama, Informatica). Regulated
  enterprises stack them: **$200K–$500K/yr plus 1–2 FTEs**.
- LLM-generated DQ rules are a mainstream vendor feature (Monte
  Carlo's monitor-recommendation agent reports ~60% acceptance), and
  the academic frontier converges on the same shape: *LLM proposes,
  code verifies, human approves*.
- **Nobody certifies coverage against a regulation at article level.**
  Industry practice is a Critical-Data-Element × quality-dimension
  matrix asserted by stewards. APDQ's requirement-register ×
  lineage-node certification, proven by execution, has no commercial
  counterpart — that is the differentiator.
- Gartner puts poor data quality at ~$12.9M/yr per organisation; 59%
  of organisations do not measure it.
- **Spain is the beachhead**: 10 significant institutions under ECB
  supervision (the IRB users), ~73 LSIs buying through consultancies,
  a supervisor (Banco de España) announcing "more intrusive"
  inspections, SAS-heavy risk estates, conservatism about data egress
  favouring the local-LLM/on-prem path, and consultancy-built DQ rule
  batteries at €200K–€500K per (recurring) engagement as the price
  anchor.

## 3. The regulatory foundations, in technical detail

What follows is the regulation stack the standard certifies against,
what each instrument technically demands of *data*, and how each maps
to APDQ artifacts. Paragraph/article references are the working anchors
used in this repo's artifacts; the requirement register (§6.5) pins the
actual normative text by hash, which is the authoritative link.

### 3.1 BCBS 239 and the ECB RDARR guide (the meta-layer)

BCBS 239 (*Principles for effective risk data aggregation and risk
reporting*, 2013) sets 14 principles; the data ones are **accuracy &
integrity (P3)** — largely automated aggregation, documented and
reconciled; **completeness (P4)** — all material risk data captured;
**timeliness (P5)**; **adaptability (P6)**. The ECB's **RDARR guide**
(May 2024) operationalises them for SSM banks: full data lineage at
**attribute level**, ownership, quality controls and monitoring across
the entire lifecycle, assessed in SREP from 2025 with escalation
threats. Only 2 of 31 G-SIBs are fully compliant per the last Basel
Committee assessment.

**Maps to**: the lineage-obligation walk (§6.6) *is* attribute-level
lineage with controls per node; the audit pack (§7) is the evidence
format; reconciliation oracles (class 7) are P3's "reconciled";
control totals (class 8) are P4's completeness.

### 3.2 EBA GL/2017/16 — PD & LGD estimation (the primary corpus)

The EBA *Guidelines on PD estimation, LGD estimation and the treatment
of defaulted exposures* (EBA/GL/2017/16, applicable since 2021) is the
richest single source of data-level obligations for IRB banks, and the
corpus this repo ingested first (in the Spanish official translation —
the language Spanish validation teams actually work in). The
obligations the implementation currently anchors to:

| Anchor | Obligation (as bound in this repo) | APDQ expression |
|---|---|---|
| §14–15 (4.2.1) | data used in estimation complete, accurate, consistent | classes 1, 2, 6, 7, 8 across the estimation datamart |
| §21 | exposures assigned to calibration segments coherently | segment domain (class 2) + segment/collateral/product constraints (class 5) |
| §43–44 | Margin of Conservatism quantified by deficiency **categories A/B/C**, MoC = their aggregation | `MOC = MOC_CAT_A + MOC_CAT_B + MOC_CAT_C` as a class-6 recomputation |
| §50 | minimum historical observation window; deviations flagged | `FLAG_NC = if(VENTANA < 5, 1, 0)` derivation |
| §101 | default entry/exit (cure) recorded consistently; cycle dates coherent | date orderings (class 9), cure/outcome derivations (class 6), panel stage logic (class 10/6) |
| §127 | collateral in LGD valued and documented | conditional collateral valuations (`when:` columns), class 5 constraints |
| §135 | realised LGD = economic loss over EAD, from discounted recoveries and costs | `LGD_REALIZADA` as a **conditionally derived** field (formula binds for closed cycles) — the exact case the `when:` feature exists for |
| §140 | EAD includes drawn plus converted off-balance amounts | `EAD_TOTAL = EAD_BALANCE + CCF·undrawn` recomputation chain |
| §161 | final parameters respect floors / are reviewed | `PD_FINAL = max(PD_ESTIMADA, PD_SUELO)` recomputations; grade↔PD monotonicity constraint |

The register (`apdq/examples/mini_ciclos/register.yaml`) holds these as
atomic rows with SHA-256 pins of the paragraph text; an amendment flips
the affected rows to `pending` automatically.

### 3.3 CRR (Regulation 575/2013) — the level-1 hooks

- **Art. 178** — definition of default: 90 days past due as a trigger;
  hence `STAGE/CAUSA_DEFAULT ↔ DPD` coherence rules (the repo derives
  `CAUSA_DEFAULT = '90_DIAS_VENCIDO'` where `DPDS ≥ 90`).
- **Art. 174/176** — model use and **data maintenance**: institutions
  must collect and store the data underpinning their models — the
  legal basis for "an unbound applicable concept is a finding".
- **Art. 181(1)(b)** — downturn LGD not less than the long-run
  average: `LGD_DOWNTURN` recomputation.
- **Art. 208(3)** — collateral monitoring/revaluation cadence for real
  estate; expressed in the example as a YYYYMM staleness constraint
  (documented as a granularity simplification in the crosswalk).

### 3.4 Banco de España Circular 4/2017 and Anejo IX (the Spanish layer)

Circular 4/2017 aligns Spanish credit institutions' accounting with
IFRS 9; **Anejo IX** is where risk, accounting and data collide:
classification into *riesgo normal / vigilancia especial / dudoso*
(operationally: IFRS 9 stages driven by arrears and qualitative
triggers), minimum coverage schedules by collateral type and vintage,
foreclosed-asset (*adjudicados*) lifecycle with valuation haircuts and
sale discounts, and cure/refinancing probation rules. It is amended
repeatedly (most recently Circular 1/2025) — which is precisely why the
register pins text hashes: **every amendment is an automated diff, not
a new consulting engagement**. The `ciclos_full` example already
carries the adjudicados lifecycle (foreclosure dates, types, values,
sale ordering) and stage↔DPD coherence; a full Anejo IX register is
roadmap item E8's first target.

### 3.5 Supervisory reporting: COREP/FINREP, AnaCredit, CIRBE

- **COREP/FINREP** (EBA ITS on supervisory reporting): datapoints
  defined in the EBA **DPM** with machine-readable **validation
  rules**, updated in quarterly packages; the NCA (Banco de España)
  rejects submissions failing them, forcing fix-and-resubmit cycles.
  APDQ can import these rules mechanically as a pre-submission check
  layer (roadmap E8) — catching the rejection *before* filing.
- **AnaCredit** (Regulation (EU) 2016/867) and **CIRBE** (BdE Circular
  1/2013): loan-by-loan monthly registers. The same exposure appears
  in the IRB datamart, AnaCredit, CIRBE and FINREP — a four-surface
  reconciliation problem, which is exactly the class-7/class-8 oracle
  family generalized across registers (roadmap E10 binds the real
  registers as surfaces).

### 3.6 BIRD and IReF (the canonical-model anchor)

The ECB's **BIRD** (Banks' Integrated Reporting Dictionary) publishes
an open input-layer logical data model with transformation and
validation rules; the **IReF** regulation (confirmed June 2026,
first reporting planned Q4 2029) will oblige EU banks to map to an
integrated reporting model. APDQ's concept vocabulary (§6.2) is
BIRD-anchored by design (`bird_hint` fields today, full input-layer
vocabulary as roadmap E1) so the per-bank binding manifest becomes a
by-product of work every EU bank must do before 2029 anyway. This is
the adoption-timing argument for the standard.

## 4. The standard: auditor parity

Full normative text: [`AUDITOR_PARITY_STANDARD.md`](AUDITOR_PARITY_STANDARD.md).
The two load-bearing ideas:

### 4.1 The parity claim

An auditor works from three inputs: the data, the bank's documentation
of the data (dictionary, derivation rules, lineage), and the
regulation. Any defect they detect must manifest as a contradiction
derivable from those inputs. Therefore the certifiable guarantee is:

> every defect detectable from **data ∪ documentation ∪ regulation**
> is detected — at full population, where an auditor samples.

The **declared residual** (printed in every audit pack): defects
requiring external evidence (a value consistently wrong at origin),
coordinated falsification preserving all internal invariants,
regulatory-interpretation and estimation judgment. No data-side actor,
human or machine, closes those from the database alone.

### 4.2 The completeness argument

Model the schema as its lineage DAG. Obligations per node type —
derived field: recomputation oracle from its documented formula
(auditors' *reperformance*); source field: validity + reconciliation
on every surface where the fact recurs + plausibility; table:
control totals against an independent surface (catches missing and
fabricated rows); structures: uniqueness, referential integrity,
temporal orderings, panel consistency. An **undocumented derivation is
itself a finding**, as is an applicable regulatory concept bound to no
column. Then, by induction over the DAG: any single-point corruption
either fires its own node's oracle or propagates into a descendant's
recomputation mismatch; whatever fires nothing is consistent on every
surface and every formula — i.e. inside the declared residual. The
check "does every node carry its obligations?" is **a program**
(`apdq/lineage.py`), re-runnable by any third party.

## 5. The defect taxonomy (normative)

Twelve classes, finite, phrased in the vocabulary auditors already use.
Classes 1–10 are deterministic (zero tolerance for false positives on
a clean twin) and certifiable; 11 is advisory by design; 12 is served
by code-diff analysis rather than row mutations.

| # | Class | Auditor's term | Oracle shape | Status in `apdq/` |
|---|---|---|---|---|
| 1 | missing value | completeness exception | `col IS NULL` (branch-guarded for conditional fields; PK-null is aggregate) | generic |
| 2 | domain violation | validity exception | range / enum / date-bound predicate | generic |
| 3 | duplicate key | occurrence | `GROUP BY pk HAVING COUNT(*)>1` on verbatim re-inserts | generic |
| 4 | broken reference | existence | anti-join to the declared parent | generic |
| 5 | intra-row incoherence | internal consistency | `NOT (declared constraint)` | generic (auto-violation search: singles + pairs; `plant:` hints beyond) |
| 6 | derivation error | reperformance mismatch | null-aware & text-aware recompute vs. documented formula, `when`-guarded for conditional derivations | generic |
| 7 | cross-surface mismatch | reconciliation difference | join to authoritative surface, tolerance or exact | generic |
| 8 | missing/fabricated rows | population completeness | count/sum ties to control-total surface (aggregate) | generic |
| 9 | temporal impossibility | cut-off / sequence | adjacent-pair violations over declared date chains | generic |
| 10 | panel inconsistency | period-on-period consistency | period gaps, duplicate periods, decreasing cumulatives per series | generic |
| 11 | distributional anomaly | analytical review exception | statistical monitors | advisory only — never counts toward certification |
| 12 | semantic drift | unauthorized definition change | SAS AST diff (`src/sas_logic_tree.py`) vs. the manifest's formulas | integration on roadmap (E5) |

Anti-gaming devices built into the harness: **specificity** (all
oracles silent on the clean twin), **k-point sensitivity** (each defect
planted at k distinct rows; catching one memorised row scores
partially), the **confusion matrix** with an **overbroad** flag
(an oracle firing across many unrelated defects is a tautology smell —
recomputation and temporal oracles are exempted where breadth is
legitimate propagation), and in the legacy harness, **decoys**.

## 6. Implementation reference

Package `apdq/` (Python, stdlib + PyYAML; SQLite as the certification
engine). One diagram:

```
regulation text ──► requirement register (atomic, SHA-256-pinned) ──┐
                                                                    │ gates
bank schema ──► binding manifest ──► lineage-obligation walk ───────┤
                    │                                               │
                    ├─► synthetic twin (clean by construction)      │
                    ├─► defect generators (mutations + oracles)     │
                    │         │                                     │
                    │   specificity (clean) + recall (k traps)      │
                    │         │                                     │
                    └────► certification ──► audit pack (JSON+HTML) ┘
```

### 6.1 The binding manifest (`apdq/manifest.py`)

The single per-bank artifact; YAML; human-reviewable without reading
any generated SQL. Field reference:

- **Table**: `name`, `primary_key`, `rows` (twin size),
  `foreign_keys` (class 4), `date_orderings` (class 9; loader enforces
  **monotone domains** along each chain — the twin sorts sampled
  values into compliance and non-monotone domains would let that sort
  push a value out of range), `constraints` (class 5: `id`, `expr`,
  optional `plant:` violating values, `regulation_refs`), `control`
  (class 8: surface + count/sum checks), `panel` (class 10:
  `series_key`, `period_column`, `periods`, `cumulative_columns`),
  `waivers` (signed reasons for absent obligations).
- **Column**: `concept` (must resolve in the vocabulary), `role`
  (`source` | `derived`), `domain` (`int|real|text|yyyymm|date`, min /
  max / `values` enum / `nullable` + `null_rate` / `unique`),
  `formula` (required for derived — *an undocumented derivation is
  refused at load*), **`when`** (conditional derivation: the formula
  binds only where the condition holds; elsewhere the value is free and
  sampled from the mandatory `domain` — this is how "realised LGD is
  formula-bound for closed cycles" is expressed), `reconcile`
  (class 7 surfaces; `join_column` must be the PK — surfaces are
  auto-mirrored per row; real cross-register joins are E10),
  `regulation_refs`, `waivers`.

Validation is deliberately unforgiving: unknown concepts, formulas
referencing unknown columns, circular derivations, non-monotone
ordering domains, non-PK reconcile joins and unsatisfiable constraint
setups all fail at load with actionable messages.

### 6.2 Concept vocabulary (`apdq/concepts/pd_lgd.yaml`)

~60 canonical PD/LGD concepts (bilingual EN/ES, `bird_hint` where a
BIRD input-layer counterpart exists). The manifest binds columns to
concepts; the register binds requirements to concepts; **an applicable
requirement whose concepts are unbound is detected as a gap** — the
absence of data an auditor would ask for. Swappable per
`concepts_file:`; full BIRD input layer is roadmap E1.

### 6.3 The formula language (`apdq/expr.py`)

Small by design — every formula must be evaluable in Python (to build
the twin) *and* compilable to SQL (to build the oracle), from one AST,
with the two backends tested to agree.

```
expr    := or ;   or := and ('or' and)* ;   and := not ('and' not)*
not     := 'not' not | cmp
cmp     := add (('='|'!='|'<'|'<='|'>'|'>=') add)?
add     := mul (('+'|'-') mul)* ;   mul := unary (('*'|'/') unary)*
unary   := '-' unary | primary
primary := NUMBER | 'string' | COLUMN | '(' expr ')' | func '(' args ')'
func    := min | max | abs | sqrt | round | if | isnull | null | yyyymm
```

NULL semantics mirror SQL (arithmetic propagates NULL; comparisons with
NULL are falsy) so recomputation oracles skip incomplete rows —
completeness is class 1's job. `isnull()`/`null()` exist for
*conditional* completeness ("a closed cycle carries a closure date");
`yyyymm(date)` expresses date↔period consistency as a derivation
(`MES_DEFAULT = yyyymm(FECHA_DEFAULT)`) instead of an unsatisfiable
sampling constraint.

### 6.4 The synthetic twin (`apdq/twin.py`)

Clean **by construction**, from the manifest alone: tables built in
FK-topological order; source columns sampled from domains (FKs from
actual parent keys, uniques via counters, nullables at their declared
`null_rate`); date chains sorted into compliance; constraints enforced
by rejection sampling (bounded retries, actionable error naming the
near-unsatisfiable constraint); derived columns evaluated in
dependency order, `when:` columns taking the formula where the
condition holds and a fresh sample elsewhere; panel tables generated as
consecutive-month series with distinct series keys and non-decreasing
cumulatives (free branches re-sampled per period because cumulative
columns move conditions across thresholds); reconciliation surfaces
mirrored from the facts; control totals computed from the generated
rows. Every generated oracle returning zero rows on this twin is the
executable meaning of "the invariants hold".

### 6.5 The requirement register (`apdq/register.py`)

One row = one atomic obligation: regulation, section, paragraph, the
obligation sentence, `text_sha256` (normalization-stable hash of the
source text — PDF reflows don't void pins), the concepts constrained,
the defect classes implied, and a sign-off state (`pending` / `signed`
+ `signed_by` / `waived`). Three machine gates: **unsigned** (any
pending row blocks certification), **stale** (supplied regulation text
no longer matches a pin — amendments reopen exactly the affected
rows), **unbound** (a signed requirement names a concept the manifest
doesn't bind — a finding by construction).

### 6.6 Lineage obligations (`apdq/lineage.py`)

The §4.2 completeness argument as a program: walks every node and
reports `ok` / `waived` / `missing` for each obligation (formula,
domain, reconcile, control). Certification requires zero `missing`;
waivers are signed statements and appear on the audit pack's face —
a waiver hides nothing, it signs for the gap.

### 6.7 Defect generators (`apdq/defects.py`)

Per (class × applicable node), a `GeneratedDefect` pairing
`plant(twin, rng, k)` (k mutated rows under fresh attribution keys —
or series keys / aggregates where row attribution is impossible) with
`oracle_sql` (projects the attribution key). Notable mechanics: class-6
oracles are null-aware (`NULL where the formula yields a value` is a
defect, not a skip) and text-aware (exact null-safe equality for
text-valued formulas); conditional columns get branch-guarded class
1/2/6 oracles; constraint violation search tries single columns then
pairs, with reviewable `plant:` hints beyond that; population defects
mutate by deletion/fabrication and are attributed in aggregate.

### 6.8 Harness, audit pack, CLI (`apdq/harness.py`, `apdq/audit_pack.py`, `apdq/__main__.py`)

See §7 for the protocol. CLI:

```bash
python -m apdq validate MANIFEST     # load + validation errors
python -m apdq lineage  MANIFEST     # obligation walk (CI-gateable)
python -m apdq defects  MANIFEST     # list generated defects
python -m apdq twin     MANIFEST -o twin.db
python -m apdq certify  MANIFEST [--register REG] [--out DIR] [--seed N] [--k N] [--json]
python -m apdq propose  EXTRACT.db --name X -o draft.yaml   # §9
```

`certify` exits non-zero unless every gate passes — wire it into CI
directly.

## 7. The certification protocol

**Level B (bank deployment)** — what `certify` runs:

1. **Lineage gate** — zero missing obligations (waivers enumerated).
2. **Register gate** — zero unsigned rows; zero unbound concepts;
   (with supplied text) zero stale pins.
3. **Specificity** — every generated oracle returns 0 rows on the
   clean twin. A firing oracle is a bug in the oracle.
4. **Recall** — per defect, k mutations planted on a fresh twin copy;
   the oracle must catch all k (aggregates must fire). Certification
   demands 1.0 — per defect, hence per class and per node.
5. **Mixed run** — all point defects planted at once; hits attributed
   by planted key; overlap reported, overbroad oracles flagged
   (classes 6 and, in date-heavy schemas, 9 legitimately propagate).

The **audit pack** (JSON + HTML) carries: verdict and per-gate status,
per-class coverage, per-defect clean/caught/recall with the oracle SQL,
waived obligations, register state, overlap/overbroad, the declared
limits (§11), and the pinned inputs (manifest, register, seed) that
make the run third-party reproducible.

**Level T (tool conformance)** — the published reference bank +
mutation corpus any implementation must pass; `apdq/examples/` is its
seed (roadmap E11). This is what makes APDQ a *standard* rather than a
product: the conformance suite is the spec, as with compiler test
suites and OWASP ASVS.

**Versioning discipline**: certificates pin (regulation hash, binding
hash, ruleset, twin seed); any changed input voids only the affected
cells; quarterly EBA validation-rule updates and yearly circular
amendments become incremental diffs (E9).

## 8. Worked proof: the ciclos_full certification

`apdq/examples/` contains three manifests, in ascending order of proof:

1. **`mini_ciclos`** — compact IRB schema (2 tables, 26 columns, 50
   defects, classes 1–9) + the GL/2017/16 example register. CERTIFIED.
2. **`retail_mortgages`** — a deliberately different domain proving
   the *zero-code-edit* gate: certified end-to-end from its YAML alone.
3. **`ciclos_full`** — the DQC eval schema at production width:
   contract master, ~60-column fact table (full derivation chain
   PD/LGD/MoC/EAD/ECL/RWA transcribed from
   `DQC/eval/generate_db.derive()`), monthly panel, basilea/colaterales
   surfaces, GL control totals. **CERTIFIED: 168 defects, all ten
   generic classes, specificity 1.0, recall 1.0.**

**Catalog parity**: `examples/ciclos_full/crosswalk.yaml` maps all 67
entries of the hand-written catalog (`DQC/eval/defect_catalog.py`) onto
generated defects: **58 mapped, 7 partial** (each with the
simplification named: 2-band grade↔PD monotonicity, YYYYMM-granular
collateral staleness, per-cycle snapshot surfaces for the contract
attributes, cure-unconditional panel monotonicity), **3 excluded with
reasons** (a string-prefix predicate the expression language lacks;
two catalog decoys whose honesty role the overbroad check plays).
`tests/test_apdq.py::test_crosswalk_is_total` keeps the mapping total
against both the catalog and the generator output — the crosswalk
cannot silently rot.

This closes the original objection to the MVP ("your 100% is relative
to a hand-written catalog on one schema"): the catalog is now
*reproduced from a declarative manifest*, and the same generators
certified a second, unrelated schema without code changes.

## 9. Where the LLM sits

Outside the trust chain, always. The certification path (§7) consults
no model. What models do — and where (`apdq/assist.py` + the RegLLM
DQC agent, `api/routers/dqc.py`):

- **Propose bindings**: `apdq propose extract.db` drafts a manifest
  from a real extract (schema introspection, domain guessing including
  enum detection) with every judgment as an explicit `TODO`; an LLM
  with the data dictionary and the regulation RAG pre-fills TODOs;
  `load_manifest` structurally refuses the draft until every
  derivation is documented and every concept resolves.
- **Extract requirements**: regulation text → draft register rows,
  landing as `status: pending`, unusable until signed.
- **Transcribe formulas** out of SAS estates (the AST compiler) into
  the expression language — proposals, then mutation-verified.

The economics: a full generation pass costs single-digit dollars hosted
or ~nothing on the local GGUF/Ollama path (the data-egress-sensitive
deployment Spanish banks prefer), against consultancy rule batteries at
€200K+ per engagement. The certificate never depends on believing a
model — which is the difference between this and every "GenAI rule
suggestion" feature on the market.

## 10. Verification of the implementation itself

The certifier is code too (`tests/test_apdq.py`, 46 tests; plus the
legacy harness's own 22 in `tests/test_dqc_eval.py`):

- expression language: eval/SQL backend agreement, NULL semantics,
  garbage rejection;
- manifest validation: every refusal path (missing formula, unknown
  concept, cycles, non-monotone ordering domains, non-PK reconcile);
- twin: all generated oracles silent on clean twins; determinism per
  seed;
- lineage: gaps detected when obligations are stripped;
- register: unsigned/stale/unbound gates, hash normalization;
- end-to-end: all three example schemas certify; every generic class
  present; **the verdict is not vacuous** (flipping any specificity or
  recall bit un-certifies);
- crosswalk totality (§8).

## 11. Roadmap and declared limits

Expansions E1–E11 with their code seams are specified in
[`../apdq/README.md`](../apdq/README.md) (E2, generic panel, is done);
the product sequencing (Anejo IX register first, then EBA
validation-rule import, then AnaCredit/CIRBE surfaces, dialect
compilers, findings ledger, Level T publication, certification program
for audit firms) is in [`MVP_ROADMAP.md`](MVP_ROADMAP.md).

Printed in every audit pack, and repeated here because a standard that
hides its limits is marketing: external-evidence defects and coordinated
consistent falsification are out of scope for any data-side actor;
class 11 findings are advisory; interpretation and estimation judgment
are flaggable, not certifiable; waived obligations and crosswalk
`partial`s are enumerated on the face of the certificate; certification
covers the bound scope, no more.

## 12. Glossary

| Term | Meaning |
|---|---|
| Audit pack | The versioned JSON+HTML evidence bundle a certification run emits; reproducible from pinned inputs. |
| Auditor parity | The guarantee: everything detectable from data + documentation + regulation is detected; the rest is declared. |
| Binding manifest | The single per-bank YAML mapping columns to concepts, with formulas, domains, constraints, surfaces, controls, waivers. |
| BIRD / IReF | ECB's open input-layer data dictionary / the 2029 integrated-reporting regulation that will make banks map to it. |
| Clean by construction | The synthetic twin satisfies every declared invariant because of how it is generated, not because it was checked afterwards. |
| Concept | A canonical business notion (e.g. *exposure at default*) independent of any bank's column names. |
| Conditional derivation (`when:`) | A field whose documented formula binds only under a condition; free (sampled) elsewhere. |
| Control totals | Aggregates (counts, sums) tied to an independent surface; the population-completeness oracle. |
| Decoy / overbroad | Honesty devices: trivial planted checks that expose score inflation; oracles firing across unrelated defects. |
| Lineage DAG | Fields as nodes, documented derivations as edges; the object the completeness argument walks. |
| Mutation testing | Planting a known defect and demanding the check that claims to cover it fires. |
| Oracle | Code that knows what "correct" looks like and can say pass/fail — here, generated SQL projecting an attribution key. |
| Recall / specificity | Fraction of planted defects caught / silence on clean data. Certification demands 1.0 / 1.0. |
| Reconciliation surface | A second place the same fact is recorded and must agree (source system, register, report). |
| Register (requirement) | Atomic regulatory obligations, hash-pinned to their source text, human-signed. |
| Residual | The declared out-of-scope set: what no auditor detects from the data either. |
| Synthetic twin | The generated database with the bank's structure and none of its data, used for planting mutations. |
| Waiver | A signed, visible statement accepting an obligation gap — shown on the certificate, never silent. |

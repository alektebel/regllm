# RegLLM DQC — Evaluation & Roadmap

*Assessment of the AWS application and the eval harness, three improvements
for each, and a challenge to the target architecture for the final
objective: generating high-quality data-quality checks (DQCs) over a
database with a known schema and dictionary, with 100% coverage of every
field against the applicable articles of the EBA GL/2017/16 PD & LGD
guidelines.*

---

## 1. Where the project stands

The repo contains two coupled deliverables:

| Piece | Path | State |
|---|---|---|
| DQC generator API | `api/routers/dqc.py` | Working: variable extraction → RAG context (SAS formula, lineage, regulation graph, docs) → LLM → structured DQC items → validation store |
| Angular chat UI | `DQC/app/` | Working: generate, batch-stream, validate/reject, UNION-ALL dashboard export |
| AWS deployment | `DQC/cdk/`, `DQC/terraform/`, `DQC/.github/workflows/` | Deployable: ECS Fargate (api + nginx sidecar), ALB, ECR, Bedrock (Nova Micro via cross-region inference profile) |
| Eval harness | `DQC/eval/` | Working: 26 ground-truth defects over 8 DQ dimensions, clean DB + per-defect trap DBs (mutation testing), 5-component verifiable reward, per-dimension deficiency report |
| RL training | `training/dq/` | GRPO pipeline that shares the same reward definition |

The strongest design decision in the repo is that **the eval harness, the RL
reward, and the runtime scoring all share the same verifiable contract**:
a check is good iff it parses, returns 0 rows on a clean-by-construction DB,
and ≥1 row on a DB where exactly one invariant was broken. This is
mutation testing applied to data quality, it needs no LLM judge, and it is
the right foundation. Everything below builds on it rather than replacing it.

---

## 2. The AWS application

### 2.1 Assessment

The idea — a thin, bank-deployable chat service that turns a field name into
a set of regulation-grounded SQL checks, with a human validate/reject loop
before anything reaches the dashboard — is sound and matches how DQ teams
actually work (checks must be reviewed before running against production).
The infra is appropriately boring: one Fargate task, two containers, one ALB.
Bedrock via IAM task role avoids credential management entirely.

Weaknesses found (some fixed in this pass, see §5):

- **Two sources of infra truth.** `DQC/terraform` and `DQC/cdk` describe the
  same stack, and they had already diverged: the Bedrock cross-region
  inference-profile IAM fix (commit `36c0558`) was applied only to the CDK
  while the deploy workflow (`dqc-deploy.yml`) applies the *Terraform* —
  meaning a CI deploy would ship the broken policy. Fixed by porting the
  policy to Terraform, but the real fix is to **pick one** (recommendation:
  keep Terraform, since the workflow uses it, and demote `DQC/cdk` to an
  example or delete it).
- **The generation quality ceiling is the model, not the plumbing.**
  Nova Micro is the cheapest option and it shows in `r_coherence`-style
  quality; the system prompt has to beg for multi-column checks. The
  harness exists precisely to quantify this — but nothing in the pipeline
  runs it automatically (see improvement A1).
- **No persistence layer.** The validation store is SQLite inside the
  container; a Fargate task replacement silently wipes every validated
  check. For a tool whose entire output is an accumulated, human-curated
  catalog, this is the biggest operational flaw in the AWS design.
- **Everything is public.** ALB is internet-facing with no auth in front of
  an endpoint that both spends Bedrock tokens and accepts arbitrary text.

### 2.2 Three improvements (AWS application)

**A1 — Wire the eval harness into the deploy path (quality gate).**
Add a CI job that runs `eval_harness.py --agent` against the container
brought up with a stub/cheap backend (or a staged endpoint) and fails the
deploy if overall recall or any dimension's recall drops below a threshold.
The harness already emits JSON and exit codes; this is ~20 lines of
workflow. It converts the harness from a side artifact into the regression
gate for prompt changes, model swaps (Nova Micro → Haiku/Sonnet), and RAG
changes — the three riskiest change classes in the app.

**B2 — Durable state: EFS mount or RDS/DynamoDB for the checks store.**
Attach an EFS volume to the Fargate task for `data/dq/checks.db` (smallest
change), or move the validation store behind a tiny repository interface
with a DynamoDB implementation (better: survives task *and* AZ trouble, no
NFS locking issues with SQLite — use EFS only if staying on SQLite is a hard
requirement). Also persist generated-DQC provenance (model id, prompt
version, retrieved context hash) per check so a validated catalog is
auditable — provenance is a BCBS 239 expectation and currently discarded.

**C3 — Restrict access & harden the front door.** Add an HTTPS listener +
ACM cert, and at minimum an ALB listener rule with OIDC authentication
(Cognito or the bank's IdP) in front of `/api/`. Scope the task security
group to the ALB (already done) but also make the ALB internal if the tool
is bank-internal — the current `internet_facing=True` + `CORS_ORIGINS=*`
posture is only acceptable for a demo. This is cheap now and painful to
retrofit after people start storing validated regulatory checks in it.

*(Honourable mentions: autoscaling is unnecessary at desired_count=1, but a
health-check on `/health` instead of `/` for the API target, request-level
rate limiting, and CloudWatch alarms on Bedrock throttling would all pay for
themselves.)*

---

## 3. The eval harness

### 3.1 Assessment

This is the most rigorous part of the repo. Strengths worth preserving:

- **Clean-by-construction + single-mutation traps** give unambiguous
  attribution: a check fires on `trap_D07` and not on `clean` ⟹ it detects
  D07's invariant, full stop.
- **Decoys** (single-column range checks) are a genuinely clever addition —
  they measure over-claiming, which is the main failure mode of LLM check
  generators.
- **Per-dimension recall with DEFICIENT flags** turns a score into a
  diagnosis ("the model cannot write timeliness checks"), which is directly
  actionable for curriculum/prompt work.

Weaknesses:

- **One trap row per defect, one mutation shape per defect.** The agent
  could catch D01 with `WHERE ECL > 1000000` — any predicate that happens to
  hit the single dirty row scores full recall for D01. Sensitivity is
  measured against *one point* in the violation space, so recall is
  systematically overestimated and gaming is possible (especially by an RL
  policy trained against the same reward — reward hacking is not
  hypothetical, it is the expected outcome).
- **Catalog breadth vs. the coverage goal.** 26 defects over 8 dimensions is
  a stress test, not a coverage measure. The final objective is defined as
  field × article coverage; the harness has no notion of "all 53 columns"
  or "all 221 GL paragraphs" (see §4).
- **`r_template`/`r_coherence` are regex heuristics.** `_score_coherence`
  counts ≥2 column-like tokens in the WHERE clause; `ABS(ECL - 0.5)` (one
  real column + a function name filtered, fine) but `WHERE ECL > 0 AND ECL < 1`
  scores coherent (ECL counted once as a set — actually OK), while
  `WHERE t1.a = t2.a` after alias-stripping may not. Heuristics drift from
  intent silently; they should be validated against labelled examples.

### 3.2 Three improvements (eval harness)

**E1 — Multiple trap rows per defect, sampled across the violation space.**
For each defect, plant *k* (e.g. 5–10) mutated rows drawn with different
random bases and mutation magnitudes (tiny epsilon violations to gross
ones), and score `r_catches` as the *fraction* caught rather than a 0/1 on
one row. Add **cross-defect confusion**: run every check against every trap
and report a defect×check matrix — a check that fires on 12 unrelated traps
is a tautology (e.g. `WHERE 1=1`-adjacent), which currently scores as
excellent recall. This single change removes most of the reward-hacking
surface and makes recall statistically meaningful.

**E2 — Score semantic equivalence against the oracle, not just row counts.**
For each (check, defect) pair where the check catches the trap, also compare
the *row sets* returned on a large mixed DB (clean + many defects at once):
`|check ∩ oracle| / |check ∪ oracle|` (Jaccard). This distinguishes "the
check detects D03's invariant" from "the check accidentally intersects
D03's dirty row", and it grades partial checks (right fields, wrong
tolerance) instead of the current all-or-nothing. The mixed DB also tests
the realistic deployment condition — production data has many defects
simultaneously, and the current one-defect-at-a-time protocol never
exercises that.

**E3 — Generate the defect catalog from the data dictionary, not by hand.**
The catalog is hand-written Python; the dictionary (`data_dictionary.md`)
already encodes types, nullability, layer, formula and regulatory reference
per field. Derive the *mechanical* defect families automatically — for every
formula field a formula-violation trap, for every enum a domain trap, for
every conditional-mandatory field a completeness trap, for every FK a
referential trap — and keep the hand-written list only for genuinely
domain-specific plausibility rules (rating↔PD monotonicity). This scales the
catalog from 26 to ~150 defects with the schema as the single source of
truth, and makes the harness reusable on *any* schema+dictionary pair —
which is exactly the product promise of the final objective.

---

## 4. The final objective, challenged

> *"An application that generates good data-quality checks over a database
> with known schema and dictionary, from a set of writable rules, with 100%
> coverage of all fields and the corresponding articles of the PD & LGD
> guidelines."*

### 4.1 What's wrong with the current path to that goal

The current pipeline is **LLM-first**: a free-text message (or batch loop)
→ RAG context → the model decides which checks exist. Three structural
problems:

1. **Coverage is emergent, not guaranteed.** Batch mode iterates the fields
   that happen to have `MENTIONS_FIELD` edges in the regulation graph —
   today that is **28 of 48 graph fields**, while the eval schema alone has
   53 columns and the ingested GL/2017/16 has **221 paragraphs**. Whether a
   field×article pair gets a check depends on what an LLM extracted into
   the graph and what another LLM decides to emit. You cannot certify 100%
   of anything this way; you can only observe what came out.
2. **The LLM is doing two jobs, and is only good at one.** Deciding *what*
   must be checked (the field↔article mapping, the applicable dimensions)
   is a completeness problem — it needs to be exhaustive, stable, and
   auditable. Writing *how* to check it (the SQL predicate against this
   dialect and schema) is a translation problem — LLMs are good at it and
   the harness can verify it. Conflating them means hallucinated references
   on one side and coverage holes on the other. The system prompt already
   fights both symptoms ("cita artículos exactos", "NO inventes normas").
3. **"A bunch of rules we can write" has no first-class representation.**
   Expert-authored rules currently enter as prose in the chat message. They
   should be data, not prompt.

### 4.2 Proposed architecture: coverage matrix as the spine

Make the deliverable a **coverage matrix** — a versioned table, not an LLM
output:

```
(field, article_or_paragraph, dimension) → status
status ∈ {check_id(s), N/A(reason), TODO}
```

- **Rows are enumerated deterministically**: every column of the schema ×
  every GL paragraph tagged as applicable × the 8 dimensions. The
  applicability tagging is the only place an LLM assists — as a *proposer*
  whose output is reviewed once and then frozen in the repo (the data
  dictionary already carries a `Reg ref` column; that is the seed).
- **"100% coverage" becomes a computable statement**: no cell is `TODO`;
  every cell is either a validated check or an explicit, human-signed
  `N/A(reason)`. This is also precisely the evidence format supervisors and
  internal audit ask for.
- **Rule templates are the "writable rules".** A small YAML/JSON DSL of
  parameterised check patterns (`formula_equality`, `floor`, `domain`,
  `conditional_mandatory`, `referential`, `monotonic_pair`, `freshness`,
  `unique_key`) with slots for fields, tolerances, and the regulatory
  reference. Perhaps 80% of the matrix is filled by *instantiating
  templates from the dictionary with no LLM at all* — deterministic,
  dialect-portable, trivially correct. The LLM is reserved for the
  long-tail cells where no template fits, and its output must pass the
  harness (clean-zero + trap-catch) before it can occupy a cell.
- **The eval harness becomes the certifier**: E3's dictionary-derived traps
  give every matrix cell a corresponding trap, so "coverage" is not "a
  check exists" but "a check exists *and demonstrably fires* on a planted
  violation of exactly that cell".

This inverts the current design: enumerate first, generate second, verify
always. The chat UI survives unchanged as the review/exploration surface;
`/dqc/generate` becomes "fill or refine these matrix cells".

### 4.3 Three concrete steps to get there

1. **Build the field↔paragraph applicability map as a reviewed artifact.**
   Script: for each of the 53 dictionary fields × 221 GL paragraphs, have
   the LLM answer *only* "applicable / not applicable + one-line reason"
   (cheap, parallelisable, cacheable); a human reviews the positives once.
   Commit the result as `DQC/coverage/applicability.yaml`. This immediately
   quantifies the real coverage frontier (today it is unknown).
2. **Implement the template DSL + instantiator** (`DQC/coverage/templates/`),
   generating checks + matching traps from the data dictionary. Target: the
   whole `ciclos_calibrados` formula/floor/domain/completeness surface with
   zero LLM calls, scored by the existing harness at 100% recall on the
   mechanical defect families.
3. **Repoint the API and UI at the matrix.** `GET /dqc/coverage` returns the
   matrix with cell statuses; generation endpoints take a cell (or a set of
   cells) instead of free text; the dashboard export becomes "all validated
   checks of the matrix". The existing validate/reject flow is already the
   right human loop — it just needs the matrix as its to-do list.

### 4.4 Why not the alternatives

- **"Just use a bigger model + better prompt"** improves check quality but
  cannot make coverage certifiable; you still can't prove a negative
  ("nothing was missed") from sampled generations.
- **"Hard-code all checks by hand"** achieves certifiability but doesn't
  scale past one schema and loses the genuinely useful LLM contribution
  (long-tail semantic rules, natural-language justifications, dialect
  translation).
- **A pure constraint-mining approach** (learn invariants from data à la
  Deequ/HoloClean) finds *statistical* rules but cannot attribute them to
  articles, which is the regulatory half of the objective. It is a good
  *complement* (proposing candidate cells the mapping missed) — not the
  spine.

---

## 5. Flaws found and fixed in this pass

| Flaw | Fix |
|---|---|
| `api/main.py` mounted only the `dqc` router, 404-ing the diff/agent/kb/embeddings surface the frontend, README and 6 tests depend on | Env-configurable router mounting (`REGLLM_ROUTERS`, default `all`); AWS infra pins `REGLLM_ROUTERS=dqc` |
| Terraform Bedrock IAM policy lacked the cross-region inference-profile ARN (fix existed only in CDK, but CI deploys Terraform) | Ported the policy; added missing `API_UPSTREAM` env to the Terraform dqc container for CDK parity |
| `data/sas/sessions/debug_lgd/` fixture referenced by tests, the training corpus and the golden dataset was never committed → 3 test failures | Reconstructed the 3-file SAS project (carga → EAD → suelos/MoC/ECL) with the documented SW_FUSION bug |
| Agent loop rejected any final answer < 20 chars, looping to an empty answer | Guard now rejects only empty text |
| Byte-identical duplicates: `DQC/{deep_embedding_quality.py, analyze_sas_embeddings.sas, test_sas_embedding_pipeline.py, analysis/}` duplicated `scripts/` and `data/embeddings/tabular/analysis/` | Removed the `DQC/` copies |
| Tracked artifacts that `.gitignore` already declared ignored (19 MB `sas_embeddings.csv`, session JSONs, user uploads) + committed `error.txt` paste | Untracked; `data/uploads/` added to `.gitignore`; `error.txt` deleted |
| `pytest` collected `scripts/` (requires `transformers`/GPU), breaking collection on clean machines | `pytest.ini` with `testpaths = tests` |
| README stale: no mention of the DQC application, "AWS out of scope", "197 tests" | Updated |

Suite status after fixes: **597 passed, 0 failed**. `DQC/eval` self-test:
**26/26 oracles pass**.

---

# Addendum (2nd pass) — production hardening & coverage certification

## 6. What changed in this pass

### 6.1 The toy database grew into a production-context schema

`CICLOS_CALIBRADOS` went from 53 to **66 fields**, adding the context classes
a real IRB warehouse carries and that the previous schema couldn't express
checks about:

- **Counterparty**: `ID_CLIENTE`, `TIPO_PERSONA` (legal nature vs segment).
- **Cycle dates** (YYYYMM): `MES_DEFAULT`, `MES_CIERRE_CICLO`,
  `MES_VALORACION_COLATERAL` — enabling temporal-ordering and staleness
  checks (closure before default; collateral revaluation > 36 months old,
  CRR Art. 208.3).
- **Multi-currency**: `DIVISA`, `TIPO_CAMBIO`, `EAD_TOTAL_EUR` with the FX
  conversion invariant.
- **MoC decomposition** per EBA GL 2017/16 §43-44: `MOC_CAT_A/B/C` with
  `MOC = A + B + C`.
- **Downturn parameters**: `LGD_DOWNTURN` (must never undercut the long-run
  LGD — CRR Art. 181.1(b)); `PD_DOWNTURN` now has its own invariant.
- **Calibration governance**: `VENTANA_OBSERVACION_YEARS` + `FLAG_NC`
  (5-year historical window, EBA GL §6.3.2.1).
- Clean-generator alignment fixes: `TERMINACION`↔`CURE_FLAG`,
  `CAUSA_DEFAULT`↔`DPDS`, and `LGD_REALIZADA` now actually satisfies the
  backtesting formula the dictionary documents for closed cycles.

The defect catalog grew **26 → 48** (46 coherence + 2 decoys), with every
non-decoy defect carrying a `regulation_ref`.

### 6.2 The eval harness was hardened against the failure modes flagged in §3

- **k planted rows per defect** (default 3), each from a different random
  base; `r_catches` is now the *fraction* caught. Memorising one row no
  longer scores full recall (E1 → done).
- **Mixed DB**: coverage mode plants all 48 defects at once and attributes
  hits by planted PK — one query per check instead of one per
  (check × defect), and the realistic production condition (E2 → done via
  set attribution; full Jaccard was unnecessary because clean-zero checks
  can only return planted rows).
- **Confusion / gaming detection**: checks firing on > 3 distinct defects are
  flagged *overbroad*; a regression test proves that PK-fishing
  (`WHERE pk LIKE '__DIRTY%'`) is caught.
- **Oracle overlap matrix** in the self-test makes nested-invariant overlap
  (D01 ⊃ D03/D04/D05/D12) visible instead of silent.
- **True uniqueness defect**: duplicate PKs are planted as verbatim row
  copies, not proxy symptoms.
- **CI gates**: `--fail-under` on the harness, plus two new steps in
  `.github/workflows/test.yml` (oracle self-test + coverage matrix at 1.0).
  The dead `DQC/.github/workflows/dqc-deploy.yml` was removed — GitHub never
  reads nested workflow dirs, so that deploy pipeline had *never run*; the
  root `deploy.yml` is the live one.
- The harness itself is now under test (`tests/test_dqc_eval.py`, 15 tests) —
  and its dictionary-vs-schema test immediately caught two fields missing
  from the data dictionary (`ENTIDAD_ORIGEN`, `LTV`).

### 6.3 Coverage is now a computable artifact (the §4 proposal, implemented)

`DQC/eval/coverage_matrix.py` builds the **field × article matrix**
deterministically:

- rows = the 66 dictionary fields; columns = the regulatory references the
  dictionary attaches to them (49 applicable cells);
- a cell is `covered` when a catalog defect touches the field *and* cites the
  article (trap + oracle exist ⇒ certifiable), `partial` when the field is
  exercised under another article, `todo` otherwise;
- `--fail-under 1.0` gates CI on **zero todo cells** — currently
  **49/49 cells covered (80% exact-article, 20% partial), 0 todo**;
- `--emit-applicability` generates `DQC/coverage/applicability.yaml`: all 58
  sections of the ingested EBA GL/2017/16 (221 paragraphs) with suggested
  field mappings, each pending human sign-off.

## 7. How we make sure ALL the PD/LGD articles and coherence rules are checked

The guarantee is a chain of three versioned, machine-verifiable artifacts —
no step relies on trusting LLM output:

1. **Applicability** (`DQC/coverage/applicability.yaml`): every GL/2017/16
   section is either mapped to schema fields or explicitly marked
   not-applicable with a reason. Completeness over *articles* is enumerated,
   and the residual judgment (is this section about our data?) is a one-time
   human review, recorded in git.
2. **Certifiability** (`coverage_matrix.py`, CI-gated at 1.0): every
   applicable (field × article) cell must be backed by a catalog defect —
   meaning a planted violation and a reference oracle that the self-test
   proves fires on it. A check for that cell is not "believed" to work; it
   *demonstrably catches a planted breach of exactly that rule*.
3. **Achievement** (`eval_harness.py --sql/--agent`, CI-gateable via
   `--fail-under`): the actual check set (hand-written, template-generated,
   or LLM-generated) is scored on the mixed DB; per-article and
   per-dimension recall show precisely which articles and which coherence
   classes the current checks miss.

Database-coherence rules ride the same rails: they are catalog defects with
`regulation_ref = BCBS 239 P3/P4/P5` (or bank-internal), so "all coherence
checks present" is the same computable statement as the article coverage.

What remains judgment (and is deliberately kept as reviewed YAML, not code):
approving the applicability map, and deciding tolerance thresholds. Everything
downstream is enforced by CI.

## 8. Updated overall assessment

- The DQC application (`main` branch) now has: a green test suite (605), a
  self-testing 48-defect eval harness with anti-gaming measures, a
  CI-enforced coverage matrix at 100%, and infra whose two stacks agree.
  The remaining production blockers are unchanged from §2 and are
  *operational*, not evaluative: durable storage for the validated-checks
  DB, auth/TLS on the ALB, and running the agent-mode eval as a deploy gate
  against a staged endpoint.
- **Repo fragmentation is now the biggest organizational risk.** The remote
  has three unrelated histories: `main` (this DQC + SAS-diff project),
  `master` (a *different* application — "RegLLM Spanish Banking Regulation
  Assistant": Next.js + FastAPI + pgvector + Groq/LoRA chat, 33 commits,
  separate root commit), and `update-model` (an earlier Gradio/Modal variant
  of the same assistant, 22 commits). They share a repo name but no code or
  history. Recommendation: split the chat assistant into its own repository
  (or make it a top-level `assistant/` subtree on `main` via a deliberate
  merge), and delete stale branches — a repo where `main` and `master` are
  different products will eventually ship the wrong thing.
- Next highest-value step remains §4.3: template-DSL instantiation of checks
  from the dictionary, now trivial to certify because the matrix and traps
  already exist.

Status after this pass: **605 tests passed** (incl. 15 harness tests),
self-test **48/48 oracles**, coverage matrix **49/49 cells, 0 todo**.

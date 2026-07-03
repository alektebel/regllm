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

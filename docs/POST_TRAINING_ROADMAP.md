# Post-training roadmap — a staged recipe for a DB-analysis LLM

*How to take the DQC agent from prompt-engineered (today) to
post-trained, using phased SFT→RL as the recent text-to-SQL / agentic
data-analysis literature now does it — grounded in the files this repo
already has.*

---

## Why this repo is unusually ready

Most teams that want to post-train a text-to-SQL / data-analysis model
have to build a reward first. We already have one, and it is the right
kind — **verifiable, execution-based, no LLM judge**:

> A check is good iff it **parses**, returns **0 rows** on a
> clean-by-construction DB, and **≥1 row** on a DB where exactly one
> invariant was broken. (`DQC/eval/eval_harness.py`, mirrored by the RL
> reward in `training/dq/`.)

That is mutation testing applied to data quality. The frontier papers
below don't replace it — they **densify** it and **stage** it.

## What "phased post-training" means now (2025-2026)

The default recipe has moved past a single SFT→GRPO step to **staging by
sub-task and by reward density**:

- **Reasoning-SQL** (Pourreza et al., 2025) — SQL-tailored *partial
  rewards* (schema-linking accuracy, n-gram similarity, syntax validity,
  AI feedback) on top of sparse execution accuracy; RL-with-these-rewards
  generalises better than SFT. <https://arxiv.org/abs/2503.23157>
- **Progress-SQL** — *progressive rewards*: structural/lexical-alignment
  improvement, execution-status transition, format — a curriculum baked
  into the reward. <https://arxiv.org/html/2606.06825v1>
- **Reward-SQL** — a Process Reward Model for step-wise reasoning
  supervision (denser than terminal reward).
- **TRUST-SQL** — tool-integrated *multi-turn RL over unknown schemas*;
  schema discovery becomes part of the agent loop (maps onto our Tier-1
  filter + multi-table grounding direction).
  <https://arxiv.org/pdf/2603.16448>
- **Graph-Reward-SQL** — *execution-free* RL via graph matching + stepwise
  reward; compute the reward without running SQL on real data.
  <https://arxiv.org/abs/2505.12380>
- **MARS-SQL** — multi-agent RL (grounding / generation / validation),
  the baseline we already mirror architecturally.
  <https://openreview.net/forum?id=EURAfiUpVJ>

Broader agentic data-analysis (beyond single-shot SQL):

- **Scaling Generalist Data-Analytic Agents** — scaling agent
  post-training in the data-analytic scenario specifically.
  <https://arxiv.org/pdf/2509.25084>
- **Rewarding the Scientific Process** — process-level reward modelling
  for multi-step analysis (maps onto our decision trace).
  <https://arxiv.org/pdf/2604.24198>
- **Mixture-of-Minds** — multi-agent RL for table understanding.
  <https://arxiv.org/pdf/2510.20176>
- **The Landscape of Agentic RL for LLMs: A Survey** — read first to place
  everything above. <https://arxiv.org/abs/2509.02547>

## The staged recipe, tied to our files

| Phase | Goal | Reward / data | Where it plugs in |
|---|---|---|---|
| **0. Cold-start SFT** | fluent DQC JSON + valid SAS PROC SQL shape, so RL isn't wasted on format | golden traces (`DQC/eval/golden_traces.json`) + synthetic pairs | new SFT script beside `training/dq/` |
| **1. RL — schema linking** | pick the right fields/tables (and, later, join paths) | dense partial reward = **linking recall** vs gold fields (Reasoning-SQL) | reward over `select_relevant_fields` + the Tier-1 embedder (`dqc_dictionary.py`) |
| **2. RL — generation** | queries that parse, run, and catch the defect | our **mutation-test execution reward** (`DQC/eval/`) + partial rewards (syntax, format, progressive execution-status transition) | extend the GRPO reward in `training/dq/` |
| **3. RL — agentic (optional)** | run the full loop; know when to abstain | phase-2 reward **+ correct-abstention reward** (the `ambigua` branch) | the ReAct loop in `api/routers/dqc_react.py` as the RL environment |

Phase 3's abstention reward is the one thing none of the cited papers do
and that a compliance tool specifically wants: rewarding "this rule is
under-specified, refuse" as a *correct* terminal state, not a failure.

## The caveat that is specific to a regulated setting

Most of these reward functions **execute generated SQL against data**. On
real bank data that is a governance problem. Two mitigations we already
have the pieces for:

1. **Train against the synthetic twin, never real data.** The APDQ
   manifest-driven synthetic DB (`apdq/`, `DQC/eval/generate_db.py`) is
   clean-by-construction and carries planted defects — it is a lawful RL
   environment.
2. **Execution-free reward as a fallback** (Graph-Reward-SQL) when even
   the synthetic execution loop is undesirable.

That pairing — **synthetic-twin environment + staged partial→execution
rewards + an abstention reward** — is, as far as the current literature
goes, a novel and defensible recipe for this domain, assembled from
components already in this repo plus the papers above.

## First concrete step

Before any training: **wire `DQC/eval/eval_harness.py` to the ReAct
endpoint** (`/generate_stream`) so the reward reflects the *real* pipeline
(sufficiency, grounding, judge, correction), and add an abstention metric.
Everything above is only meaningful once the harness scores the pipeline
we actually ship. See `docs/EVALUATION.md` for the harness internals.

## A note on what "validation" means here (it is not a SAS parser)

The pipeline's SQL **validation** — distinct from any training reward — is
two deterministic layers plus one optional LLM layer:

1. **Static** (`dqc_react.static_validate`): a regex tokenizer + set
   membership — every identifier in the query must be a known dictionary
   field, SQL keyword, function, alias, or table part. This is a
   lightweight lexical check, **not a grammar-based SAS parser**.
2. **Dynamic** (`dqc_react.run_query`): the query is executed against the
   uploaded cases loaded into **in-memory SQLite** — a real engine, but
   **ANSI SQLite, not a SAS interpreter**. It catches syntax and runtime
   errors for the ANSI subset; pure-SAS idioms would fail here (which is
   why the generation prompt restricts the model to ANSI operators).
3. **Semantic judge** (optional, `dqc_react.judge_dqc`): LLM-as-judge —
   but this checks *meaning* (does the query implement the rule?), not
   validity, and only runs when `semantic_judge` is enabled.

So: **no custom SAS parser, and LLM-as-judge is only the optional
semantic layer** — validity itself is lexical + SQLite execution.

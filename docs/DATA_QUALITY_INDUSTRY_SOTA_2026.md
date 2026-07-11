# Industry SOTA: Data Quality over Reporting Databases (mid-2026)

Research memo for the RegLLM / DQC project. Covers (1) the current
state-of-the-art in industry, (2) what organisations budget and pay,
(3) the tool landscape, and (4) a concrete path to **100% regulation +
coherence test-suite coverage** for this repo, grounded in the current
state of `DQC/eval/` and `DQC/coverage/`.

Date of research: 2026-07-11. All external figures carry a source in
[References](#references).

---

## 1. Executive summary

- The market has consolidated into **three tiers**: managed ML
  observability platforms ($50K–$200K+/yr), open-source in-pipeline
  testing frameworks (free + engineering time), and governance/quality
  suites ($90K–$300K/yr). Enterprises with regulatory requirements
  typically stack one of each and budget **$200K–$500K/yr** plus 1–2
  FTE admins.
- **LLM-generated data quality rules are now a mainstream vendor
  feature**, not a research curiosity: Monte Carlo's GenAI monitoring
  agent recommends rules/thresholds with a reported **60% acceptance
  rate**; Anomalo, Databricks-ecosystem tooling, and several academic
  systems (LLM-DQR, arXiv:2509.10572) follow the same
  *LLM-proposes → code-verifies → human-approves* pattern.
- **No commercial platform certifies coverage against a regulation at
  article level.** The industry's regulatory answer is the Critical
  Data Element (CDE) × quality-dimension matrix plus attribute-level
  lineage. RegLLM's `coverage_matrix.py` (field × article, machine
  verified against planted defects) is ahead of commercial practice
  and is the project's clearest differentiator.
- The regulatory pressure is real and rising: BCBS 239 compliance is
  assessed in SREP from 2025 under the ECB's RDARR guide; only 2 of 31
  G-SIBs are fully compliant, and the ECB has threatened escalation
  measures. Gartner puts the average cost of poor data quality at
  **$12.9M/yr per organisation**, and 59% of organisations don't
  measure it at all.
- For this repo, "100% coverage" is already correctly *defined*
  (applicability approved ∧ zero matrix TODOs ∧ 100% harness recall).
  The measured gaps as of today: **58/59 applicability entries still
  `review: pending`**, **8/51 matrix cells only `partial`** (84%
  exact-article coverage), and the recall gate is not yet wired to
  1.0 in CI. Section 6 gives the closure plan.

---

## 2. Market landscape: three tiers

Gartner estimates ~50% of enterprises had adopted data observability
tooling by 2026, up from 20% in 2024 — the buying question is now
*which* tool, not *whether*.

| Tier | Representative tools | Cost | Strengths | Weaknesses |
|---|---|---|---|---|
| Managed observability | Monte Carlo, Anomalo, Bigeye, Validio | $50K–$200K+/yr | ML anomaly detection (freshness, volume, schema, distribution, lineage) with near-zero rule authoring; GenAI rule recommendation | Expensive; rules are *learned*, not *derived from regulation* — no auditability story per article |
| Open-source in-pipeline | dbt tests (+dbt-expectations), Great Expectations, Soda Core, Elementary, Deequ | Free + engineering time | Versioned, code-reviewed, CI-gated checks; dbt-native stacks cover 60–70% of modern platforms | Hand-authoring burden; limited anomaly detection; coverage is whatever you wrote |
| Governance / quality suites | Collibra (+DQ module), Ataccama ONE, Informatica | $90K–$300K/yr | CDE registries, lineage, stewardship workflow, BCBS 239 positioning | 6+ month implementations; ROI ~25 months; 1–2 FTE admins; rules still authored manually |

Common guidance in 2026 buyer's guides: teams under ~15 engineers
start with Great Expectations / Elementary; teams with 20+ engineers,
multiple warehouses, **or compliance requirements** add a managed
platform, and regulated enterprises pair it with a governance suite.

### Where RegLLM sits

RegLLM's DQC generator occupies a gap none of the three tiers fills:
**regulation-grounded rule generation with machine-verifiable
coverage**. Managed platforms learn statistical normality; OSS
frameworks execute what humans wrote; governance suites document
intent. None can answer the supervisor's question *"show me that every
applicable paragraph of GL/2017/16 has a firing control"* — which is
exactly what `coverage_matrix.py --fail-under 1.0` computes.

---

## 3. SOTA in LLM-based data quality (vendor + academic)

### Vendor practice

- **Monte Carlo** shipped GenAI observability agents: a *monitoring
  agent* that examines warehouse data and recommends monitors/
  thresholds (early adopters generated thousands of recommendations,
  ~60% accepted) and a *troubleshooting agent* for root-cause
  analysis. It has since extended to LLM-as-judge evaluations and
  agent/AI-output observability.
- **Anomalo** takes a "data trust" angle: ML checks on actual data
  content (not just metadata), no-code business-logic validation, and
  monitoring of data feeding custom LLMs, including unstructured data.
- **Databricks-ecosystem** consultancies ship "dynamic DQ rule
  generation": LLM reads schema + profile statistics, emits
  standardized rules in seconds instead of hand-writing dozens of SQL
  checks.

The convergent architecture across all of these is the one this repo
already implements: **LLM proposes; deterministic code executes and
scores; a human approves**. Nobody trusts raw LLM output as a control.

### Academic SOTA

- **LLM-DQR** (Expert Systems with Applications, 2026): automated DQ
  rule generation for electronic health records; prompts an open model
  (Gemma-3-12B) per rule type, then a *rule-enrichment* phase adds
  precise specification and pseudocode before execution.
- **arXiv:2509.10572** — *Quality Assessment of Tabular Data using
  LLMs and Code Generation*: three-stage framework — statistical
  outlier profiling → LLM rule generation → LLM-synthesized executable
  validators. Same shape as DQC's generate-then-verify loop.
- **arXiv:2507.10934** — *Generating Authentic Errors via LLMs* for
  benchmarking data-cleaning: uses LLMs to plant realistic errors in
  tables. This validates the **mutation-testing evaluation contract**
  `DQC/eval/defect_catalog.py` uses (planted defects with oracles),
  and suggests a future extension: LLM-generated *novel* defect
  variants to stress-test the catalog itself.
- **arXiv:2406.09843** — LLM mutation testing study: LLM-generated
  mutants are more diverse and 1.75× closer to real bugs than
  traditional mutation operators.

Takeaway: the eval harness here (k-row traps, mixed-DB confusion
matrix, decoys, verifiable 5-component reward with **no LLM judge**)
is at or beyond published SOTA. The academic frontier is moving toward
LLM-generated *defects*; the vendor frontier toward *agents that
triage incidents* — both are natural roadmap items (§6.6).

---

## 4. What the industry budgets and pays

### Tooling

| Item | Typical annual cost |
|---|---|
| Monte Carlo / Anomalo (observability) | $50K–$150K (to $200K+ at scale) |
| Ataccama ONE | from ~$90K |
| Collibra base subscription | ~$170K |
| Collibra Data Quality module | ~$156K (budgeted separately) |
| Governance suite all-in (modules, connectors) | $170K–$295K before services |
| Full enterprise stack (observability + governance) | **$200K–$500K** |
| OSS stack (dbt + GE/Soda + Elementary) | $0 licence; ~0.5–1 FTE engineering |

Implementation of a governance suite runs 6+ months with ROI landing
around month 25 — a key argument for the lighter, code-first approach
this repo takes.

### Staffing (US, 2026)

| Role | Median / range |
|---|---|
| Data Quality Engineer | ~$130K avg ($104K–$163K p25–p75, $199K p90) |
| Data Governance Specialist | ~$124K avg ($93K–$168K) |
| Data Governance Manager | ~$119K avg |
| Governance-suite administration | 1–2 FTEs commonly assumed |

### The counterfactual cost

- Gartner: poor data quality costs organisations **$12.9M/yr on
  average** (some 2026 sources cite $15M).
- 59% of organisations do not measure data quality at all; ~60% don't
  measure its financial cost.

### LLM inference cost in context

A full DQC generation pass over a 66-field schema with a 221-paragraph
regulation RAG is single-digit dollars on a hosted frontier model and
effectively free on the local GGUF/Ollama path this repo supports —
noise against the $50K+ licence floor of any commercial tier. The
economic argument for LLM-generated, harness-verified checks is
strong: the expensive part everywhere else is the *human authoring and
review time*, which the coverage matrix + golden traces reduce to an
approval workflow.

---

## 5. The regulatory bar (BCBS 239 / ECB RDARR / EBA)

- The ECB's **RDARR guide** (May 2024) operationalises BCBS 239;
  remediation of risk-data-aggregation deficiencies is a top ECB
  supervisory priority for **2025–2027**, assessed in SREP from 2025,
  with explicit threat of escalation measures.
- Compliance is poor industry-wide: only **2 of 31 G-SIBs** fully
  compliant per the latest Basel Committee assessment; one study finds
  14% of banks fully compliant and 43% materially non-compliant. No
  single BCBS 239 principle is fully implemented by all banks.
- Established practice for demonstrating coverage: identify **Critical
  Data Elements** used in regulatory reporting; attach ownership,
  **attribute-level lineage** (now explicitly expected by the ECB),
  and quality rules per dimension (completeness, accuracy, timeliness,
  consistency, validity). No regulator prescribes a rule *count* —
  coverage is argued via the CDE × dimension matrix and evidenced by
  monitoring output and audit trail.

**Implication for RegLLM:** the field × article matrix is strictly
stronger evidence than the industry-standard CDE × dimension matrix,
because each cell is backed by an oracle that demonstrably fires on a
planted violation — coverage is *proven by execution*, not asserted.
Presenting the three artifacts (approved `applicability.yaml`,
zero-TODO matrix, 100%-recall harness run) as an audit pack maps
directly onto what ECB inspectors ask for under RDARR.

---

## 6. Path to 100% regulation + coherence coverage in this repo

The definition in `DQC/eval/README.md` is already the right one:

> "all articles checked" = applicability.yaml fully approved ∧ matrix
> has no TODO ∧ harness recall = 100% on the target check set.

Measured state (2026-07-11, this branch):

| Artifact | State | Gap |
|---|---|---|
| `DQC/coverage/applicability.yaml` | 59 sections: 50 applicable, 9 n/a; **1 approved, 58 pending** | Human sign-off missing on 58 entries |
| `coverage_matrix.py` | 73 fields, 51 applicable cells, **0 todo**; 43 covered (84%), 8 partial | 8 partial cells lack an exact-article defect |
| `eval_harness.py` recall | gate exists (`--fail-under`) but not pinned at 1.0 in CI | Recall certification not enforced |

### 6.1 Close the human-review gap (blocking, cheap)

Drive the 58 `review: pending` entries to `approved` / corrected. Make
it a PR workflow: one reviewer with regulatory competence per section
batch, CODEOWNERS on `DQC/coverage/`, and a CI check that fails on any
`review: pending`. This is the only step that fundamentally cannot be
automated — it is the human attestation the audit pack rests on.

### 6.2 Upgrade the 8 `partial` cells to `covered`

Each partial cell means the field is exercised by some defect but no
defect carries that exact `regulation_ref`. Author one defect per
missing article in `defect_catalog.py` (oracle + `mutate(row)` +
`regulation_ref`), verify with `--selftest`, and tighten the CI gate
from "0 todo" to "0 todo ∧ 0 partial" (or add
`--fail-under-exact 1.0` to `coverage_matrix.py`).

### 6.3 Pin the recall gates in CI

- `eval_harness.py --sql <shipped_checks.sql> --fail-under 1.0` on the
  mixed DB (overall recall).
- Add **per-dimension** and **per-article** floors (any dimension
  < 100% fails, not just flagged `DEFICIENT`) so a regression in one
  dimension can't hide behind aggregate recall.
- Keep the coherence discipline that already exists: `r_coherence`
  weighting, decoy over-claim = failure, overbroad-check (confusion
  matrix) = failure. These are what make the coverage claim mean
  *coherence coverage* rather than tautological row counting.

### 6.4 Certify the certifier

`tests/test_dqc_eval.py` already regression-tests the harness. Add two
industry-standard hardenings:

- **LLM-generated novel defects** (per arXiv:2507.10934 /
  2406.09843): periodically ask a strong model to propose defect
  variants *not* in the catalog; any variant the shipped check set
  misses becomes a new catalog entry. This converts the catalog from a
  fixed 67-defect set into a ratcheting one and answers the obvious
  auditor objection ("your 100% is relative to your own catalog").
- **Regulation-drift watch**: hash the ingested GL/2017/16 paragraphs;
  any change to the regulation corpus flips affected
  `applicability.yaml` entries back to `pending`.

### 6.5 Produce the audit pack as a build artifact

One CI job emitting a versioned bundle: applicability.yaml (all
approved) + matrix JSON + harness JSON (`--json`) + golden-trace
scores + the git SHA. That single artifact is the "prove data
integrity to regulators" deliverable that Collibra et al. market at
$150K+/yr — here it is a build output.

### 6.6 Complementary industry practices worth adopting (non-blocking)

- **Statistical/anomaly monitors** (freshness, volume, distribution à
  la Monte Carlo/Elementary) as a *complement*: regulation-derived
  checks catch known invariant violations; anomaly monitors catch the
  unknown-unknowns. The cheap path is Elementary or Soda Core on the
  same warehouse.
- **Incident-triage agent**: the vendor frontier is root-cause
  agents; the SAS field-diff explainer half of this repo (Shapley +
  GraphRAG changelog) is already most of one — wiring a failing DQC to
  a field-diff explanation would leapfrog the commercial offerings.
- **Data contracts** at the L0 source-table boundary (`contratos`,
  `basilea_mensual`, `colaterales`), so cross-table reconciliation
  defects (D48–D57) are prevented upstream, not only detected.

### Definition of done

```
CI green ⇔
  applicability.yaml: 0 pending, 0 rejected-without-reason
∧ coverage_matrix:    0 todo ∧ 0 partial   (exact-article = 100%)
∧ eval_harness:       recall = 1.0 overall, per-dimension, per-article
                      ∧ 0 decoy over-claims ∧ 0 overbroad checks
∧ regulation corpus hash unchanged (else auto-revert to pending)
```

---

## References

Market / tools / pricing:
- [Atlan — Top 14 Data Observability Tools in 2026](https://atlan.com/know/data-observability-tools/)
- [Basedash — Best data observability tools compared 2026](https://www.basedash.com/blog/best-data-observability-tools-compared-2026)
- [Medium — Monte Carlo vs Great Expectations vs Soda (2026)](https://medium.com/@aidelearning/data-observability-in-2026-monte-carlo-vs-great-expectations-vs-soda-a-data-engineers-honest-7c8cab1b68f1)
- [Modern DataTools — Monte Carlo Review 2026](https://www.modern-datatools.com/tools/monte-carlo)
- [Atlan — Collibra Pricing Explained](https://atlan.com/collibra/pricing/)
- [DQLabs — Best Data Quality Tools for Enterprise 2026](https://www.dqlabs.ai/blog/best-data-quality-tools-for-enterprise-use-in-2026-a-practitioners-guide/)
- [Data Pilot — Data Quality Tools 2026 Buyer's Guide](https://data-pilot.com/blog/data-quality-tools/)
- [PipeCode — GE vs dbt Tests vs Soda Core](https://pipecode.ai/blogs/data-quality-frameworks-great-expectations-dbt-tests-soda-core)

LLM-based DQ (vendor + academic):
- [TechTarget — Monte Carlo launches first agents for data observability](https://www.techtarget.com/searchdatamanagement/news/366622933/Monte-Carlo-launches-first-agents-for-data-observability)
- [Monte Carlo — Agent Observability Platform](https://montecarlo.ai/platform/agent-observability/)
- [Anomalo — Monte Carlo vs Anomalo](https://www.anomalo.com/blog/monte-carlo-vs-anomalo/)
- [LatentView — Dynamic DQ Rule Generation using LLM in Databricks](https://www.latentview.com/blog/dynamic-data-quality-rule-generation-using-llm-in-databricks/)
- [LLM-DQR — automated DQ rules for EHR (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1532046425001807)
- [arXiv:2509.10572 — Quality Assessment of Tabular Data using LLMs and Code Generation](https://arxiv.org/abs/2509.10572)
- [arXiv:2507.10934 — Generating Authentic Errors via LLMs](https://arxiv.org/pdf/2507.10934)
- [arXiv:2406.09843 — LLMs for Mutation Testing](https://arxiv.org/pdf/2406.09843)

Costs / budgets / staffing:
- [Gartner — Data Quality: Why It Matters](https://www.gartner.com/en/data-analytics/topics/data-quality)
- [Gartner — How to Stop Data Quality Undermining Your Business](https://www.gartner.com/smarterwithgartner/how-to-stop-data-quality-undermining-your-business)
- [Actian — Poor Data Quality Costs $15M Annually](https://www.actian.com/blog/data-management/the-costly-consequences-of-poor-data-quality/)
- [Glassdoor — Data Quality Engineer Salary 2026](https://www.glassdoor.com/Salaries/data-quality-engineer-salary-SRCH_KO0,21.htm)
- [Salary.com — Data Governance Manager Salary](https://www.salary.com/research/salary/benchmark/data-governance-manager-salary)

Regulatory:
- [ECB — Guide on effective risk data aggregation and risk reporting (May 2024)](https://www.bankingsupervision.europa.eu/ecb/pub/pdf/ssm.supervisory_guides240503_riskreporting.en.pdf)
- [EY — Why BCBS 239 Compliance is essential in 2025](https://www.ey.com/en_nl/industries/banking-capital-markets/why-bcbs-239-compliance-is-essential-in-2025)
- [Capco — ECB Final Guidelines Complement BCBS 239](https://www.capco.com/intelligence/capco-intelligence/ecb-final-guidelines)
- [Soda — BCBS 239 and the RDARR Guide](https://soda.io/blog/bcbs-239-rdarr-data-quality-financial-services)
- [Collibra — BCBS 239: proving data integrity to regulators](https://www.collibra.com/blog/bcbs-239-explained-how-banks-can-prove-data-integrity-to-regulators)
- [Alation — Critical Data Elements for Financial Services](https://www.alation.com/blog/critical-data-elements-financial-services/)
- [OvalEdge — BCBS 239 Data Lineage 2026](https://www.ovaledge.com/blog/bcbs-239-data-lineage)

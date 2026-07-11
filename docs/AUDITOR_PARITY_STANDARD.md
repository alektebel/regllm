# APDQ: An Auditor-Parity Data Quality Standard for Banks

Design specification (v0 draft). Goal: define the industry standard —
and the reference tool — such that **any bank** can certify that its
data quality test suite detects **every issue any auditor could
detect from the data**, and prove that claim by execution rather than
assertion.

This document deliberately starts from the objections, because a
standard that cannot survive them is marketing.

---

## 1. The problem with "100% coverage" as currently built

The existing DQC eval stack (`DQC/eval/`) proves something narrower
than it appears to:

1. **Coverage is relative to our own defect catalog.** 67 defects,
   hand-written. An auditor's first question — *"why would 67 be all
   of them?"* — has no answer today. A bigger catalog does not fix
   this; only a *completeness argument* does (§3).
2. **Everything is coupled to one schema.** Defects, oracles,
   `generate_db.py`, the reward heuristics (`r_template`,
   `r_coherence`) and the attribution PK all assume
   `ciclos_calibrados`. "Any bank" is currently false by
   construction.
3. **The applicability map is one regulation, hand-suggested, at
   section granularity.** Sections are not testable units; atomic
   requirements are.
4. **A whole defect class is missing**: population completeness. An
   auditor's most basic move — reconcile the datamart's totals
   against the general ledger / CIRBE / FINREP to find *missing or
   fabricated rows* — has no counterpart in the catalog, which only
   mutates rows that exist.
5. **SQLite-only, single-table-biased scoring, no dialect story** for
   the Teradata/Oracle/SAS estates banks actually run.

None of this is fatal — the *certification mechanics* (mutation
testing, clean-by-construction twin, verifiable reward, no LLM judge)
are the right foundation and ahead of both vendors and academia. But
the mechanics currently certify one database against one catalog. The
standard below is what turns them into a claim about *any bank* and
*any auditor*.

---

## 2. What "any issue an auditor could detect" actually means

"Catch everything wrong with the data" is not achievable and no
auditor achieves it either. The achievable — and sufficient — bar is
**auditor parity**:

> An auditor working from (a) the data, (b) the bank's documentation
> of that data (dictionary, derivation rules, lineage), and (c) the
> applicable regulation, can only ever detect a defect that manifests
> as a violation of some proposition derivable from (a) ∪ (b) ∪ (c).

This is the load-bearing observation. Auditors do not have magic
access to ground truth; they have the same three inputs plus external
evidence (source documents, confirmations, appraisals). Therefore:

- **In scope (parity claim):** every defect detectable from data +
  documentation + regulation. This set can be characterized
  constructively (§3) and tested mechanically (§5).
- **Explicit residual (out of scope, stated like audit's "reasonable
  assurance"):** defects requiring *external evidence* — a collateral
  value that is consistently wrong everywhere because the appraisal
  was wrong; consistent falsification at origin (fraud that preserves
  all internal invariants); disputes over regulatory *interpretation*;
  judgment on estimates and overlays. No data-side tool can close
  these, and neither can an auditor without leaving the database. The
  standard must say so in its conformance statement, or it will be
  torn apart at the first QA session.

The product of the standard is therefore two lists: the **certified
detectable set** and the **declared residual** — both explicit, both
versioned.

## 3. The completeness argument (the core of the standard)

The claim "we detect everything in the detectable set" must not rest
on a catalog. It rests on **structural induction over the lineage
graph**, which is finite and known:

Model the reporting database as a DAG. Nodes are fields (and one
population node per table); edges are documented derivations.

**Obligations per node type:**

| Node type | Mandatory oracles |
|---|---|
| **Derived field** | *Recomputation*: re-derive the value from its parents using the documented formula; any mismatch fires. (This is what an auditor does when they "reperform the calculation".) |
| **Source field** | *Validity* (domain/format/nullability per dictionary), *reconciliation* against every other surface where the same fact is recorded (source system, CIRBE, AnaCredit, FINREP cell, collateral register), *plausibility* (distributional prior). |
| **Table population** | *Control totals & register reconciliation*: row counts and aggregates tie to an independent surface (GL, regulatory submission), catching missing and fabricated rows — the class the current catalog lacks. |
| **Cross-record structure** | Uniqueness of business keys, referential integrity, temporal monotonicity of lifecycle dates, panel consistency across snapshots. |
| **The documentation itself** | If a derived field has *no* documented formula, that is not "untestable" — it is a **finding by construction** (an auditor would write it up as one). Undocumented ⇒ non-conformant. Same for an applicable regulatory requirement bound to no field (§4). |

**The induction:** if every node meets its obligations, then any
single-point corruption (a wrong cell, a dropped row, a duplicated
row, a broken derivation) either violates its own node's oracle or
propagates into a recomputation mismatch at a descendant. Corruption
that violates *no* oracle must be consistent across every recorded
surface and every derivation — which is exactly the external-evidence
residual of §2 that no auditor detects from the data either.

This converts "trust our catalog" into "check our graph": the
completeness check is itself computable — walk the lineage DAG and
fail certification on any node missing an obligation. That is the
statement a regulator or external auditor can verify without trusting
us.

**Corollary that falls out for free:** the tool detects at *full
population* what human auditors verify by *sampling*. Auditor parity
is actually auditor dominance on the in-scope set; the residual is
where the human stays.

## 4. The defect taxonomy (normative)

Twelve classes. Finite, mapped to the node obligations above, and to
the vocabulary auditors already use (so the report reads as an audit
program, not a tool log). A conformant suite must demonstrate
sensitivity per class × applicable node:

| # | Class | Auditor's name for it | Oracle type |
|---|---|---|---|
| 1 | Missing value | Completeness exception | validity |
| 2 | Domain/format violation | Validity exception | validity |
| 3 | Duplicate business key | Occurrence/uniqueness | structure |
| 4 | Broken reference | Existence | structure |
| 5 | Intra-row incoherence | Internal consistency | coherence |
| 6 | Derivation error | Reperformance mismatch | recomputation |
| 7 | Cross-surface mismatch | Reconciliation difference | reconciliation |
| 8 | Missing/fabricated rows | Completeness of population | control totals |
| 9 | Temporal impossibility | Cut-off / sequence | structure |
| 10 | Panel inconsistency | Period-on-period consistency | recomputation over snapshots |
| 11 | Distributional anomaly | Analytical review exception | statistical |
| 12 | Semantic drift | Change without authorization | code/definition diff (the SAS AST-diff engine is this oracle) |

Classes 1–10 and 12 are deterministic (zero false positives on a
clean twin, mandatory). Class 11 is probabilistic and *advisory* —
the standard must not let statistical monitors count toward the
certified set, because they cannot be certified for specificity. This
is also the clean dividing line against the observability vendors:
what they sell is class 11; the standard certifies 1–10 and 12.

## 5. Architecture: how one standard fits any bank

Five layers. The bank-specific part is exactly one artifact (the
binding), which is also the artifact auditors already review.

```
Regulation texts ──► R1 Requirement register (atomic, per-paragraph, versioned)
                              │
ECB BIRD input layer ──► R2 Canonical concept model (don't invent one)
                              │
Bank dictionary/lineage ──► R3 Binding manifest (concept ↔ column, formula ↔ code)
                              │
                     R4 Check IR (declarative rules over concepts)
                              │ compile per dialect
                     SQL / SAS / PySpark against the bank's estate
                              │
                     R5 Conformance harness (synthetic twin + mutation corpus)
```

- **R1 Requirement register.** Each regulation (GL/2017/16, Anejo IX,
  CRR art. 174, RDARR, EBA validation rules) is decomposed into
  *atomic, testable requirements*: one row = one obligation +
  paragraph ref + referenced concepts + defect classes it implies.
  LLM-extracted, human-signed, hash-pinned to the regulation text so
  amendments (e.g. Circular 1/2025) automatically re-open affected
  rows. This replaces section-level `applicability.yaml`.
- **R2 Canonical concepts.** Use the **ECB BIRD input layer** as the
  concept vocabulary (exposure, instrument, protection, default date,
  …) instead of inventing an ontology. BIRD is open, maintained by
  the ECB, already maps to FINREP/AnaCredit, and IReF (confirmed
  2026, go-live Q4 2029) will push every EU bank to map to it anyway
  — the standard rides a mapping banks must produce regardless.
- **R3 Binding manifest.** The bank maps concepts to columns and
  documented formulas to code locations. Three properties make this
  the keystone: (i) an applicable requirement whose concepts are
  unbound is a **gap by construction** — the tool detects the *absence*
  of data an auditor would ask for; (ii) the binding is reviewable by
  a human without reading any generated SQL; (iii) it is the only
  per-bank artifact, so certification effort scales with schema size,
  not with rule count.
- **R4 Check IR.** Rules are written once, against concepts, in a
  small declarative language typed by defect class (a recomputation
  rule *must* name a formula; a reconciliation rule *must* name two
  surfaces). Compiled to the bank's dialect. Generated SQL is an
  artifact, never the source of truth.
- **R5 Conformance harness.** The generalization of
  `eval_harness.py` + `generate_db.py`: from the binding manifest and
  the documented formulas, build a **synthetic twin** that satisfies
  every documented invariant by construction, then plant k mutations
  per (defect class × node) and demand: 100% recall on planted
  mutations, 0 firings on the clean twin, no overbroad checks
  (confusion-matrix discipline), decoys for coherence honesty. All of
  this already exists in the repo for one schema; the work is making
  it *derive from the manifest* instead of being hand-coded.

**Where the LLM sits — and where it must not.** LLMs do the expensive
transcription work: extracting atomic requirements from regulation
text, proposing bindings, transcribing formulas out of SAS/COBOL into
the IR, proposing rules. Every LLM output lands in a human-signed,
mutation-verified artifact. The certificate never depends on trusting
a model — that is the difference between this and every "GenAI rule
suggestion" feature on the market, and it is what makes the output
regulator-grade.

## 6. Certification protocol (what "conformant" means)

Two certificates, deliberately analogous to how compilers and
security standards are certified:

**Level T — tool conformance** (any implementation, ours or a
competitor's): pass the *reference bank* — a published synthetic
institution (schema + dictionary + binding + regulation register)
with a published mutation corpus covering all 12 classes. 100%
recall, 100% specificity, no overbroad checks. This is what makes
APDQ a *standard* rather than a product: the conformance suite is the
spec, like a compiler test suite or the OWASP ASVS checklists. It is
also honest self-pressure: our own tool must pass a corpus we don't
control the shape of.

**Level B — bank deployment conformance**: at a given institution,
the certificate is the tuple:

```
1. Requirement register: 0 unsigned rows, regulation hashes current
2. Binding manifest:     0 applicable concepts unbound (or signed waiver each)
3. Lineage obligations:  every node carries its §3 oracles (computed, not asserted)
4. Twin conformance:     recall = 1.0 per class × node, specificity = 1.0,
                         0 overbroad, 0 decoy over-claims
5. Production run:       findings ledger with per-finding requirement ref
```

Items 1–4 are machine-checkable by a third party from the artifacts
alone; item 5 is the operational output. That bundle *is* the audit
pack — it answers the ECB RDARR inspector, the external auditor and
internal validation with the same evidence, and it is regenerable on
every regulation amendment or schema change.

**Versioning discipline:** certificates pin (regulation hash, binding
hash, IR ruleset hash, twin seed). Any input changing voids the
affected cells only — re-certification is incremental, which is what
makes quarterly EBA validation-rule updates and yearly circular
amendments an automated diff instead of a new consulting engagement.

## 7. Making it *the* industry standard, not our product

A standard nobody else can implement is a product with pretensions.
The play, in order:

1. **Publish the spec + reference bank + mutation corpus openly**
   (the equivalent of ASVS / a conformance test suite). The moat is
   not the spec — it is the reference implementation, the SAS
   transcription tooling, and certification services.
2. **Anchor on BIRD/IReF now.** Between 2026 and the 2029 IReF
   go-live, every EU bank must build concept mappings anyway. A
   standard whose binding layer *is* the BIRD mapping gets adopted as
   a by-product of mandatory work. This window is the strategic
   timing argument.
3. **Recruit the people whose job it makes cheaper**: external audit
   firms (they can certify Level B instead of sampling), consultancies
   (white-label Level B delivery), and one SI validation department as
   design partner whose IMI/SREP response uses the pack.
4. **Seek supervisory acknowledgment, not endorsement.** Realistic
   milestone: an ECB/BdE inspection accepting the Level B pack as
   evidence for a finding's remediation. That single precedent is the
   adoption event.

## 8. Gap plan: from this repo to reference implementation

Ordered by dependency, each item ends in a verifiable state:

| # | Work | Turns | Into | Done when |
|---|---|---|---|---|
| 1 | Defect **classes**, not instances | `defect_catalog.py` (67 hand-written) | Mutation generators parameterized by (class × node type), instantiated from a binding manifest | Same 67 defects regenerate from the manifest for the current schema; selftest still 100% |
| 2 | Lineage-obligation checker | implicit design knowledge | Executable §3 walk over the DAG | CI fails on any node missing an oracle; current schema passes |
| 3 | Requirement register | section-level `applicability.yaml` | Atomic rows, regulation-hash pinned, multi-corpus namespaces | GL/2017/16 re-expressed atomically; 0 unsigned rows |
| 4 | Binding manifest + BIRD concepts | hard-coded schema knowledge | R3 artifact; unbound-concept gap detection | A *second* toy schema certifies end-to-end with no Python edits |
| 5 | Manifest-driven twin generator | hand-coded `generate_db.py` | Clean-by-construction twin from declared formulas | Twin for the second schema passes selftest |
| 6 | Check IR + dialect compilers | raw SQL scoring with single-table heuristics | Typed rules over concepts; SQLite first, one enterprise dialect second | Reward computed on IR, not string heuristics |
| 7 | Population-completeness class | absent | Control-totals oracles + row-deletion/fabrication mutations | Class 8 planted and caught on both schemas |
| 8 | Reference bank + published corpus | private eval | Level T conformance suite | An external implementation could run it from the docs alone |

Item 4 is the crux — it is the moment the tool stops being about one
database. Items 1–3 are prerequisites; 5–8 are the standard's public
face. The SAS AST engine slots in at R3 (formula transcription out of
legacy code) and as the class-12 oracle — that asset is unique in
this market and should be treated as such.

## 9. Honest limits (to be printed in every certificate)

- External-evidence defects: values consistently wrong at origin
  (bad appraisal, mis-keyed contract confirmed nowhere else).
- Coordinated falsification preserving all internal invariants.
- Regulatory interpretation and estimation judgment (overlays,
  forward-looking adjustments) — flaggable, not certifiable.
- Class 11 (distributional) findings are advisory by design.
- The certificate covers the bound scope; concepts waived in the
  binding manifest are enumerated on its face.

An auditor reading this list should recognize it: it is the standard
scope-limitation language of an audit opinion. That is intentional —
parity includes being honest the way the profession is.

---

## References

- [ECB — Banks' Integrated Reporting Dictionary (BIRD)](https://www.ecb.europa.eu/stats/ecb_statistics/reporting/bird/html/index.en.html)
- [BIRD portal — model, transformation & validation rules](https://bird.ecb.europa.eu/)
- [EBF — ECB confirmation of IReF implementation & milestones (June 2026)](https://www.ebf.eu/ebf-media-centre/updates/ebf-welcomes-the-ecb-confirmation-of-the-implementation-of-the-integrated-reporting-framework-iref/)
- [EY — How banks should prepare for BIRD and IReF](https://www.ey.com/en_gl/insights/financial-services/emeia/how-banks-should-start-preparing-for-bird-and-iref)
- [ECB — Guide on effective risk data aggregation and risk reporting (RDARR)](https://www.bankingsupervision.europa.eu/ecb/pub/pdf/ssm.supervisory_guides240503_riskreporting.en.pdf)
- [EBA — Reporting frameworks (DPM & validation rules)](https://www.eba.europa.eu/risk-and-data-analysis/reporting/reporting-frameworks)
- Companions: [`DATA_QUALITY_INDUSTRY_SOTA_2026.md`](DATA_QUALITY_INDUSTRY_SOTA_2026.md), [`DATA_QUALITY_SPAIN_NICHE.md`](DATA_QUALITY_SPAIN_NICHE.md)

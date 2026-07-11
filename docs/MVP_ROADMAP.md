# APDQ Build Plan: MVP → Product

Companion to [`AUDITOR_PARITY_STANDARD.md`](AUDITOR_PARITY_STANDARD.md).
Ordered build list with acceptance gates.

## MVP

**Goal:** one real pilot. A bank (or a consultancy on behalf of one)
hands us a data dictionary + a data extract; we hand back a proven
check suite + audit pack for EBA GL/2017/16, generated and certified
end-to-end. Corresponds to items 1–5 of the standard's gap plan (§8).

| # | Build | Gate ("done when") |
|---|---|---|
| 1 | **Binding manifest format** — YAML: column → concept, formula, regulation refs. Mini concept list (~40 PD/LGD concepts), not full BIRD yet. | Manifest for the current eval schema exists and validates. |
| 2 | **Defect-class mutation generators** — refactor `defect_catalog.py`'s 67 instances into generators for the 12 normative classes, instantiated from a manifest. | The 67 current defects regenerate from the manifest; `eval_harness.py --selftest` still 100%. |
| 3 | **Manifest-driven twin generator** — replace hand-coded `generate_db.py` with a generator driven by declared formulas. Restrict the formula language (arithmetic, min/max, case-when, lookups) to keep it tractable. | A **second** toy schema certifies end-to-end with zero Python edits. |
| 4 | **Lineage-obligation checker** — executable walk of the lineage DAG; every derived field needs a recomputation oracle, every source field validity+reconciliation, every table control totals. | CI fails on any node missing an oracle; current schema passes. |
| 5 | **Requirement register (GL/2017/16)** — atomize `applicability.yaml` into one row per testable obligation, hash-pinned to the regulation text. | 0 unsigned rows; hash change flips affected rows to pending. |
| 6 | **Audit pack generator** — one command emits HTML/PDF + JSON: register state, manifest completeness, graph-walk result, twin recall/specificity, findings ledger. | Pack regenerates deterministically from pinned inputs. |
| 7 | **LLM assist loop** — existing generation/RAG proposes bindings, formulas and checks into a human-sign queue; nothing unsigned enters the certificate. | Proposal → review → signed artifact flow works on the pilot schema. |

**MVP scope cuts:** SQLite/CSV extracts only; one regulation; no
dialect compilers; class 11 (statistical) excluded; class 12
(semantic drift) demoed via the existing SAS AST differ only.

**Pilot motion:** 4–6 weeks with one SI validation department or one
consultancy (white-label). Success = the pack is used in a real
supervisory or internal-validation conversation.

## Larger product (post-pilot, rough order)

1. Full **BIRD input-layer** concept model as the binding vocabulary.
2. **Dialect compilers**: Teradata, Oracle, SAS datasets, Spark.
3. Proper typed **check IR** (rules typed by defect class) replacing
   string-heuristic scoring.
4. **More corpora**: Anejo IX / Circular 4/2017 (highest Spanish
   demand) → EBA COREP/FINREP validation-rule import → CIRBE/AnaCredit
   cross-register reconciliation.
5. **Population-completeness class** wired to general-ledger control
   totals on real extracts.
6. **Incremental re-certification**: any regulation/binding/ruleset
   hash change voids only affected cells; quarterly EBA updates and
   circular amendments become automated diffs.
7. **Production surface**: scheduler, findings ledger UI, ownership
   and remediation workflow.
8. **On-prem / local-LLM packaging** (GGUF/Ollama path) — the
   data-egress selling point vs. US SaaS.
9. **Public Level T conformance suite**: reference synthetic bank +
   mutation corpus + spec, published openly.
10. **SAS→IR transcription tooling** as a paid migration accelerator.
11. **Certification program** for audit firms and consultancies
    (they certify Level B deployments; they become distribution).

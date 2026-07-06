# Production-grade DQCs — how we get there

> Companion to `guarantees.md`, which covers what the stress DB + harness
> *already* guarantee. This doc covers the gap between "passes the eval" and
> "production grade".

## Reframe: "guarantee" = converging independent evidence

No single layer makes a DQC production-grade — each can be fooled in a
different way. Production grade is **defense in depth**: a DQC must survive
*several independent* checks whose weaknesses don't overlap. The harness is
one such layer; it closes some gaps and leaves others open.

## The stack — each row closes a specific harness gap

| harness gap (`guarantees.md` §5) | production layer that closes it | repo state today |
|---|---|---|
| reward not enforced | **mandatory Gate 2**: auto-score every emitted DQC, auto-reject low-reward before human review | `reward` column exists; `/dqc/generate` does not call `score_check` |
| synthetic distributions → real false positives uncaught | **real-data clean-zero backtest**: DQC must return 0 rows on a real clean warehouse snapshot | `recuperatory_cycles.csv` (22K rows) exists; harness not run on it |
| catalog-bounded (only 26 defect types) | **catalog expansion from real incidents + red-team**: seed defects from historical bugs; humans invent novel ones | incident notes (`data/experience/bug_*.md`) were removed — need re-sourcing |
| citations can be invented | **machine-checked regulatory provenance**: every `referencia_regulatoria` validated against the regulation graph | `data/regulation/graph.json` (225 articles) exists; not checked |
| cross-table DQCs unscoreable | **multi-table mode**: materialize `contratos`/`basilea_mensual` so JOIN-based DQCs parse & score | not built |
| not a production sign-off | **human gate with accountability**: HIGH-severity DQCs need named risk-officer sign-off | validate/reject UI exists; open-loop, no quorum/accountability |
| no notion of behavior in prod | **drift monitoring (closed loop)**: track each DQC's fire-rate over time; alert on PSI / volume jumps (BCBS 239 P5) | none |
| regressions slip in | **regression gate in CI**: a new model version must not lower recall on any dimension | harness exists; not in CI |
| model self-approval | **independent validation** (SR 11-7 / TRIM): a separate team audits the evidence pack | process gap |

## Operational definition: a DQC is "production-grade" when ALL are true

1. Verifiable reward ≥ threshold (parse + coherence + clean-zero + catches +
   specificity) on the stress DB.
2. Returns 0 rows on a **real** clean warehouse snapshot (no synthetic-only
   false-negative-on-edge-cases).
3. Catches ≥ 1 row on a **real** dirty snapshot OR a reviewed synthetic trap
   (proven sensitivity).
4. `referencia_regulatoria` resolves to a real article in `graph.json`.
5. Human-validated by a named owner; HIGH-severity DQCs block the reporting
   submission.
6. Versioned, with a rollback path, and its fire-rate is monitored in prod.

The harness today delivers #1 and half of #3. It is **necessary, not
sufficient**.

## Highest-leverage next steps (both feasible now)

1. **Wire Gate 2** — call `score_check` inside `/dqc/generate`, store the
   reward, auto-reject `< 0.5` before the DQC reaches the UI. Turns the
   harness from offline evaluator into a production gate.
2. **Real-data clean-zero backtest** — point the harness at
   `recuperatory_cycles.csv` so every DQC must prove zero false positives on
   real data, not just synthetic.

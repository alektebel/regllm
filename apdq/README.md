# APDQ — Auditor-Parity Data Quality (reference implementation, MVP)

Executable MVP of the standard specified in
[`docs/AUDITOR_PARITY_STANDARD.md`](../docs/AUDITOR_PARITY_STANDARD.md),
built to the plan in [`docs/MVP_ROADMAP.md`](../docs/MVP_ROADMAP.md).
The consolidated reference — proposal, market, regulation technicalities
and implementation in one document — is
[`docs/APDQ_HANDBOOK.md`](../docs/APDQ_HANDBOOK.md).

One artifact — a **binding manifest** describing a reporting schema —
drives everything: a clean-by-construction synthetic **twin**, mutation
generators + oracles for the normative **defect classes**, the
**lineage-obligation** completeness check, and a **certification run**
that emits a machine-verifiable **audit pack**. A hash-pinned
**requirement register** ties the whole thing to regulation paragraphs.
No LLM sits anywhere in the trust chain.

## Quickstart

```bash
# validate a manifest and inspect obligations / generated defects
python -m apdq validate apdq/examples/mini_ciclos/manifest.yaml
python -m apdq lineage  apdq/examples/mini_ciclos/manifest.yaml
python -m apdq defects  apdq/examples/mini_ciclos/manifest.yaml

# full Level-B certification (exit code gates CI)
python -m apdq certify apdq/examples/mini_ciclos/manifest.yaml \
    --register apdq/examples/mini_ciclos/register.yaml --out audit/

# the zero-code-edit proof: a different schema, same command
python -m apdq certify apdq/examples/retail_mortgages/manifest.yaml

# materialize a clean twin for inspection; draft a manifest from any
# SQLite extract (the human-sign queue an LLM may prefill)
python -m apdq twin    apdq/examples/mini_ciclos/manifest.yaml -o twin.db
python -m apdq propose extract.db --name mybank -o draft_manifest.yaml

pytest tests/test_apdq.py          # 39 regression tests
```

Both example schemas certify: every generated oracle returns 0 rows on
the clean twin (specificity) and catches all `k` planted mutations per
defect (recall), per class and per node.

## Module map

| Module | Role (standard §) |
|---|---|
| `expr.py` | Restricted formula language: parse → Python eval (twin) + SQL compile (oracles). One AST, two backends, tested to agree. |
| `manifest.py` | Binding manifest loader/validator (§5 R3). Refuses derived columns without formulas, unknown concepts, circular derivations. |
| `concepts/pd_lgd.yaml` | Mini canonical vocabulary (~50 PD/LGD concepts, BIRD hints) (§5 R2). |
| `twin.py` | Manifest-driven clean-by-construction twin (§5 R5): domains sampled, orderings sorted into compliance, constraints rejection-sampled, formulas evaluated, surfaces mirrored, control totals computed. |
| `defects.py` | Mutation generators + oracles per defect class × node (§4). |
| `lineage.py` | Executable completeness walk (§3): every node has its oracles or a signed waiver. |
| `register.py` | Atomic requirement register, SHA-256-pinned to regulation text; gates: unsigned / stale / unbound (§5 R1). |
| `harness.py` | Certification protocol (§6): lineage gate, register gate, specificity, recall, mixed-run confusion/overlap/overbroad. |
| `audit_pack.py` | JSON + HTML evidence bundle, reproducible from pinned inputs. |
| `assist.py` | Proposal queue — the only place LLM output enters, always landing as unsigned drafts (§5 "where the LLM sits"). |

## Defect classes in this MVP

| # | Class | Status |
|---|---|---|
| 1 | missing value | generic (PK-null variant is aggregate: a NULL key cannot be attributed by key) |
| 2 | domain violation | generic (range + enum) |
| 3 | duplicate key | generic (verbatim re-insert, true duplicates) |
| 4 | broken reference | generic (per declared FK) |
| 5 | intra-row constraint | generic (auto-search for violating values; `plant:` hints when search can't, e.g. multi-column violations) |
| 6 | derivation error | generic (reperformance oracle compiled from the formula) |
| 7 | reconciliation mismatch | generic (per declared surface) |
| 8 | population (control totals) | generic (missing + fabricated rows vs count/sum ties) |
| 9 | temporal ordering | generic (per declared date chain) |
| 10 | panel inconsistency | generic (`panel:` block → period-gap, duplicate-period, decreasing-cumulative oracles per declared cumulative column) |
| 11 | distributional | **advisory by design** — never counts toward certification (expansion E6) |
| 12 | semantic drift | served by the SAS AST differ (`src/sas_logic_tree.py`); integration is expansion E5 |

Mixed-run note: class-6 oracles legitimately fire on other classes'
planted rows — upstream corruption propagating into a recomputation
mismatch is the completeness argument working, so they are exempt from
the *overbroad* smell (which still applies to every other class). A
temporal (class 9) oracle listed as overbroad on a schema with date
domains is usually the same phenomenon: a date planted beyond its domain
is also out of order.

## Manifest format (reference)

```yaml
apdq_manifest: 1
name: mybank_irb
tables:
  - name: fact_table
    primary_key: ID
    rows: 400                      # twin size hint
    control:                       # class 8 — or waivers: {control: reason}
      surface: gl_totals
      checks: [{kind: count}, {kind: sum, column: EAD}]
    foreign_keys:                  # class 4
      - {column: PARENT_ID, ref_table: parent, ref_column: PARENT_ID}
    date_orderings:                # class 9
      - [DATE_OPENED, DATE_DEFAULT, DATE_CLOSED]
    constraints:                   # class 5
      - id: stage3_dpd
        expr: "STAGE != 3 or DPD >= 90"
        plant: {STAGE: 3, DPD: 5}  # explicit violating values (optional)
    columns:
      - name: EAD                  # source: classes 1, 2, 7
        concept: exposure_at_default
        role: source
        domain: {type: real, min: 0, max: 1000000}   # int|real|text|yyyymm
        reconcile:
          - {surface: basilea, column: EAD_SRC, join_column: ID}
      - name: ECL                  # derived: class 6
        concept: expected_credit_loss
        role: derived
        formula: "PD * LGD * EAD"  # expr.py grammar; undocumented = refused
        regulation_refs: ["Anejo IX"]
      - name: LGD_REALIZADA        # CONDITIONALLY derived: formula binds
        concept: lgd_realized      # only where `when` holds; elsewhere the
        role: derived              # value is free and sampled from domain
        formula: "max(0, min(1, 1 - (RECUP - COSTE) / EAD))"
        when: "ESTADO = 'CERRADO'"
        domain: {type: real, min: 0, max: 0.95}
      - name: FECHA_CIERRE         # dates are ISO strings; nullable columns
        concept: cycle_close_date  # can declare their twin null frequency
        role: source
        domain: {type: date, min: "2024-02-01", max: "2024-12-28",
                 nullable: true, null_rate: 0.5}
        waivers: {reconcile: "workout system out of scope (signed: J.Doe)"}
      - name: MODEL_ONLY           # gaps must be signed, never silent
        concept: pd_estimate
        role: source
        domain: {type: real, min: 0, max: 1}
        waivers: {reconcile: "model output, reperformed downstream (signed: J.Doe)"}

  - name: monthly_panel            # class 10 — snapshot semantics
    primary_key: ID_PANEL
    rows: 240                      # ≈ rows/periods distinct series
    panel:
      series_key: FACT_ID          # FK ⇒ series keys drawn from the parent
      period_column: MES           # consecutive months in the twin
      periods: 8
      cumulative_columns: [RECUP_ACUM]   # non-decreasing within a series
    ...
```

Language notes: formulas support `isnull(X)` / `null()` (conditional
completeness: "a closed cycle must carry a closure date") and
`yyyymm(D)` (date↔period consistency, `MES = yyyymm(FECHA)` as a
derivation instead of an unsatisfiable sampling constraint). Two loader
rules protect the clean-twin guarantee: `date_orderings` domains must be
monotone along each chain (the twin sorts values into compliance, and
non-monotone domains would let the sort push a value out of range), and
`reconcile.join_column` must be the table's primary key (surfaces are
auto-mirrored per row; joining real external tables on other keys is
expansion E10). Constraint auto-violation searches single columns and
pairs; anything wider needs a `plant:` hint.

Register rows (see `examples/mini_ciclos/register.yaml`): one atomic
obligation per row, `text_sha256` pinned via `register.text_hash`
(whitespace/case-normalized so PDF reflows don't void pins), `status:
signed` requires `signed_by`, and signed rows' concepts must be bound by
the manifest or certification fails with the unbound list.

## Expansions (larger product) — where each one plugs in

Numbered against [`docs/MVP_ROADMAP.md`](../docs/MVP_ROADMAP.md) §"Larger
product"; each names the seam in this codebase.

- **E1 — Full BIRD vocabulary.** Replace `concepts/pd_lgd.yaml` with the
  generated BIRD input layer; `manifest.py` already resolves any
  vocabulary via `concepts_file:`. Add a concept-level crosswalk table
  (ours → BIRD id) so existing manifests migrate mechanically.
- **E2 — Generic panel class (10). ✅ DONE** (`panel:` block; see the
  format reference above and `examples/ciclos_full`). Remaining
  refinement: an `allow_decrease_when:` clause so cure events can
  legitimately reset DPD/stage (the D62/D63 note in `crosswalk.yaml`).
- **E3 — Dialect compilers.** `expr.to_sql` is the single SQL emission
  point; add `dialect=` (Teradata/Oracle/Spark) there and in the oracle
  string templates in `defects.py`. Certification still runs on the
  SQLite twin; production execution uses the dialect output.
- **E4 — Cross-table formulas.** `manifest._check_dependencies` currently
  restricts formulas to same-row columns; extend the grammar with
  `lookup(table, column)` joined over a declared FK, and extend
  `twin._make_row` to resolve parent lookups (parents generate first
  already — `_topo_tables`).
- **E5 — Semantic drift (12) integration.** Wire `src/sas_logic_tree.py`
  AST-diff output into the pack: a formula changed in code but not in the
  manifest ⇒ stale binding, same mechanics as a stale register pin.
- **E6 — Advisory distributional layer (11).** Separate runner emitting
  *advisory* findings only; must never touch `Certification.certified`
  (the clean line vs. observability vendors).
- **E7 — LLM proposal integration.** Point the DQC agent
  (`api/routers/dqc.py` + regulation RAG) at `assist.propose_manifest`
  drafts and at register-row extraction; everything lands `pending` and
  `TODO` — `manifest.py`/`register.py` structurally refuse unreviewed
  input, which is the whole point.
- **E8 — More corpora.** One register file per regulation (Anejo IX next,
  then EBA COREP/FINREP validation rules imported mechanically, then
  CIRBE/AnaCredit as `reconcile:` surfaces across registers);
  `register.load_register` already namespaces by file.
- **E9 — Incremental re-certification.** The pack pins manifest + register
  + seed; add input-hash comparison against the previous pack to re-run
  only affected (table × class) cells; quarterly EBA updates become diffs.
- **E10 — Findings ledger + scheduler.** Run generated oracles against
  *production extracts* (not the twin) on a schedule; each firing links
  its defect id → requirement id → paragraph. The oracle SQL in the pack
  is already production-runnable.
- **E11 — Level T conformance suite.** Publish a reference bank (a frozen
  manifest + twin + mutation corpus) that *other* implementations must
  pass — `examples/` is the seed of it.

## Catalog parity (the gap-plan item 2 gate)

`examples/ciclos_full/manifest.yaml` binds the DQC eval schema at
production width (contract master + 60-column fact table + monthly
panel + basilea/colaterales surfaces + GL control totals) and certifies:
**168 generated defects across all ten generic classes, specificity and
recall 100%**. `examples/ciclos_full/crosswalk.yaml` maps every one of
the 67 hand-written catalog entries (D01–D66 + decoys) onto a generated
defect or an explicit documented exception, and
`tests/test_apdq.py::test_crosswalk_is_total` enforces that the mapping
stays total. Current tally: **58 mapped, 7 partial (documented
simplification), 3 excluded with reasons** (D44 needs a string-prefix
predicate; DA/DB are decoys, whose honesty role the overbroad-oracle
check plays here).

## Honest status (what this implementation does not yet do)

- Three crosswalk `partial`s simplify the original rule (2-band PD
  monotonicity, YYYYMM-granular staleness, cure-unconditional panel
  monotonicity) and D44 is excluded — each with the fix named in
  `crosswalk.yaml`.
- Constraint auto-violation search covers single columns and pairs;
  wider violations need `plant:` hints (by design — hints are
  reviewable).
- Reconciliation surfaces are auto-mirrored per row and keyed by the
  fact table's primary key; binding a *real* second system's extract as
  the surface (true cross-register joins) is E10 territory.
- SQLite only (E3), same-row formulas only (E4), and the class 11/12
  caveats in the table above.

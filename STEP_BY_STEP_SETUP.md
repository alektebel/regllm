# STEP BY STEP: first APDQ iteration with a client schema

Goal of this run: starting from **a client's existing DQC set and their
database schema**, produce a **proven round of data-quality tests**
covering the four rubric families you asked for, plus the audit pack
that proves each test works:

| Your rubric | APDQ defect classes | Where it comes from in the manifest |
|---|---|---|
| **Format** | 1 (missing value), 2 (domain violation) | each column's `domain:` (type, min/max, enum, nullable) |
| **Reperformance** | 6 (derivation error) | each derived column's `formula:` (recompute and compare) |
| **Coherence between fields** | 5 (intra-row constraints), 7 (reconciliation across tables/surfaces) | `constraints:` and `reconcile:` |
| **Contract / cycle behaviour** | 4 (referential integrity), 9 (lifecycle date order), 10 (monthly panel), 3 (duplicates), 8 (population vs control totals) | `foreign_keys:`, `date_orderings:`, `panel:`, `primary_key`, `control:` |

Time budget for a first iteration: **~1 day** if you have a data
dictionary, of which the manifest-filling step is 80%.

---

## Phase 0 — environment (10 minutes)

```bash
# 0.1  clone and enter
git clone https://github.com/alektebel/regllm && cd regllm

# 0.2  python 3.10+ and the two dependencies the certifier needs
python --version                  # must be >= 3.10
pip install pyyaml pytest

# 0.3  sanity check: the shipped full example must certify
python -m apdq certify apdq/examples/ciclos_full/manifest.yaml
#      expected last line:  VERDICT : CERTIFIED

# 0.4  regression suite must be green
python -m pytest tests/test_apdq.py -q     # 46 passed
```

No GPU, no AWS, no LLM is needed for anything in phases 0–6. Models
enter only in Phase 3, optionally (see Phase 7 for the Bedrock/local
decision).

## Phase 1 — collect the inputs (checklist)

From the client, in any format:

- [ ] **Schema**: DDL, or a data dictionary (column name → meaning →
      type), for every table in scope. Minimum viable: the main fact
      table + its contract/master table.
- [ ] **Their existing DQC set**: the SQL (or SAS) checks they run
      today. You will mine these — each existing check is evidence of
      an invariant (a formula, a range, a cross-field rule).
- [ ] **Derivation documentation**: how each computed field is
      calculated (methodology doc, or the SAS/SQL code that computes
      it). Undocumented derivations become findings, which is correct.
- [ ] Optional but very useful: **a small extract** (even 100 rows,
      even anonymized, even an *empty* database created from the DDL)
      as a SQLite file or CSVs.
- [ ] Optional: which source systems / registers each field should
      agree with (for `reconcile:`), and what control totals exist
      (GL, FINREP cell, row counts).

**Rule for the whole exercise: real client data never touches an LLM,
and does not need to** — the pipeline only ever executes SQL against it.

## Phase 2 — draft the manifest skeleton (30 minutes)

Create your working folder:

```bash
mkdir -p apdq/clients/<CLIENT>          # e.g. apdq/clients/bancox
```

**Path A — you have an extract or DDL.** Load it into SQLite if it is
not already (CSV → SQLite is one `pandas.DataFrame.to_sql` per file, or
`sqlite3` `.import`), then:

```bash
python -m apdq propose extract.db --name <CLIENT> \
       -o apdq/clients/<CLIENT>/manifest.yaml
```

This introspects every table, guesses each column's type/bounds/enums
from the data, and leaves every judgment as an explicit `TODO`.

**Path B — you only have a dictionary document.** Copy the full worked
example and gut it:

```bash
cp apdq/examples/ciclos_full/manifest.yaml apdq/clients/<CLIENT>/manifest.yaml
# keep the structure, replace tables/columns with the client's
```

Set `name:` to the client codename, and set each table's `rows:`
(synthetic twin size — 300–500 is plenty for a first iteration).

## Phase 3 — fill the manifest (the real work, ~½ day)

Work table by table, column by column. For **every column** decide:

**3.1 Bind a concept.** Pick the closest id from
`apdq/concepts/pd_lgd.yaml` (open it — ~60 PD/LGD concepts, bilingual).
If nothing fits, add a new concept to that file (one line: id + EN/ES
names). Unknown concepts fail validation on purpose.

**3.2 Decide the role — this is where your rubric gets built:**

- **Source column** (typed in from upstream) → give it a `domain:`.
  This generates the **format** tests:
  ```yaml
  - name: EAD_TOTAL_INFORMADO
    concept: exposure_at_default
    role: source
    domain: {type: real, min: 0, max: 100000000}      # class 2
    # types: int | real | text | yyyymm | date("YYYY-MM-DD" strings)
    # enums: values: [CORP, SME, ...]   nullability: nullable + null_rate
  ```
- **Derived column** (computed from others) → give it the `formula:`.
  This generates the **reperformance** tests:
  ```yaml
  - name: ECL
    concept: expected_credit_loss
    role: derived
    formula: "PD_FINAL * LGD_FINAL * EAD_TOTAL"       # class 6
    regulation_refs: ["Anejo IX"]
  ```
  Formula language: `+ - * /`, comparisons, `and/or/not`,
  `min max abs sqrt round if(cond,a,b) isnull(X) null() yyyymm(date)`.
  Same-row references only.
- **Conditionally derived** (formula holds only in one state — e.g.
  realised LGD once the cycle closes) → `when:` + a `domain:` for the
  free branch:
  ```yaml
  - name: LGD_REALIZADA
    concept: lgd_realized
    role: derived
    formula: "max(0, min(1, 1 - (RECUP - COSTES) / EAD_TOTAL))"
    when: "ESTADO_CICLO = 'CERRADO'"
    domain: {type: real, min: 0, max: 1}
  ```

**3.3 Mine the client's existing DQC set.** Translate each of their
checks into a manifest declaration rather than keeping it as SQL:
a check `WHERE X <> A*B` ⇒ `X` is derived with `formula: "A * B"`;
a check `WHERE X NOT IN (...)` ⇒ enum domain; a check joining two
tables on equality ⇒ a `reconcile:` or a constraint. Keep a two-column
scratch list (their check → your declaration) — it becomes the gap
report in Phase 6.

**3.4 Declare the coherence rules** (table level → **coherence**
tests):

```yaml
constraints:
  - id: stage3_requiere_mora
    expr: "STAGE_IFRS9 != 3 or DPDS >= 90"            # class 5
    description: "stage 3 requires >= 90 days past due"
    regulation_refs: ["CRR Art. 178"]
    plant: {STAGE_IFRS9: 3, DPDS: 5}   # explicit violating values —
                                       # add whenever the rule spans
                                       # several columns
```

And, where the same fact exists on another surface, per column:

```yaml
    reconcile:                                        # class 7
      - {surface: basilea_mensual, column: EAD_SRC, join_column: <PK>}
        # join_column MUST be the table's primary key (v1 limitation)
```

**3.5 Declare the contract/cycle behaviour** (table level):

```yaml
primary_key: ID_CONTR_CICLO_LGD                       # class 3 (dups)
foreign_keys:                                         # class 4
  - {column: ID_CONTRATO, ref_table: contratos, ref_column: ID_CONTRATO}
date_orderings:                                       # class 9
  - [FECHA_ALTA, FECHA_DEFAULT, FECHA_ADJUDICACION, FECHA_CIERRE]
    # rule: min/max of each column's domain must be >= its predecessor's
control:                                              # class 8
  surface: gl_totales
  checks: [{kind: count}, {kind: sum, column: EAD_TOTAL}]
panel:                                                # class 10 — only
  series_key: ID_CONTR_CICLO_LGD                      # for monthly-
  period_column: MES                                  # snapshot tables
  periods: 12
  cumulative_columns: [RECUPERACION_ACUM]
```

**3.6 Waive what you can't cover yet — never delete it.** Every source
column without a `reconcile:` and every table without `control:` needs
a signed waiver, or certification fails:

```yaml
waivers: {reconcile: "model output, reperformed downstream (signed: <your name>)"}
```

The waiver list prints on the audit pack — for a first iteration it
doubles as your findings/backlog list for the client.

## Phase 4 — the validate → certify loop (1–2 hours of iteration)

```bash
cd <repo root>
# 4.1 shape + concepts + formulas parse?
python -m apdq validate apdq/clients/<CLIENT>/manifest.yaml
# 4.2 every field carrying its obligations (or waiver)?
python -m apdq lineage  apdq/clients/<CLIENT>/manifest.yaml
# 4.3 full certification
python -m apdq certify  apdq/clients/<CLIENT>/manifest.yaml \
       --out apdq/clients/<CLIENT>/audit
```

Fix errors in the order they appear; the common ones:

| Error / failure | Fix |
|---|---|
| `unknown concept` | add it to `apdq/concepts/pd_lgd.yaml` or pick an existing id |
| `derived column requires a formula` | document the derivation (that's the point) or re-role as source |
| `needs monotone domains` on a date chain | widen/order the min/max of the chained date columns so each starts and ends no earlier than its predecessor |
| `could not satisfy declared constraints` | a constraint is nearly impossible under the sampled domains — narrow the domains, or convert the rule into a derivation (`when:`/`formula:`), which is usually the truer model |
| `cannot auto-violate constraint` | add a `plant:` hint with explicit violating values |
| `FAIL C0x:... clean_rows=N` (specificity) | your declaration contradicts another one — the harness is telling you two documented rules disagree; reconcile them |
| verdict CERTIFIED but long waiver list | fine for iteration 1 — that list is the client conversation |

Stop when you see `VERDICT : CERTIFIED`.

## Phase 5 — export the round of tests and run it on client data

```bash
# 5.1 the deliverable: every generated check as runnable SQL,
#     tagged with defect id, class and regulation refs
python -m apdq defects apdq/clients/<CLIENT>/manifest.yaml \
       --sql apdq/clients/<CLIENT>/checks.sql
```

Each statement returns **violating rows** (empty result = pass). To run
the whole suite against a client extract loaded in SQLite:

```bash
python - <<'EOF'
import sqlite3, re
from apdq.expr import register_sql_functions
conn = sqlite3.connect("extract.db"); register_sql_functions(conn)
sql = open("apdq/clients/<CLIENT>/checks.sql").read()
for block in sql.split(";"):
    lines = [l for l in block.strip().splitlines() if l.strip()]
    if not lines or all(l.startswith("--") for l in lines):
        continue
    header = next((l[3:] for l in lines if l.startswith("-- C")), "?")
    query = "\n".join(l for l in lines if not l.startswith("--"))
    try:
        n = len(conn.execute(query).fetchmany(1000))
        print(f"{'PASS' if n == 0 else f'FAIL {n:>4} rows'}  {header}")
    except sqlite3.Error as e:
        print(f"SKIP ({e})  {header}")   # e.g. surface table absent
EOF
```

Notes: reconciliation/control checks reference surface tables
(`basilea_mensual`, `gl_totales`, …) — if the client extract doesn't
include those, they SKIP; that's expected in iteration 1. The SQL is
SQLite dialect; running natively on Oracle/Teradata needs the dialect
compiler (roadmap E3) or light manual editing.

## Phase 6 — the gap report vs. the client's DQC (1 hour)

Three lists, straight from artifacts you already have:

1. **Coverage they lack**: generated checks (from `checks.sql`
   headers) with no counterpart in the client's DQC set.
2. **Checks of theirs that map** to a generated defect (your Phase 3.3
   scratch list) — now each is *proven* by mutation instead of trusted.
3. **Checks of theirs you couldn't express** — either an invariant you
   missed (add to the manifest, re-run Phase 4) or a check that catches
   nothing (a finding about their suite).

## Phase 7 — deliverables & the model question

**Hand to the client:** `checks.sql` (the test round), the audit pack
(`audit/<CLIENT>_audit_pack.html` + `.json` — proof each check fires on
its planted defect and stays silent on clean data), the waiver list
(scope + backlog), and the Phase 6 gap report.

**Bedrock vs local models** — only Phase 3 can use a model at all
(pre-filling manifest TODOs from the dictionary and their DQC set):

- Schema/dictionary shareable under NDA → **AWS Bedrock (Claude, EU
  region)** for the one-off drafting; best transcription quality,
  fastest iteration. The repo's DQC agent already runs on Bedrock.
- Client forbids cloud → the repo's **local GGUF/Ollama path**; expect
  more manual cleanup, which `validate` will catch anyway.
- Either way: proposals land as `TODO`s, a human signs everything, the
  certification path never consults a model, and **real data never
  goes to any model** — it is only ever queried with SQL.

## Optional finishing touches

- **Requirement register** (regulatory traceability): copy
  `apdq/examples/mini_ciclos/register.yaml`, adapt rows to the client's
  applicable paragraphs, hash-pin with
  `python -c "from apdq.register import text_hash; print(text_hash('<paragraph text>'))"`,
  and certify with `--register`.
- **CI gate**: add
  `python -m apdq certify apdq/clients/<CLIENT>/manifest.yaml --register ...`
  to the client repo's pipeline — non-zero exit blocks the merge.
- Deeper reading: [`docs/APDQ_HANDBOOK.md`](docs/APDQ_HANDBOOK.md)
  (everything), [`apdq/README.md`](apdq/README.md) (format reference).

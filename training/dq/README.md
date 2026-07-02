# DQ Check Generation — RL pipeline + validation UI

> Status as of 2 Jul 2026. Trains Qwen3-8B (QLoRA + GRPO) to emit
> portable SQL checks that detect **database-coherence** violations in
> the IRB recovery-cycles warehouse, then surfaces them in a validation
> UI that produces a single UNION-ALL dashboard query.

---

## What "database coherence" means here

The relational layer of DQ — predicates across **≥ 2 fields/entities**
whose violation means a row is internally contradictory. Distinct from:

| Layer | Example | Owned by |
|---|---|---|
| Pipeline correctness | `ECL ≠ PD·LGD·EAD` (bug in SAS step) | existing `training/rl_env.py` |
| Regulatory range | `PD_ESTIMADA < 0.0003` | future PD/LGD-article checks |
| Schema validity | `contratos.cliente_id` NOT NULL | DDL CHECK constraints |
| **Coherence** | `ADJUDICACION_VALOR > 0 AND ADJUDICACION_FLAG = '0'` | **this module** |

Defining property of every coherence rule: the WHERE predicate combines
≥ 2 columns. A 1-column range check is **not** coherence and scores
`r_coherence = 0` in the reward (see below).

---

## Architecture

```
Rule cards (9)          ──┐
                          ├──► prompt ──► Qwen3-8B ──► completion
Toy DB (22K rows)        │                              │
                          │                              ▼
                          │                       extract <check>{sql}
                          │                              │
                          │              ┌───────────────┴───────────────┐
                          │              ▼                               ▼
                          │     run on clean DB               run on trap DB
                          │     (r_clean_zero)                 (r_catches)
                          └── 5-component reward ──► GRPO trainer (TRL)
```

The trained model is intended to slot into the existing `/dqc/generate`
endpoint (currently driven by LLM + RAG) so that the Angular UI produces
real coherence checks at inference time.

### Repository layout (files added/modified this iteration)

| Path | Role | LoC |
|---|---|---:|
| `training/dq/coherence_rules.py` | 9 `CoherenceRule` cards + `mutate(row)` per rule + `load_toy_db()` | 241 |
| `training/dq/dq_reward.py` | 5-component verifiable reward + `extract_check()` from raw completion | 324 |
| `training/dq/dq_env.py` | `build_prompt(rule)` + `generate_batch(n)` | 88 |
| `training/dq/overfit_dq.py` | GRPO overfit harness (sibling of `overfit_single.py`) | 195 |
| `training/dq/checks_db.py` | SQLite persistence + `build_dashboard_query()` UNION ALL | 252 |
| `api/routers/dqc.py` | Added `/checks`, `/checks/{id}/status`, `/dashboard`; auto-persists generated DQCs | +180 |
| `DQC/app/src/app/validate/{validate.component}.{ts,html,css}` | Validation + dashboard UI | 465 |
| `DQC/app/src/app/app.component.ts` | Tab toggle (Chat ↔ Validar) | 45 |
| `DQC/app/src/app/models/dqc.model.ts` | `CheckRecord`, `DashboardResponse`, … | +52 |
| `DQC/app/src/app/services/dqc.service.ts` | `counts()`, `list()`, `setStatus()`, `dashboard()` | +25 |

---

## The 9 rules

All empirically verified to return 0 violations on
`data/samples/recuperatory_cycles.csv` (22 363 rows) — `r_clean_zero`
is achievable for each.

| rule_id | cols | severity | visible | notes |
|---|---|---|---|---|
| `dq_coh_adjudicacion_valor_sin_flag` | 2 | HIGH | yes | **user's original example**; the only rule GRPO has been run on |
| `dq_coh_adjudicacion_flag_sin_valor` | 2 | MED | yes | inverse |
| `dq_coh_adjudicacion_valor_sin_tipo` | 2 | HIGH | yes | |
| `dq_coh_adjudicacion_tipo_sin_valor` | 2 | MED | yes | |
| `dq_coh_stage3_sin_dpd_suficiente` | 2 | HIGH | **no** (oculto) | CRR Art. 178.1(b) |
| `dq_coh_recuperacion_mayor_exposicion` | 3 | MED | yes | sum-invariant |
| `dq_coh_cerrado_sin_terminacion` | 2 | HIGH | yes | |
| `dq_decoy_lgd_realizada_negativa` | 1 | MED | yes | single-column decoy (should score `r_coherence=0`) |
| `dq_decoy_ead_cero` | 1 | LOW | yes | single-column decoy |

---

## Reward (5 components, all program-executed, no LLM judge)

| Component | Weight | How measured |
|---|---:|---|
| `r_parse` | 0.15 | SQLite `EXPLAIN` succeeds on the emitted SQL |
| `r_template` | 0.20 | SELECT + FROM-known-table + WHERE; valid severity/category |
| `r_coherence` | 0.20 | WHERE references ≥ 2 columns OR query has a JOIN |
| `r_clean_zero` | 0.20 | Returns 0 rows on the coherent 22K-row toy DB |
| `r_catches` | 0.25 | Returns ≥ 1 row on a DB with one injected mutation |

Weighted sum shifted to `[-1, 1]` for GRPO (matches
`training/rl_env.py:776`). Mutation testing via `rule.mutate(row)` is
the key trick — it proves the check catches what it claims.

### Discrimination sanity checks (all passing)

| Input query | r_coherence | r_clean_zero | r_catches | total |
|---|---:|---:|---:|---:|
| correct: `… WHERE valor>0 AND flag='0'` | 1 | 1 | 1 | **1.0** |
| lazy: `… WHERE valor>0` (1 column) | 0 | 0 | 1 | 0.2 |
| tautology: `… WHERE valor>=0 OR flag IS NOT NULL` | 1 | 0 | 1 | 0.4 |
| unparseable: `SELECT FROM WHERE (((( ` | 0 | 0 | 0 | −1.0 |
| wrong table: `FROM contratos …` | 1 | — | — | template halved |

---

## Empirical state

### GRPO overfit on `dq_coh_adjudicacion_valor_sin_flag` (Qwen3-8B, QLoRA r=32)

Two smoke runs, identical config except completion budget:

| Metric | 384 budget | **768 budget** |
|---|---:|---:|
| GRPO steps | 12 | 12 |
| Completions scoring r=1.0 | 1 / 24 (4%) | **13 / 24 (54%)** |
| Steps at `mean_reward=1.0` | 1 / 12 | **10 / 12** |
| First non-zero loss | never | **step 5 (`0.093`), step 7 (`0.009`)** |
| Peak VRAM | 10.2 GB | 11.6 GB |
| Headroom on 16 GB | 6.4 GB | **5.0 GB** |
| Per-step | 38 s | 63 s |

The split-reward steps (5, 7) are the ones that actually train — one
completion emitted valid JSON within budget and scored +1, the other
clipped mid-`<think>` and scored −1, giving GRPO a real advantage
signal. The 384 budget was starvation, not a fundamental limit;
**768 is the default in `overfit_dq.py`**.

### Verification status

- ✅ reward signal discriminates good/lazy/broken queries (programmatic tests)
- ✅ `checks_db` accepts both Spanish (LLM) and English (RL) severity tokens
- ✅ idempotent re-insert by `(rule_id, sql)` — re-running inference won't bloat the table
- ✅ `build_dashboard_query()` runs against coherent data → 0 violations
- ✅ FastAPI endpoints (`/dqc/checks`, `/checks/{id}/status`, `/checks/counts`, `/dashboard`) return correct shapes + 404 on unknown id
- ✅ Angular production build clean (`npx ng build`, 1.6 s, no errors)
- ✅ end-to-end GRPO loop runs to completion on the 8B at 11.6 GB peak

### Verification gaps (what we do NOT know yet)

- ❌ Base-model r=1.0 rate on the **other 8 rules**. ADJUDICACION is the
  easiest (2 cols, simple predicate). The 3-column
  `dq_coh_recuperacion_mayor_exposicion` and the regulatory
  `dq_coh_stage3_sin_dpd_suficiente` may produce **zero** r=1.0
  completions from the untrained model → no GRPO signal → those rules
  won't train.
- ❌ No multi-rule training has been attempted. Single-rule overfit
  does not validate curriculum / mixing.
- ❌ Trained LoRA is not wired into `/dqc/generate` yet — the UI still
  uses the old LLM+RAG path.

---

## How to run

### Smoke (no GPU required)

```bash
# Reward discrimination
.venv/bin/python -c "
from training.dq.dq_env import get_rule
from training.dq.dq_reward import score_check
rule = get_rule('dq_coh_adjudicacion_valor_sin_flag')
print(score_check({'sql': 'SELECT ID_CONTR_CICLO_LGD FROM recuperatory_cycles WHERE ADJUDICACION_VALOR > 0 AND ADJUDICACION_FLAG = \"0\"', 'severity':'HIGH','category':'cross_field'}, rule).total)
"

# Checks-db lifecycle + dashboard SQL
.venv/bin/python - << 'PY'
from training.dq import checks_db
conn = checks_db.connect()
# ... insert_check, set_status, build_dashboard_query
PY
```

### GRPO overfit (one rule, ~12 min on 8B)

```bash
.venv/bin/python training/dq/overfit_dq.py                       # defaults: Qwen3-8B, 768 budget, ADJUDICACION rule
.venv/bin/python training/dq/overfit_dq.py --rule-id dq_coh_cerrado_sin_terminacion
```

### Validation UI (full stack)

```bash
# Terminal 1 — API on 8001 (per DQC/app/proxy.conf.json)
.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2 — Angular
cd DQC/app && npm start                                            # http://localhost:4200
```

The Chat tab generates DQCs (auto-persisted as `pending`). The Validar
tab lists/validates/rejects them; the dashboard section unlocks with
copy-all + copy-UNION-ALL once no visible pending checks remain.

---

## Open decision: pre-flight before full training

**The blocking question:** will the untrained 8B produce ≥ 1 r=1.0
completion per GRPO group on each of the 9 rules? If yes on a rule,
GRPO will train it. If no on a rule, GRPO has no signal there and that
rule needs an SFT warmup or curriculum ordering.

**Cheap pre-flight (~10 min, no training):** generate ~4 completions
per rule from the untrained model, score each, tabulate r=1.0 rate per
rule. Worth doing before committing to a 4–5 h full training run.

---

## DQC catalog eval harness (LLM full-catalog generation)

Separate from single-check GRPO: evaluates whether the LLM produces a
**complete JSON catalog** of DQCs with ID, description, and
`campos_entrada` — targeting **100% field coverage** and **field
coherence** (declared fields match SQL, multi-field rules combine ≥ 2
columns).

| File | Role |
|---|---|
| `catalog_schema.py` | Field universe (39 auditable cols), JSON parse, template |
| `catalog_reward.py` | 5-component score: parse, schema, coverage, coherence, groups |
| `catalog_prompt.py` | System/user prompt for full-catalog generation |
| `build_catalog_eval_manifest.py` | Writes `data/catalog_eval_manifest.jsonl` |
| `dq_catalog_eval.py` | CLI runner — score file or `--generate` via LLM |

### Metrics

| Component | Weight | Meaning |
|---:|---:|---|
| `r_coverage` | 0.35 | Union of `campos_entrada` / auditable fields (target **1.0**) |
| `r_coherence` | 0.30 | Per-DQC: valid fields, SQL↔field alignment, multi-field tipo |
| `r_schema` | 0.15 | Every entry has `dqc_id`, `descripcion`, `campos_entrada` |
| `r_groups` | 0.10 | Known cross-field groups (from `coherence_rules`) covered |
| `r_parse` | 0.10 | Valid catalog JSON extracted |

### Run

```bash
# Build manifest
PYTHONPATH=. .venv/bin/python training/dq/build_catalog_eval_manifest.py

# Score an existing catalog JSON
PYTHONPATH=. .venv/bin/python training/dq/dq_catalog_eval.py \\
  --catalog path/to/catalog.json

# Generate via LLM + score (needs configured LLM backend)
PYTHONPATH=. .venv/bin/python training/dq/dq_catalog_eval.py --generate \\
  --output training/dq/data/catalog_eval_results.json

# Unit tests
PYTHONPATH=. .venv/bin/python -m pytest tests/test_dq_catalog_reward.py -q
```

Expected catalog shape:

```json
{
  "dataset": "recuperatory_cycles",
  "table": "recuperatory_cycles",
  "dqcs": [
    {
      "dqc_id": "DQC_PD_ESTIMADA_001",
      "descripcion": "...",
      "campos_entrada": ["STAGE_IFRS9", "PD_ESTIMADA"],
      "tipo": "consistencia",
      "regla_sql": "SELECT ID_CONTR_CICLO_LGD FROM recuperatory_cycles WHERE ..."
    }
  ]
}
```

---

## Next-step options (pick one)

1. **Pre-flight + full training** — base-rate scan, then a multi-rule
   GRPO run over the catalog. ~10 min pre-flight + 4–5 h training.
2. **Wire the trained LoRA into `/dqc/generate`** — replace the LLM+RAG
   generation path with the trained model so the UI emits real
   coherence checks at inference time. Path to the user-facing MVP.
3. **Expand the rule catalog** — cross-table rules (`defaults ↔ contratos`,
   `colaterales ↔ clase_activo`) before training, for broader coverage.

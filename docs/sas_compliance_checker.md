# SAS Parameter Compliance Checker

Automated row-level regulatory compliance analysis of IRB/IFRS9 parameter tables using RAG + LLM.

## Problem

SAS databases contain credit risk parameter outputs (PD, LGD, EAD, CCF, staging results). Validating that each individual calculation complies with EBA/CRR requirements is currently manual, slow, and inconsistent.

## Goal

A pipeline that:
1. Reads a SAS parameter table row by row
2. Checks each row against the applicable regulatory rule (retrieved via RAG)
3. Emits a per-row compliance verdict with regulatory justification
4. Produces a final report summarising all findings

Fully on-premise — no data leaves the institution.

---

## Architecture

```
SAS Table (any size)
        │
        │  saspy / ODBC / CSV export
        ▼
┌─────────────────────────────────────────────────┐
│  Row Stream (sync, iterrows)                    │
│                                                 │
│  For each row:                                  │
│                                                 │
│  1. RAG retrieval                               │
│     Query: _RAG_QUERY_MAP[column] template      │
│     → returns EBA/CRR article chunks            │
│                                                 │
│  2. LLM call                                    │
│     Input:  [row values as JSON]                │
│           + [regulatory context from RAG]       │
│           + [running_memory.as_prompt_str()]    │
│     Output: {verdict, flags, articles, ...}     │
│                                                 │
│  3. Citation grounding                          │
│     → drop any article not in retrieved context │
│     (hallucination structurally impossible)     │
│                                                 │
│  4. Memory update                               │
│     - compliant row  → increment counter only   │
│     - flagged row    → append to findings       │
│     - findings > 20  → compress via LLM        │
│                                                 │
│  5. Collect RowVerdict                          │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Final Report                                   │
│  - rows_processed, rows_flagged, rows_uncertain │
│  - compliance_rate                              │
│  - findings[] (flagged + uncertain rows only)   │
│  - compressed_patterns (cross-row summary)      │
│  - narrative (LLM-generated, JSON)              │
└─────────────────────────────────────────────────┘
```

---

## Implementation Status

### Phase 1 — Core pipeline ✅ DONE

| Component | File | Status |
|---|---|---|
| ComplianceChecker | `src/compliance/checker.py` | ✅ |
| RunningMemory + self-compression | `src/compliance/memory.py` | ✅ |
| RAG query routing (`_RAG_QUERY_MAP`) | `src/compliance/checker.py` | ✅ |
| Citation grounding (anti-hallucination) | `src/compliance/checker.py` | ✅ |
| CLI runner | `scripts/run_compliance_check.py` | ✅ |
| Ollama backend | `checker.py:_call_ollama` | ✅ |
| Groq backend | `checker.py:_call_groq` | ✅ |
| JSON report output | `checker.py:_build_report` | ✅ |

**Design note:** The original spec included a hard-rule pre-filter (`HARD_RULES` dict). The implementation intentionally dropped it in favour of pure RAG-grounded diagnosis — the LLM applies the regulatory floors/ceilings from the retrieved text rather than hardcoded lambdas. This makes the checker resilient to regulatory changes without code edits.

---

## The Running Memory (Functionally Infinite)

Bounded at ~1K tokens regardless of table size. Lives in `src/compliance/memory.py`.

```python
RunningMemory:
    rows_processed: int       # total rows seen
    rows_flagged:   int       # total non-compliant
    findings: list[Finding]   # only recent uncompressed findings
    patterns: str             # LLM-written summary of compressed findings
    compress_threshold: int   # default 20
```

When `len(findings) >= compress_threshold`:
→ LLM compresses `findings` + `patterns` into a new `patterns` string (≤ 200 words)
→ `findings` is cleared, counters preserved
→ Memory stays bounded; no information loss on flagged items

---

## Per-Row LLM Prompt

```
[SYSTEM] Eres un experto en validación regulatoria IRB/IFRS9...
         Responde en JSON: {verdict, resumen, desarrollo, flags, articles, advertencias}
         REGLA: "articles" solo puede contener refs presentes en el CONTEXTO REGULATORIO.

[USER]   CONTEXTO REGULATORIO:
         {rag_retrieved_chunks}

         MEMORIA ACUMULADA:
         {running_memory.as_prompt_str()}

         FILA A DIAGNOSTICAR (id={row_id}):
         {row_as_json}
```

---

## RAG Query Routing

`checker.py:_RAG_QUERY_MAP` maps canonical column keys (PD, LGD, EAD, CCF, MATURITY, STAGE, ECL, RWA, K, RATING, SEGMENT, DEFAULT, CURE, MARGIN, DOWNTURN) to targeted Spanish regulatory queries. Up to 3 relevant queries are joined with `|` to create a multi-parameter retrieval.

Column names are matched via `column_map` (user-supplied) or by uppercasing the column name directly.

---

## LLM Backends

| Backend | Model | Where | Speed | Privacy |
|---------|-------|-------|-------|---------|
| **Ollama** (default) | `qwen2.5:14b-instruct-q4_K_M` | On-prem | ~3–5s/row | Data stays local |
| **Groq** | `llama-3.3-70b-versatile` | API | ~0.5s/row | Data leaves institution |

Local Ollama is the required default for bank data.

---

## SAS Connection

```python
# Option A: saspy (requires SAS on same server)
import saspy
sas = saspy.SASsession()
df = sas.sasdata('MYLIB.PD_ESTIMATES').to_df()

# Option B: ODBC (any machine with SAS ODBC driver)
import pyodbc
conn = pyodbc.connect('DSN=SAS_ODBC;...')
df = pd.read_sql("SELECT * FROM MYLIB.PD_ESTIMATES", conn)

# Option C: export → CSV/parquet (simplest, implemented in CLI)
df = pd.read_csv("pd_estimates.csv")
df = pd.read_parquet("pd_estimates.parquet")
```

---

## CLI Usage

```bash
# CSV with local Ollama (default)
python scripts/run_compliance_check.py pd_estimates.csv --table MYLIB.PD_ESTIMATES

# Groq backend
python scripts/run_compliance_check.py pd_estimates.csv --backend groq

# Map CSV column names to canonical regulatory keys
python scripts/run_compliance_check.py data.csv --col-map PD_ESTIMATE:PD LGD_12M:LGD

# Without DB/RAG (no DB connection needed)
python scripts/run_compliance_check.py data.csv --no-rag

# Full options
python scripts/run_compliance_check.py data.csv \
  --table MYLIB.PD_ESTIMATES \
  --output report.json \
  --backend ollama \
  --ollama-model qwen2.5:14b-instruct-q4_K_M \
  --row-id-col OBLIGATION_ID \
  --col-map PD_ESTIMATE:PD LGD_FINAL:LGD EAD_FINAL:EAD \
  --compress-threshold 20 \
  --rag-results 5
```

---

## Output Format

```json
{
  "table": "MYLIB.PD_ESTIMATES",
  "run_date": "2026-05-21",
  "rows_processed": 45231,
  "rows_flagged": 312,
  "rows_uncertain": 47,
  "compliance_rate": 0.9931,
  "llm_calls": 312,
  "compressed_patterns": "Se detectaron PDs por debajo del suelo regulatorio...",
  "findings": [
    {
      "row_id": "OBL_00412",
      "verdict": "flagged",
      "resumen": "PD de 0.015% por debajo del suelo regulatorio de 0.03%",
      "flags": ["PD inferior al mínimo CRR Art. 160(1)"],
      "articles": ["CRR Art. 160(1)"],
      "rag_sources": ["CRR_Reglamento_575_2013.pdf"],
      "advertencias": null,
      "row_data": {"PD": 0.00015, "SEGMENT": "CORP", ...}
    }
  ],
  "narrative": {
    "resumen": "...",
    "desarrollo": "...",
    "articulos": ["CRR Art. 160", "EBA/GL/2017/16 §8.3"],
    "advertencias": "..."
  }
}
```

---

## RAG System Improvements

The compliance checker inherits the existing pgvector RAG system (`src/rag_system.py`). The following improvements are needed to raise retrieval precision for compliance queries.

### Current state

- `paraphrase-multilingual-mpnet-base-v2` embeddings in pgvector
- Hybrid search: semantic cosine + BM25 (in-memory, cosine weight 0.7)
- BM25 index built over full corpus at init time

### Identified weaknesses

1. **BM25 index lives in RAM** — rebuilt from DB every restart; no persistence. For 20 docs (~50K chunks) this is acceptable but will break at scale.

2. **Single vector per chunk** — no parent-child retrieval. A 200-token chunk may lack the surrounding legal context needed to assess a PD floor correctly.

3. **Retrieval is query-level, not column-level** — the `_RAG_QUERY_MAP` in `checker.py` is a workaround; ideally the RAG system knows which article hierarchy each chunk belongs to.

4. **No cross-encoder reranking** — hybrid scores are linear combinations. A cross-encoder (`ms-marco-MiniLM-L-6-v2`) would rerank the top-20 candidates using full query-document attention.

### Improvement plan

#### RAG-1: Cross-encoder reranking (high impact, low effort)

Add an optional `rerank=True` flag to `buscar_hibrida`. Retrieve `n_resultados * 4` candidates, rerank with a cross-encoder, return top `n_resultados`.

```python
# src/rag_system.py addition
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def buscar_hibrida(self, pregunta, n_resultados=5, rerank=False, ...):
    candidates = self._hybrid_candidates(pregunta, n=n_resultados * 4)
    if rerank and self.reranker:
        pairs = [(pregunta, c["texto"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        candidates = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
    return candidates[:n_resultados]
```

Cross-encoder models are small (~80 MB) and run on CPU in ~50 ms per batch.

#### RAG-2: Parent-child chunk retrieval (high impact, medium effort)

Store `parent_chunk_id` in `document_chunks` metadata. Retrieve child chunks (short, ~100 tokens) for precision but return the parent chunk (~400 tokens) as context to the LLM.

Schema addition:
```sql
ALTER TABLE document_chunks ADD COLUMN parent_chunk_id TEXT;
```

`formatear_contexto` would then fetch the parent text for each matched child, giving the LLM a full article paragraph instead of a sentence fragment.

#### RAG-3: Article-level metadata index (medium impact, medium effort)

Add a separate `regulatory_articles` table:
```sql
CREATE TABLE regulatory_articles (
    article_id   TEXT PRIMARY KEY,  -- e.g. "CRR:160:1"
    documento    TEXT,
    articulo     TEXT,
    paragrafo    TEXT,
    full_text    TEXT,
    embedding    VECTOR(768)
);
```

`CitationRAG` (already built, used in `chat_engine.py`) already partially does this. The compliance checker should use it for a second retrieval pass: first retrieve chunks by query, then expand with full article text from `regulatory_articles`.

#### RAG-4: Persist BM25 index (low impact, low effort)

Serialize the BM25 corpus to disk (pickle/joblib) so it survives restarts without a full DB scan.

```python
# On init, load if cache exists; rebuild and save otherwise
_BM25_CACHE = "/tmp/bm25_cache.pkl"
```

---

## Phase 2 — Scale & Robustness

### 2a. Async/parallel row processing

Current `run()` is sequential (`iterrows`). For tables > 10K rows, process rows in async batches.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def run_async(self, df, table_name, row_id_col=None, workers=4):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = [
            loop.run_in_executor(pool, self.diagnose_row, row_id, row)
            for row_id, row in self._iter_rows(df, row_id_col)
        ]
        for task in asyncio.as_completed(tasks):
            verdict = await task
            self._all_verdicts.append(verdict)
            self._update_memory(verdict)  # memory updates must be serialised
```

**Note:** Memory updates (`_memory.add_finding`) must remain single-threaded to avoid race conditions. Use a lock or process memory updates in a dedicated collector coroutine.

### 2b. Checkpoint / resume

For very large tables, save progress to disk so a crash doesn't restart from zero.

```python
# checker.py addition
def run(self, df, table_name, checkpoint_path=None, ...):
    start_idx = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        start_idx = ckpt["rows_processed"]
        self._memory = RunningMemory(**ckpt["memory"])
        self._all_verdicts = [RowVerdict(**v) for v in ckpt["verdicts"]]
        logger.info("Resumed from checkpoint at row %d", start_idx)

    for i, (row_id, row) in enumerate(self._iter_rows(df, row_id_col)):
        if i < start_idx:
            continue
        verdict = self.diagnose_row(row_id, row)
        self._all_verdicts.append(verdict)
        if checkpoint_path and (i + 1) % 100 == 0:
            self._save_checkpoint(checkpoint_path)
```

Checkpoint file format: `{rows_processed, memory: {…}, verdicts: [{…}]}`.

### 2c. Stratified sampling for very large tables

For tables > 500K rows, run hard-rule pre-filter (pure Python, no LLM) on all rows, then stratified sample only the ambiguous ones for LLM diagnosis.

```python
HARD_RULES = {
    "PD":  [("floor", lambda v: float(v) >= 0.0003, "CRR Art. 160(1): PD ≥ 0.03%")],
    "LGD": [("floor_unsecured", lambda v: float(v) >= 0.45, "CRR Art. 161(1)(a)")],
    "EAD": [("non_negative", lambda v: float(v) >= 0, "EAD cannot be negative")],
    "CCF": [("range", lambda v: 0 <= float(v) <= 1.0, "CCF must be 0–100%")],
}

def hard_rule_scan(df, column_map) -> tuple[pd.DataFrame, list[RowVerdict]]:
    """Returns (ambiguous_rows_df, obvious_violations_as_verdicts)."""
    ...
```

Stratified sample for LLM: sample proportionally from each `(segment, rating_grade, vintage)` bucket, targeting ~5K rows regardless of table size.

### 2d. Schema auto-detection

Infer column semantics from column names without a `--col-map` argument.

```python
_COLUMN_ALIASES = {
    r"^pd[_\-]": "PD",
    r"lgd": "LGD",
    r"ead|exposure": "EAD",
    r"ccf|conv": "CCF",
    r"stage|etapa": "STAGE",
    r"ecl|provision": "ECL",
    r"rating|grade|calif": "RATING",
    r"default|incumpl": "DEFAULT",
    r"mat(urity)?": "MATURITY",
    r"segment|cartera": "SEGMENT",
}

def auto_detect_columns(df_columns) -> dict[str, str]:
    ...
```

---

## Phase 3 — Fine-tuned Model

The base LLM (Qwen2.5-14B or similar) is a general-purpose model. A LoRA fine-tune on compliance-labelled IRB data would improve precision, reduce hallucination, and allow a smaller model (7B) to match 14B quality.

### 3a. Dataset creation

- Label ~500 rows manually: `{row_json, rag_context, verdict, flags, articles}`
- Use existing checker output as a starting point — review flagged rows with a regulatory expert
- Augment with synthetic variations (scale PD values, swap segments, invert compliance status)
- Target: 500 real + 1500 synthetic = 2000 examples

### 3b. Fine-tuning

Follow the same LoRA pipeline already used in this repo (`scripts/finetune_*.py`).

```bash
python scripts/finetune_compliance.py \
  --model qwen2.5:7b \
  --data data/compliance_labelled.jsonl \
  --output models/compliance_lora \
  --epochs 3 --lora-r 16
```

### 3c. Evaluation

Metrics on a held-out 20% split:
- **Verdict accuracy** (compliant / flagged / uncertain)
- **Flag precision/recall** against expert labels
- **Citation hallucination rate** (articles not in retrieved context)

Compare: base 14B vs. fine-tuned 7B vs. RAG-only (no LLM).

### 3d. Deployment

Replace `ollama_model` default with the fine-tuned model served through Ollama's Modelfile system.

---

## Phase 4 — API Endpoint & UI

### 4a. FastAPI endpoint

Add `api/routers/compliance.py`:

```python
POST /compliance/run
  Body: {
    "table_name": "MYLIB.PD_ESTIMATES",
    "rows": [...],          # array of row dicts
    "column_map": {...},    # optional
    "backend": "ollama"
  }
  Response: 202 Accepted + {"job_id": "..."}

GET /compliance/jobs/{job_id}
  Response: {"status": "running|done|failed", "progress": 0.72, "report": {...}}
```

Jobs run in a background thread pool; results stored in PostgreSQL (`compliance_runs` table).

Schema:
```sql
CREATE TABLE compliance_runs (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name   TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    finished_at  TIMESTAMPTZ,
    status       TEXT DEFAULT 'pending',
    rows_total   INT,
    rows_done    INT DEFAULT 0,
    report       JSONB
);
```

### 4b. Frontend integration

Add a "Compliance" tab to the existing Next.js frontend:
- Upload CSV / paste table name
- Column mapping UI (auto-detect with manual override)
- Live progress bar (poll `GET /compliance/jobs/{id}` every 2s)
- Report viewer: compliance rate gauge, findings table, narrative panel

### 4c. PDF report generation

Use `reportlab` or `weasyprint` to generate a formatted PDF from the JSON report.

```python
GET /compliance/jobs/{job_id}/pdf
  Response: application/pdf
```

Include: executive summary, compliance rate chart, findings table, regulatory references, methodology note.

---

## Scalability Reference

| Table size | Strategy | LLM calls | Estimated runtime (14B local) |
|---|---|---|---|
| < 1K rows | All rows through LLM | ~1K | ~1–2h |
| 1K–10K | All rows through LLM | ~10K | ~10–15h |
| 10K–500K | Hard-rule pre-filter → LLM on ambiguous 5–20% | ~10–50K | ~1–2 days |
| > 500K | Hard-rule full scan + stratified sample 5K for LLM | ~5K | ~6h |

For large tables, stratified sampling by `(segment, rating_grade, vintage)` ensures every meaningful sub-population is represented.

---

## Key Dependencies

```
pandas              # table handling (already installed)
ollama              # local LLM serving
sentence-transformers  # embeddings (already installed)
psycopg2 + pgvector    # RAG retrieval (already installed)
cross-encoder          # for RAG-1 reranking (sentence-transformers includes this)
saspy / pyodbc         # SAS connection (install on-prem only)
reportlab / weasyprint # PDF generation (Phase 4)
```

---

## Open Questions

1. **What tables?** — PD estimates, LGD curves, EAD/CCF, IFRS9 staging, stress parameters?
2. **What SAS environment?** — SAS 9.4, SAS Viya, or SAS output files only?
3. **Segment structure?** — retail vs corporate vs sovereign, collateral types? (needed for `_RAG_QUERY_MAP` refinement)
4. **Reference period?** — point-in-time vs through-the-cycle matters for EBA GL application
5. **Standalone tool or integrated into regllm.xyz?** — current code is integrated; a separate repo makes sense for on-prem bank deployments without the chat frontend
6. **Hard-rule pre-filter scope** — should it cover only numerical bounds (PD floor, CCF range) or also structural checks (segment membership, required field presence)?

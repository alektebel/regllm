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
        │  saspy / ODBC
        ▼
┌─────────────────────────────────────────────────┐
│  Row Stream (async, batched)                    │
│                                                 │
│  For each row:                                  │
│                                                 │
│  1. RAG retrieval                               │
│     Query: "regulation governing {column_name}" │
│     → returns EBA/CRR article text              │
│                                                 │
│  2. LLM call                                    │
│     Input:  [row values]                        │
│           + [regulatory context]                │
│           + [running_memory]                    │
│     Output: {verdict, flags, memory_update}     │
│                                                 │
│  3. Memory management                           │
│     - compliant row  → increment counter only   │
│     - flagged row    → append to memory         │
│     - memory > limit → compress (LLM call)      │
│                                                 │
│  4. Store per-row verdict                       │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Final Report                                   │
│  - total rows processed                         │
│  - compliance rate per column / segment         │
│  - flagged rows with regulatory citation        │
│  - LLM-generated narrative (structured JSON)    │
└─────────────────────────────────────────────────┘
```

---

## The Running Memory (Functionally Infinite)

The key data structure. Bounded at ~1K tokens regardless of table size.

```python
memory = {
    "rows_processed": 0,
    "rows_flagged": 0,
    "findings": [
        # Only notable entries — compliant rows add nothing here
        {
            "row_id": "...",
            "column": "PD",
            "value": 0.0001,
            "flag": "Below regulatory floor (0.03%)",
            "article": "CRR Art. 160(1)"
        },
        ...
    ],
    "patterns": ""  # LLM-written summary of cross-row patterns seen so far
}
```

When `len(findings) > threshold`:
→ LLM compresses `findings` + `patterns` into a shorter `patterns` string
→ `findings` is cleared, counter preserved
→ Memory stays bounded, no information loss on flagged items

---

## Per-Row LLM Prompt

```
Eres un validador de parámetros de riesgo de crédito bajo CRR/EBA.

CONTEXTO REGULATORIO:
{rag_retrieved_chunk}

MEMORIA ACUMULADA:
{running_memory}

FILA A ANALIZAR:
{row_as_json}

Responde en JSON:
{
  "verdict": "compliant" | "flagged" | "uncertain",
  "flags": ["descripción del problema si lo hay"],
  "articles": ["CRR Art. X", "EBA/GL/YYYY/ZZ §N"],
  "memory_update": "texto breve para añadir a memoria, o null si es conforme"
}
```

---

## Models

### RAG retrieval
- Same pgvector + hybrid BM25 as regllm.xyz
- Query is constructed from column names + table schema context
- e.g. `"requisitos mínimos PD método IRB exposiciones corporativas"`

### LLM (two options)

| Option | Model | Where | Speed | Cost |
|--------|-------|-------|-------|------|
| **Local** | Qwen2.5-7B (fine-tuned LoRA) | On-prem via Ollama | ~2s/row | Free |
| **Cloud** | `llama-3.1-8b-instant` via Groq | API | ~0.3s/row | ~$0.05/1K rows |

Local is the right default — bank data cannot leave the institution.

---

## SAS Connection

```python
# Option A: saspy (requires SAS installation on same server)
import saspy
sas = saspy.SASsession()
df = sas.sasdata('MYLIB.PD_ESTIMATES').to_df()

# Option B: ODBC (works from any machine with SAS ODBC driver)
import pyodbc
conn = pyodbc.connect('DSN=SAS_ODBC;...')
df = pd.read_sql("SELECT * FROM MYLIB.PD_ESTIMATES", conn)

# Option C: export SAS → CSV/parquet, load with pandas (simplest)
df = pd.read_csv("pd_estimates.csv")
```

---

## Regulatory Rules (Hard-coded, Fast Pre-filter)

Run these before the LLM to skip rows that are obviously fine or obviously wrong.
Reduces LLM calls significantly.

```python
HARD_RULES = {
    "PD": [
        ("floor", lambda v: v >= 0.0003, "CRR Art. 160(1): PD ≥ 0.03%"),
        ("ceiling", lambda v: v <= 1.0,  "PD cannot exceed 100%"),
    ],
    "LGD": [
        ("floor_unsecured", lambda v: v >= 0.45, "CRR Art. 161(1)(a): LGD ≥ 45% unsecured senior"),
        ("floor_subordinated", lambda v: v >= 0.75, "CRR Art. 161(1)(b): LGD ≥ 75% subordinated"),
    ],
    "EAD": [
        ("non_negative", lambda v: v >= 0, "EAD cannot be negative"),
    ],
    "CCF": [
        ("range", lambda v: 0 <= v <= 1.0, "CCF must be between 0% and 100%"),
    ],
}
```

Only rows that pass hard rules but have ambiguous values go to the LLM.

---

## Scalability

| Table size | Strategy | LLM calls | Runtime (local) |
|---|---|---|---|
| < 10K rows | All rows through LLM | 10K | ~6h |
| 10K–500K | Hard rule pre-filter (pass 5-20% to LLM) | ~50K | ~1 day |
| > 500K | Stratified sample per segment + full hard-rule scan | ~10K | ~6h |

For large tables, stratified sampling by `(segment, rating_grade, vintage)` ensures every meaningful sub-population is represented.

---

## Output Format

```json
{
  "table": "MYLIB.PD_ESTIMATES",
  "run_date": "2026-05-20",
  "rows_processed": 45231,
  "rows_flagged": 312,
  "compliance_rate": 0.9931,
  "findings": [
    {
      "row_id": "OBL_00412",
      "column": "PD",
      "value": 0.00015,
      "verdict": "flagged",
      "flag": "PD de 0.015% por debajo del suelo regulatorio de 0.03%",
      "articles": ["CRR Art. 160(1)"],
      "regulatory_text": "..."
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

## Implementation Plan

### Phase 1 — Core pipeline (local, single table)
- [ ] SAS connector (CSV export path first, ODBC later)
- [ ] Hard rule engine (`HARD_RULES` dict, configurable per table schema)
- [ ] Row-level LLM call with RAG retrieval
- [ ] Running memory with self-compression
- [ ] JSON report output

### Phase 2 — Scale & robustness
- [ ] Async batching (process N rows in parallel)
- [ ] Checkpoint/resume (save progress to disk, restart-safe)
- [ ] Stratified sampling for very large tables
- [ ] Schema auto-detection (infer column semantics from names)

### Phase 3 — Fine-tuned model
- [ ] Label ~500 rows manually (compliant / flagged + reason)
- [ ] Fine-tune Qwen2.5-7B with LoRA on this dataset
- [ ] Compare fine-tuned vs RAG-only accuracy
- [ ] Deploy fine-tuned model via Ollama locally

### Phase 4 — UI / reporting
- [ ] Simple FastAPI endpoint: POST table → GET compliance report
- [ ] PDF report generation
- [ ] Integration with existing regllm.xyz frontend

---

## Key Dependencies

```
saspy / pyodbc          # SAS connection
pandas                  # table handling
ollama                  # local LLM serving
sentence-transformers   # embeddings (already in regllm)
psycopg2 + pgvector     # RAG retrieval (already in regllm)
```

---

## Open Questions

1. **What tables?** — PD estimates, LGD curves, EAD/CCF, IFRS9 staging, stress parameters?
2. **What SAS environment?** — SAS 9.4, SAS Viya, or SAS output files only?
3. **Segment structure?** — retail vs corporate vs sovereign, collateral types?
4. **Reference period?** — need to know if parameters are point-in-time or through-the-cycle to apply correct EBA rule
5. **Standalone tool or integrated into regllm.xyz?** — separate repo recommended

# Regulatory Audit Finding Generator — System Documentation

## Purpose

Automatically generate structured audit findings by comparing a database of recovery cycles against official regulation articles, using the existing SAS codebase as the bridge between regulatory concepts and database column names.

**Core value proposition:** the system maps regulation language ("período de dotación mínimo") to database columns (`PROVISION_PERIOD_MONTHS`) automatically, then generates SQL-level checks that run at scale.

---

## What the system is NOT (scope boundaries)

The following are explicitly deferred to future iterations:

- Cycle embeddings / TabBERT / anomaly detection
- t-SNE / UMAP visualisation
- Contrastive fine-tuning for article-cycle alignment
- Cycle narration pipeline
- Code audit (checking SAS logic against regulation, as opposed to data values)
- ChatRAG conversational interface

These ideas are documented in [audit_system_v2_ideas.md](audit_system_v2_ideas.md).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  INPUTS                                             │
│  • data/regulation/*.md   (regulation articles)     │
│  • data/sas/{v2,v3}/*.sas (SAS source code)        │
│  • data/docs/**/*.md      (variable definitions)    │
│  • mylib.ciclos_recuperacion (1M+ cycle rows)       │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 1 — Variable Mapper          (run once)      │
│  src/audit/variable_mapper.py                       │
│                                                     │
│  SASLogicTree.trace_lineage() per field             │
│  + GraphRAG over regulation articles                │
│  + LLM maps: regulation concept → DB column         │
│  → data/audit/mapping.json                          │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 2 — DQC Queries          (per article)       │
│  data/audit/dqc/art_12.sql, art_15.sql, ...        │
│                                                     │
│  LLM drafts SQL from: article md + mapping.json     │
│  Human reviews once, stored as versioned files      │
│  Queries run directly on the database (not CSVs)    │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 3 — DQC Engine           (per audit run)     │
│  src/audit/dqc_engine.py                            │
│                                                     │
│  Runs approved SQL queries against DB               │
│  Returns: [{ciclo_id, article, field, value, min}]  │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 4 — Finding Generator    (per violation)     │
│  src/audit/finding_generator.py                     │
│                                                     │
│  violation row + regulation article excerpt         │
│  → LLM generates structured finding dict            │
│  → data/audit/findings/{run_id}.jsonl               │
└─────────────────────────────────────────────────────┘
```

---

## Data contracts

### `data/audit/mapping.json`

```json
{
  "version": "v3",
  "generated_at": "2026-06-15",
  "mappings": [
    {
      "regulation_concept": "período de dotación",
      "regulation_variable": "PROVISION_PERIOD_MONTHS",
      "articles": ["art_12", "art_23"],
      "sas_variable": "PERIODO_DOT",
      "db_column": "PROVISION_PERIOD_MONTHS",
      "sas_lineage": ["MESES_EN_DEFAULT", "FECHA_DEFAULT"],
      "confidence": 0.92,
      "notes": ""
    }
  ]
}
```

**Confidence < 0.7** flags a mapping for mandatory human review before any DQC query referencing it is approved.

### Structured finding

```json
{
  "finding_id": "ART12-CIC_00031-20260615",
  "ciclo_id": "CIC_00031",
  "article": "art_12",
  "article_section": "Segmento CORP (corporativo)",
  "db_column": "PROVISION_PERIOD_MONTHS",
  "observed_value": 10,
  "regulatory_minimum": 12,
  "gap": -2,
  "segmento": "CORP",
  "cycle_phase": "FASE_CONTRACCION",
  "severity": "HIGH",
  "rationale": "Ciclo CORP en fase de contracción con 10 meses de dotación. El artículo 12 exige un mínimo de 18 meses para este segmento y fase.",
  "evidence_section": "art_12_periodos_dotacion.md § Segmento CORP",
  "run_id": "audit-2026-06-15"
}
```

### DQC query contract

Each file in `data/audit/dqc/` must satisfy:
- Named `{article_id}.sql`
- Returns columns: `ciclo_id`, `db_column`, `observed_value`, `regulatory_minimum`, `segmento`, any additional context columns
- Contains a header comment block with: article reference, author, review date, last validated against findings dataset

---

## New modules

### `src/audit/variable_mapper.py`

**Responsibility:** one-time mapping of regulation concepts to database columns.

**Inputs:**
- Regulation articles via `GraphRAG` (`src/knowledge/graph_rag.py`)
- SAS lineage via `SASLogicTree.trace_lineage()` (`src/sas_logic_tree.py`)
- Variable definitions via `DocsIndex.search()` (`src/agent/docs_index.py`)

**Key function:**
```python
def build_mapping(sas_version: str = "v3") -> dict:
    """
    For each field mentioned in regulation articles, find the
    corresponding SAS variable and DB column. Returns mapping dict
    ready to write to data/audit/mapping.json.
    """
```

**LLM usage:** one `chat_json()` call per regulation field mention. Not called at query time.

---

### `src/audit/dqc_engine.py`

**Responsibility:** store, validate, and execute DQC SQL queries.

**Key functions:**
```python
def list_dqc_queries() -> list[str]:
    """Return article IDs for which an approved DQC query exists."""

def run_dqc(article_id: str, db_conn) -> list[dict]:
    """
    Execute the approved SQL for article_id against the database.
    Returns list of violation dicts matching the finding data contract.
    """

def run_all_dqc(db_conn) -> list[dict]:
    """Run every approved DQC query and return combined violations."""
```

**SQL storage:** `data/audit/dqc/{article_id}.sql` — versioned in git, never generated at runtime.

**Database connection:** accepts any PEP 249-compliant connection (SQLAlchemy, psycopg2, etc.). Defaults to the configured RDS/pgvector connection from the existing API setup.

---

### `src/audit/finding_generator.py`

**Responsibility:** convert a DQC violation row into a structured audit finding with regulatory rationale.

**Key function:**
```python
def generate_finding(violation: dict, rag: GraphRAG) -> dict:
    """
    Given a violation dict from the DQC engine, retrieve the relevant
    regulation article section and ask the LLM to generate a structured
    finding. Returns a finding dict matching the data contract.
    """
```

**LLM usage:** one `chat_json()` call per violation. The prompt includes:
- The violation values
- The relevant regulation article section (retrieved via `GraphRAG`)
- The field's definition from the docs index
- Output schema (finding dict above)

**Batching:** violations are grouped by article before LLM calls to allow the system prompt to include the full article context once per group.

---

## New agent tools

Two tools added to `TOOL_REGISTRY` in `src/agent/tools.py`:

### `run_dqc`

```
Run all approved DQC queries (or a specific article's query) against
the database and return the list of regulatory violations found.
Parameters: article_id (optional, string) — omit to run all.
```

### `list_findings`

```
Retrieve structured audit findings for a specific cycle or article.
Parameters: ciclo_id (optional), article_id (optional), run_id (optional).
Returns findings from the most recent audit run matching the filters.
```

---

## Build order

### Step 1 — Variable mapper
- `src/audit/variable_mapper.py`
- `data/audit/mapping.json` (generated, then reviewed)
- Validate: every field in `data/regulation/*.md` has a mapping entry

### Step 2 — DQC queries
- `data/audit/dqc/art_08.sql`
- `data/audit/dqc/art_12.sql`
- `data/audit/dqc/art_15.sql`
- `data/audit/dqc/art_23.sql`
- LLM drafts from article md + mapping.json; each reviewed before use
- Validate: each query returns expected violations from the audit findings dataset

### Step 3 — DQC engine
- `src/audit/dqc_engine.py`
- Validate: `run_all_dqc()` reproduces known findings from the dataset

### Step 4 — Finding generator
- `src/audit/finding_generator.py`
- Validate: structured output matches data contract for a sample of violations

### Step 5 — Agent tools
- `run_dqc` and `list_findings` in `src/agent/tools.py`
- Update `tests/test_agent_tools.py` expected tool set

---

## Validation strategy

At each step, the audit findings dataset is the ground truth:

```
known_findings = load("data/audit/known_findings.jsonl")

# After Step 2
for finding in known_findings:
    assert dqc_catches(finding), f"DQC missed known finding: {finding['finding_id']}"

# After Step 3
results = run_all_dqc(db_conn)
precision = len(results ∩ known_findings) / len(results)
recall    = len(results ∩ known_findings) / len(known_findings)
# Target: recall > 0.85 before Step 4
```

Precision is less critical than recall for an audit tool — missing a finding is worse than a false positive.

---

## What stays unchanged

| Component | Status |
|---|---|
| `src/knowledge/graph_rag.py` | Unchanged — used by finding_generator |
| `src/knowledge/change_log_graph.py` | Unchanged |
| `src/knowledge/llm_client.py` | Unchanged — used for LLM calls |
| `src/agent/docs_index.py` | Unchanged — used by variable_mapper |
| `src/sas_logic_tree.py` | Unchanged — used by variable_mapper |
| `data/regulation/*.md` | Unchanged — source for DQC generation |
| `data/docs/regulation/*.md` | Unchanged — source for variable_mapper |
| All existing agent tools | Unchanged |
| All existing tests | Unchanged |

---

## Deferred to v2

See [audit_system_v2_ideas.md](audit_system_v2_ideas.md) for:

- **Cycle narration pipeline** — converting tabular rows to text for LLM explanation
- **Shared embedding space** — aligning article embeddings with cycle embeddings via sentence transformers
- **Contrastive fine-tuning** — using audit findings to make nearest-article = violated-article
- **TabBERT / SCARF embeddings** — anomaly detection for unknown violations
- **UMAP + DBSCAN** — visualisation and clustering of cycle embeddings coloured by TERMINACION / CALIBRACION_SEGMENT
- **Code audit** — comparing SAS AST structure against regulation logic (as opposed to data values)
- **ChatRAG interface** — conversational exploration of findings and regulation

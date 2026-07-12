# Two RAG channels: context RAG and regulation RAG

The DQC generator answers one question per request: *what data-quality
checks apply to this field?* Answering it well needs two genuinely
different kinds of grounding, so the system keeps them as two separate
retrieval channels rather than one undifferentiated "search everything" —
`api/routers/dqc.py::_gather_context()` calls both and hands the LLM the
union.

```
                     ┌─────────────────────────────────────────┐
                     │            _gather_context()             │
                     └─────────────────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
       CONTEXT RAG            REGULATION RAG (graph)    REGULATION RAG (semantic)
   "what IS this field?"    "which article, IF the     "which article, for
                              field is already linked   ANY query, by meaning"
                              into the graph"
              │                        │                        │
   SAS lineage/formula        search_regulation          search_regulation_semantic
   + docs_index (BM25)        (networkx 2-hop            (embedding cosine
   over data/docs/**/*.md     subgraph traversal          similarity over
                               over graph.json)            chunked EBA GL/2017/16)
```

## Context RAG — what a field *is*

- `get_sas_formula` / `trace_dependencies` / `backtrace_sas_field` — where
  the field comes from in the SAS pipeline lineage.
- `search_docs` / `get_field_definition` (`src/agent/docs_index.py`) — BM25
  over the markdown glossary/table-dictionary corpus in `data/docs/`.

This answers "what does `MOC_CAT_A` mean and where is it computed" —
necessary but not sufficient: it says nothing about which regulatory
*requirement* the field must satisfy.

## Regulation RAG — what the guidelines *require*

Two complementary retrieval strategies over regulatory text, because they
have opposite strengths:

|  | `search_regulation` (existing) | `search_regulation_semantic` (new) |
|---|---|---|
| Mechanism | `networkx` 2-hop subgraph traversal over `data/regulation/graph.json` | cosine similarity over embedded paragraph chunks |
| Requires | the field to already be **linked** to a regulatory section (an earlier LLM extraction pass built `MENTIONS_FIELD` edges) | nothing — works on any text, including fields never linked |
| Precision | high — an edge is an explicit, reviewed claim | lower — semantic similarity, not a verified claim |
| Recall | low — only ~28/48 fields in the graph have edges (see `docs/EVALUATION.md` §4) | high — every one of the 221 EBA GL/2017/16 paragraphs is searchable |
| Query shape | works best on exact field names | works on natural language too ("suelo de LGD para hipotecas") |

Neither replaces the other. The graph gives a curated, precise citation
when it exists; the semantic index is the recall backstop for everything
the graph hasn't captured yet — and it's also what makes free-form
questions (not just field names) answerable at all.

## Chunking the guidelines

`src/knowledge/regulation_chunker.py` turns
`data/regulation/eba_gl_2017_16_articles.json` (221 paragraphs, already
segmented by the ingestion pipeline, each tagged with its section/
subsection) into ~275 retrieval chunks:

- **One chunk per paragraph by default.** The paragraph is the citation
  unit ("EBA GL 2017/16 §73") — chunking any coarser (merging paragraphs)
  would blur which paragraph a retrieved passage actually supports, which
  matters a great deal when the system prompt explicitly forbids inventing
  regulatory references.
- **Long paragraphs (>1000 chars, ~10% of them) are split** into
  overlapping, **sentence-aligned** windows — never mid-sentence, never
  mid-number — so no chunk exceeds an embedding model's practical context
  while each split part stays self-contained enough to embed meaningfully
  on its own. The `overlap` (default 150 chars) carries trailing sentences
  from one window into the next so a search term near a split boundary
  isn't missed by either window.
- Paragraphs are **never merged** — a chunk always maps back to exactly one
  paragraph number.

## Embedding backends

`src/knowledge/embeddings.py::EmbeddingService` mirrors the chat client's
multi-backend design (`src/knowledge/llm_client.py`): Ollama → standalone
GGUF → Bedrock Titan → zero-vector stub, selected via
`REGLLM_EMBED_BACKEND` independently of the chat backend (`REGLLM_LLM`) —
so, for example, chat can run on Bedrock while embeddings run on a cheaper
local Ollama model. See `.env.example` for every variable.

## Building and using the index

```bash
python scripts/build_regulation_embeddings.py
```

writes `data/regulation/embeddings/pd_lgd_chunks.json` — a single JSON file
with a `manifest` (embedding backend/model/dim/build time) and the chunk
records, each carrying its embedding vector inline. It's gitignored (a
generated artifact, like the DQC eval harness's `.db` files) — rebuild
after chunker changes or a backend/model switch. The build script loudly
warns if it detects it's running in stub mode (no backend reachable), since
that would silently produce an all-zero-vector, unranked index.

`src/knowledge/regulation_vector_store.py::RegulationVectorStore` loads it
and does **pure-Python cosine similarity** — deliberately not
`ChromaDB`-backed like the SAS-diff explainer's knowledge-graph RAG
(`src/knowledge/vector_store.py`). At this corpus size (~275 short chunks)
a compiled vector database buys nothing but a dependency, and staying
dependency-free means the DQC production image
(`requirements-dqc.txt` — no chromadb, no numpy) can serve semantic
regulation search too. See `docs/DEPLOYMENT.md`.

## Try it

```bash
python scripts/build_regulation_embeddings.py   # needs Ollama/GGUF/Bedrock configured for real vectors
python -c "
from src.knowledge.embeddings import get_embedding_service
from src.knowledge.regulation_vector_store import RegulationVectorStore
store = RegulationVectorStore.load()
for hit in store.search_text('suelo de LGD para hipotecas', get_embedding_service(), k=3):
    print(f'{hit.citation}  (score {hit.score})')
    print(f'  {hit.text[:160]}...')
"
```

Both `_t_search_regulation` and `_t_search_regulation_semantic` results flow
into the same `ctx` dict `_gather_context()` builds and the same
`_extract_sources()` pass that surfaces citations to the UI — see
`RAGSource.source_type` (`"regulation"` vs `"regulation_semantic"` vs
`"definition"` vs `"docs"`) to tell them apart in the API response.

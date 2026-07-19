# Deployment — the DQC production backend

The DQC generator API ships as a single Docker image, built from the root
`Dockerfile` and deployed by `.github/workflows/deploy.yml` to ECS Fargate
(`DQC/terraform/`, with `DQC/cdk/` as an equivalent CDK translation). This
document explains the backend's dependency footprint and how it's kept slim.

## Two requirements files, one codebase

| File | Used by | Contents |
|---|---|---|
| `requirements.txt` | local full-stack dev, `pytest`, CI's main `test` job | Superset: adds `torch`, `kuzu`, `chromadb`, `scikit-learn`, `umap-learn`, `pandas`, `openpyxl`, and (commented, opt-in) `llama-cpp-python` |
| `requirements-dqc.txt` | the production Docker image (`Dockerfile`) | `fastapi`, `uvicorn`, `pydantic`, `python-multipart`, `networkx`, `httpx`, `pyyaml`, `boto3` |

`llama-cpp-python` (the standalone GGUF backend, see README's "Local LLM
integration") is opt-in everywhere, including full dev: it's a compiled
dependency, lazily imported, and only touched when `REGLLM_LLM=gguf` /
`REGLLM_EMBED_BACKEND=gguf` is explicitly configured with a real weight
file. Uncomment it in `requirements.txt` (and add it to
`requirements-dqc.txt` too) if your deployment should run fully local with
no Ollama/Bedrock dependency at all.

The codebase is one FastAPI app (`api/main.py`) whose **mounted router set**
is controlled by `REGLLM_ROUTERS` (default `all`; the Dockerfile bakes in
`REGLLM_ROUTERS=dqc`). Routers that need a dependency not installed are
skipped with a logged warning instead of crashing the app — see
`api/main.py`'s router-loading loop. With `REGLLM_ROUTERS=dqc`, only
`api/routers/dqc.py` is mounted; `sas`, `diff`, `embeddings`, `tabular`, and
`kg` (the routers that need torch/chromadb/scikit-learn) are never imported.

## Why `requirements-dqc.txt` doesn't need torch/kuzu/chromadb

Tracing the DQC request path end to end:

- `api/routers/dqc.py` calls into `src/agent/tools.py` (SAS lineage/formula
  lookup — pure stdlib) and `src.knowledge.get_client()` /
  `src.knowledge.GraphRAG` (LLM transport + regulation-graph search).
- `GraphRAG` traverses `data/regulation/graph.json` with **`networkx`** — a
  pure-Python graph library. It does not touch `kuzu` (a separate, compiled
  graph database used by the *interactive* knowledge-graph builder —
  `src/knowledge/graph_store.py` — which the DQC generator never calls).
- `torch` backs the SAS field-diff explainer's differentiable evaluator
  (`src/sas_diff/`), imported lazily only by the `sas`/`diff` routers.
- `chromadb`/`scikit-learn`/`umap-learn` back the embedding/tabular
  visualizer routers.

The one real leak, fixed alongside this split: `src/knowledge/__init__.py`
used to eagerly `from .graph_store import GraphStore` at package import
time — so merely importing `src.knowledge` (which `get_client()` needs)
forced `kuzu` to be installed, even though nothing in the DQC path uses
`GraphStore`. Every actual caller already imports it from the submodule
directly (`from src.knowledge.graph_store import GraphStore`), so removing
the package-root re-export was a safe, non-breaking fix.

## Verifying the image stays slim

Docker isn't always available in every environment this repo is worked in,
so the guarantee is enforced two ways, both checked in CI
(`.github/workflows/test.yml`'s `dqc-slim` job):

1. **`tests/test_dqc_slim_imports.py`** spawns subprocesses that install
   nothing beyond what's importable, boot the app with
   `REGLLM_ROUTERS=dqc`, hit `/health` and `/dqc/generate` for real, and
   assert `torch`/`kuzu`/`chromadb`/`sklearn`/`umap` never enter
   `sys.modules`. It also asserts `requirements-dqc.txt` and the `Dockerfile`
   itself stay slim (string-level guard against regressions).
2. **The `dqc-slim` CI job** installs *only* `requirements-dqc.txt` in an
   isolated runner (the main `test` job installs the full `requirements.txt`,
   which would mask a heavy import creeping back in) and runs that test file
   against a real environment where torch/kuzu genuinely aren't installed —
   the same condition the production image will be in.

To reproduce locally without Docker:

```bash
python -m venv /tmp/dqc-slim-venv
/tmp/dqc-slim-venv/bin/pip install -r requirements-dqc.txt pytest
/tmp/dqc-slim-venv/bin/pytest tests/test_dqc_slim_imports.py -v
```

Or with Docker, to build and smoke-test the real image:

```bash
docker build -t regllm-dqc -f Dockerfile .
docker run --rm -p 8000:8000 -e REGLLM_LLM=stub regllm-dqc &
curl http://localhost:8000/health
curl -X POST http://localhost:8000/dqc/generate \
     -H 'Content-Type: application/json' \
     -d '{"message": "Genera DQCs para PD_ESTIMADA"}'
```

## The regulation RAG index is also slim-compatible

`api/routers/dqc.py` now also queries an embedding-based semantic index over
the EBA GL/2017/16 PD & LGD guidelines (`search_regulation_semantic` — see
[`docs/REGULATION_RAG.md`](REGULATION_RAG.md) for the full design). Like the
rest of the DQC path, this stays dependency-free in the slim image:
`src/knowledge/regulation_chunker.py` and
`src/knowledge/regulation_vector_store.py` are pure stdlib (no numpy, no
chromadb — cosine similarity over a few hundred chunks is trivial in plain
Python), and `tests/test_dqc_slim_imports.py` exercises exactly this code
path (build a tiny index, search it) inside the same subprocess check that
guards the rest of the image.

The index itself (`data/regulation/embeddings/pd_lgd_chunks.json`) is a
**generated artifact** — gitignored, like the DQC eval harness's `.db`
files — built with:

```bash
python scripts/build_regulation_embeddings.py
```

If it hasn't been built, `search_regulation_semantic` degrades gracefully
(`{"available": false, "hint": "..."}`) rather than failing the request —
`/dqc/generate` keeps working on the graph-based regulation search and
context RAG alone, same as any other missing-context degrade in this
codebase.

## Choosing a production LLM backend: Bedrock vs standalone GGUF

The slim image (`requirements-dqc.txt`) already ships `boto3`, so **Amazon
Bedrock is the zero-extra-dependency choice** for the deployed ECS
task — no weight file to bake into the image or mount, no GPU, no Ollama
sidecar. It's what `DQC/terraform/` and `DQC/cdk/` provision IAM for by
default. Set:

```bash
REGLLM_LLM=bedrock
BEDROCK_MODEL_ID=eu.amazon.nova-micro-v1:0   # default; any Converse-API model works
BEDROCK_REGION=eu-west-1
```

and give the ECS task role `bedrock:InvokeModel` / `bedrock:InvokeModelWithResponseStream`
on the model ARN (already wired in the Terraform/CDK IAM policy). Auth uses
the standard AWS credential chain — nothing else to configure. All three
LLM-client surfaces (`chat`, `chat_json_stream` streaming, and
`chat_tools`/tool-calling for the SAS-lineage agent) are implemented for
Bedrock via the Converse API (`src/knowledge/llm_client.py`).

The standalone **GGUF** backend (`REGLLM_LLM=gguf`) is the right choice
instead when the deployment must be fully air-gapped (no AWS egress at
all): it needs `llama-cpp-python` (opt-in, compiled — see the note above)
and a weight file baked into the image or mounted at `GGUF_MODEL_PATH`, but
then has zero external network dependency at inference time. See the
README's "Local LLM integration" section for the full GGUF setup.

## What's still needed for production (see `docs/EVALUATION.md`)

The slim image addresses image size, cold-start time, build time, and
attack surface. It does **not** by itself address the other production
gaps already tracked in `docs/EVALUATION.md`: durable storage for the
validated-checks SQLite (wiped on every ECS task replacement — an EFS mount
or a DynamoDB-backed store is the fix), auth/TLS in front of the
internet-facing ALB, and running the eval harness against a staged endpoint
as a deploy gate.

## Azure deployment (Container Apps)

`DQC/azure/deploy.sh` is the Azure mirror of `DQC/cdk/deploy.sh`: one
Container App with the same two sidecar containers (api + nginx frontend
sharing localhost, demo Excels bundled), managed HTTPS ingress, images
built server-side with `az acr build` (no local Docker). The LLM backend
is Azure OpenAI via the client's `azure` backend
(`REGLLM_LLM=azure` + `AZURE_OPENAI_ENDPOINT/API_KEY/DEPLOYMENT`);
without those variables it deploys with the stub backend and can be
re-pointed later with `az containerapp update`. One command after
`az login`:

    AZURE_OPENAI_ENDPOINT=https://<res>.openai.azure.com \
    AZURE_OPENAI_API_KEY=... ./DQC/azure/deploy.sh

Teardown: `az group delete --name regllm-dqc-rg --yes`.

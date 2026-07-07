# Deployment — the DQC production backend

The DQC generator API ships as a single Docker image, built from the root
`Dockerfile` and deployed by `.github/workflows/deploy.yml` to ECS Fargate
(`DQC/terraform/`, with `DQC/cdk/` as an equivalent CDK translation). This
document explains the backend's dependency footprint and how it's kept slim.

## Two requirements files, one codebase

| File | Used by | Contents |
|---|---|---|
| `requirements.txt` | local full-stack dev, `pytest`, CI's main `test` job | Superset: adds `torch`, `kuzu`, `chromadb`, `scikit-learn`, `umap-learn`, `pandas`, `openpyxl` |
| `requirements-dqc.txt` | the production Docker image (`Dockerfile`) | `fastapi`, `uvicorn`, `pydantic`, `python-multipart`, `networkx`, `httpx`, `pyyaml`, `boto3` |

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

## What's still needed for production (see `docs/EVALUATION.md`)

The slim image addresses image size, cold-start time, build time, and
attack surface. It does **not** by itself address the other production
gaps already tracked in `docs/EVALUATION.md`: durable storage for the
validated-checks SQLite (wiped on every ECS task replacement — an EFS mount
or a DynamoDB-backed store is the fix), auth/TLS in front of the
internet-facing ALB, and running the eval harness against a staged endpoint
as a deploy gate.

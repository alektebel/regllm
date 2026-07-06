# RegLLM — DQC (Data Quality Check) Generator

> Generate portable **SAS/SQL data-quality checks** for IRB / IFRS 9
> regulatory fields (COREP/FINREP), grounded in a SAS lineage graph and a
> regulation corpus, and reviewed through a validate/reject pipeline with a
> copy-ready dashboard query.

Runs **local-first** (offline, on Ollama + SQLite) and deploys to **AWS**
(Bedrock + DynamoDB) with a single script.

---

## What it does

For a target regulatory variable (e.g. `LGD_ESTIMADA`, `ECL`, `PD_ESTIMADA`),
DQC:

1. **Gathers context** — six read-only tools (`src/agent/tools.py`) pull the
   SAS formula, the dependency lineage, the field definition, relevant
   regulation sections, and BM25 doc hits.
2. **Generates checks** — the context is sent to an LLM with
   `DQC_SYSTEM_PROMPT`, which returns structured DQC objects. Each carries a
   `regla_sql` (a SAS `DATA` step or `PROC SQL`) that flags offending rows.
3. **Persists** — generated checks are stored as `pending` (SQLite locally,
   DynamoDB in AWS).
4. **Validates** — the Angular UI lets you validate / reject each check; once
   all visible checks are resolved, a UNION-ALL **dashboard query** is
   unlocked for one-click copy.

```
┌────────────────────┐     /api/dqc/*      ┌──────────────────────────┐
│  Angular UI        │ ──────────────────▶ │  FastAPI  (api/)         │
│  DQC/app           │ ◀──── JSON ──────── │  api/routers/dqc.py      │
│  • Chat            │                     │   ├─ SAS lineage tools    │
│  • Tests en lote   │                     │   ├─ Regulation graph     │
│  • Validate / Dash │                     │   └─ LLM (Ollama/Bedrock/ │
└────────────────────┘                     │        Gemini)           │
                                           └───────────┬──────────────┘
                                                       │  checks store
                                           ┌───────────▼──────────────┐
                                           │  SQLite (local)          │
                                           │  DynamoDB (AWS)          │
                                           └──────────────────────────┘
```

---

## Repository layout

| Path                              | Role                                                     |
|-----------------------------------|----------------------------------------------------------|
| `DQC/app/`                        | Angular UI (chat, batch tests, validate/dashboard)       |
| `DQC/cdk/`                        | AWS CDK stack + one-command `deploy.sh`                  |
| `DQC/eval/`                       | DQC eval harness (stress DBs, scoring)                    |
| `api/main.py`                     | FastAPI entry point                                      |
| `api/routers/dqc.py`              | The `/dqc/*` endpoints                                    |
| `src/agent/tools.py`              | Read-only lineage / regulation / docs tools              |
| `src/knowledge/`                  | Local LLM client + regulation/change-log GraphRAG        |
| `training/dq/checks_store.py`     | Pluggable checks store: `SqliteStore` \| `DynamoStore`   |
| `training/dq/checks_db.py`        | SQLite implementation of the checks table                |
| `data/regulation/`, `data/docs/`  | Regulation corpus + doc corpus (BM25) read at runtime    |
| `data/samples/`                   | Bundled sample SAS + `irb_schema.sql`                     |
| `Dockerfile`, `docker-compose.yml`| API image + local `ollama` + `api` + `dqc` stack         |

---

## Setup

### Prerequisites

- **Python 3.11+** (the API), **Node.js LTS 18/20** (the Angular UI).
- Optional: **Docker Desktop** (one-command local stack), and an LLM —
  local **Ollama**, a **Gemini** API key, or **AWS Bedrock** access.

### Option A — Docker (one command)

Starts Ollama (pulls the model on first run), the FastAPI API, and the
Angular UI:

```bash
cp .env.example .env
docker compose --profile ollama up --build
```

Open **http://localhost:4200**. The UI's nginx proxies `/api/*` to the API
container. First boot is slow (Ollama pulls the model); later boots are fast.

### Option B — Manual (two terminals)

**Backend** — full walkthrough (LLM backends, checks store, env reference) in
**[api/README.md](api/README.md)**. Note the port: the dev UI proxy targets
`:8001`, so run uvicorn there (not the 8000 default):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --port 8001 --reload
# API docs: http://localhost:8001/docs
```

**Frontend** — see **[DQC/app/README.md](DQC/app/README.md)** for the full
Windows-with-npm walkthrough. In short:

```bash
cd DQC/app
npm install
npm start            # ng serve, proxy.conf.json → http://localhost:8001
# open http://localhost:4200
```

### LLM backends

The backend is selected by `REGLLM_LLM` (default auto-detects Ollama, else a
deterministic stub so the pipeline still runs offline):

| Backend   | Env                                                          |
|-----------|--------------------------------------------------------------|
| `ollama`  | `OLLAMA_URL`, `OLLAMA_MODEL` (default `qwen2.5:14b-instruct-q4_K_M`) |
| `gemini`  | `GEMINI_API_KEY`, `GEMINI_MODEL` (default `gemini-2.5-pro`)   |
| `bedrock` | `BEDROCK_MODEL_ID`, `BEDROCK_REGION` (IAM auth)              |
| `stub`    | none — deterministic placeholder for offline dev             |

---

## Using the app

- **Chat** — ask for checks on a variable; generated DQCs land in the sidebar
  as `pending`.
- **Tests en lote** — paste one natural-language test per line (or attach a
  `.txt`/`.md`/`.csv`); each becomes one DQC via `POST /api/dqc/generate/tests`.
- **Validate / Dashboard** — validate or reject each check. When no visible
  `pending` checks remain and at least one is validated, the dashboard unlocks
  a UNION-ALL query (every check's `sql` must `SELECT` the PK column
  `ID_CONTR_CICLO_LGD`) with a **copy-all** button.

Example batch (`.txt`, one test per line):

```
Verifica que PD_ESTIMADA cumple los suelos regulatorios
Comprueba que ECL = PD x LGD x EAD
Valida que STAGE_IFRS9=3 implica PD=1.0
```

---

## Checks persistence (SQLite ↔ DynamoDB)

The router talks to a store abstraction (`training/dq/checks_store.py`),
selected by `CHECKS_BACKEND`:

| `CHECKS_BACKEND` | Store         | Needs                        | Used for            |
|------------------|---------------|------------------------------|---------------------|
| `sqlite` (default)| `SqliteStore` | nothing (writes `data/dq/checks.db`) | local dev   |
| `dynamodb`       | `DynamoStore` | `CHECKS_TABLE` + AWS creds    | AWS deployment      |

Both backends expose the same interface and return identical row shapes, so
the endpoints are backend-agnostic (see `tests/test_checks_store.py` for the
moto-backed parity tests). The CDK stack sets `CHECKS_BACKEND=dynamodb` and
`CHECKS_TABLE` automatically.

---

## Deploy to AWS (Bedrock + DynamoDB)

The CDK stack (`DQC/cdk/`) is **self-contained**: it creates its own VPC
(public subnets, no NAT gateways), ECR repos, a DynamoDB table, IAM roles,
an ALB, and an ECS Fargate service running two sidecar containers — the
FastAPI API (`:8000`) and nginx serving the Angular build (`:80`, proxying
`/api/` to the API). `deploy.sh` also builds and pushes both images.

**Prerequisites:** `aws` CLI (configured for the target account), `docker`,
`node`/`npm`, Python 3.11+, and — for the default Bedrock backend —
**Bedrock model access enabled** for the model in the target region
(Bedrock console → *Model access*; default `eu.amazon.nova-micro-v1:0` in
`eu-west-1`).

```bash
cd DQC/cdk
./deploy.sh                              # Bedrock backend (default)
# or, to use Gemini instead of Bedrock:
export GEMINI_API_KEY="your-key"
./deploy.sh
```

`deploy.sh` prints the ALB URL when done. Verify and use it:

```bash
curl http://<alb-dns>/api/health         # → {"status":"ok","llm_backend":"bedrock"}
```

Deployed DQCs persist to the DynamoDB table `<project>-checks` (default
`regllm-dqc-checks`), surfaced as the `ChecksTableName` stack output.

Override defaults via environment variables:

| Variable          | Default                     | Effect                                |
|-------------------|-----------------------------|---------------------------------------|
| `AWS_REGION`      | `eu-west-1`                 | Target region                         |
| `PROJECT`         | `regllm-dqc`                | Resource name prefix (table, cluster) |
| `GEMINI_API_KEY`  | —                           | Set → switch LLM backend to Gemini    |
| `BEDROCK_MODEL_ID`| `eu.amazon.nova-micro-v1:0` | Bedrock model / inference profile     |

**Notes**

- First deploy is slowest (VPC + ALB). Later runs update in place and only
  rebuild changed images.
- The Gemini key (when used) is passed via CDK context and ends up as a
  plaintext env var on the ECS task definition. `cdk.context.json` is
  gitignored so a cached key is never committed.
- Redeploy code only: rebuild + push both images, then
  `aws ecs update-service --cluster <project> --service <project> --force-new-deployment`.
- A Terraform variant exists under `DQC/terraform/`; CDK (`deploy.sh`) is the
  canonical path and the one wired for DynamoDB.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -q          # checks-store parity (SQLite + DynamoDB via moto)
```

---

## Out of scope

- No DynamoDB ↔ SQLite migration (local dev and AWS keep separate stores).
- No auth / multi-user / chat history.
- Schema migrations are `CREATE IF NOT EXISTS` only (no Alembic).

## License

See repository.

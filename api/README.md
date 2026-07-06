# DQC backend (FastAPI) — setup

The backend is a FastAPI app (`api/main.py`) exposing the DQC generator and
validation pipeline under `/dqc/*`. It orchestrates read-only SAS-lineage /
regulation tools + an LLM, and persists checks to a pluggable store (SQLite
locally, DynamoDB in AWS).

---

## 1. Prerequisites

- **Python 3.11+**
- An **LLM backend** (pick one below). Without one the API still runs in
  deterministic **stub** mode — useful for wiring/UI work, but generated
  checks are placeholders.
- For DynamoDB persistence: AWS credentials (only if you set
  `CHECKS_BACKEND=dynamodb`; the default SQLite needs nothing).

## 2. Install

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt     # includes boto3 + google-genai
```

`torch` (CPU build) is the heaviest dependency; the SAS lineage tools use it.

## 3. Choose an LLM backend

Selected by `REGLLM_LLM` (default `auto` → probes LiteRT, then Ollama, else
stub). All deps ship in `requirements.txt`.

**Ollama (local, recommended, offline)**
```bash
# install Ollama from https://ollama.com, then:
ollama pull qwen2.5:14b-instruct-q4_K_M     # or a smaller tag
export REGLLM_LLM=ollama
export OLLAMA_MODEL=qwen2.5:14b-instruct-q4_K_M   # default; OLLAMA_URL defaults to :11434
```

**Google Gemini (cloud)**
```bash
export REGLLM_LLM=gemini
export GEMINI_API_KEY=your-key
export GEMINI_MODEL=gemini-2.5-pro           # default
```

**AWS Bedrock (cloud, IAM auth)**
```bash
export REGLLM_LLM=bedrock
export BEDROCK_MODEL_ID=eu.amazon.nova-micro-v1:0   # default
export BEDROCK_REGION=eu-west-1
# credentials via the standard AWS chain (env / ~/.aws / instance role);
# enable model access for the model in the Bedrock console first.
```

**Stub (no LLM)** — `export REGLLM_LLM=stub`.

## 4. Choose the checks store

Selected by `CHECKS_BACKEND` (default `sqlite`):

| Value              | Store         | Needs                                   |
|--------------------|---------------|-----------------------------------------|
| `sqlite` (default) | `SqliteStore` | nothing — writes `data/dq/checks.db`    |
| `dynamodb`         | `DynamoStore` | `CHECKS_TABLE` + AWS credentials/region |

```bash
# local (default) — nothing to set
# AWS-style local run against a real table:
export CHECKS_BACKEND=dynamodb
export CHECKS_TABLE=regllm-dqc-checks
export AWS_REGION=eu-west-1
```

The regulation / docs / samples data the backend reads is already committed
under `data/`, so no data setup is required.

## 5. Run

The dev frontend proxy (`DQC/app/proxy.conf.json`) targets **port 8001**, so
for local full-stack dev run uvicorn there:

```bash
uvicorn api.main:app --port 8001 --reload
```

(Default is 8000 if you omit `--port`; the Docker/AWS containers use 8000.)

## 6. Verify

```bash
curl http://localhost:8001/health
# → {"status":"ok","llm_backend":"ollama"}   (or gemini / bedrock / stub)

curl http://localhost:8001/dqc/checks/counts
# → {"pending_visible":0,"validated":0,"rejected":0,"oculto":0,"dashboard_ready":false}
```

Interactive API docs: **http://localhost:8001/docs**.

## 7. Run in Docker (API only)

```bash
docker build -t regllm-dqc-api .
docker run --rm -p 8000:8000 \
  -e REGLLM_LLM=stub \
  -v "$PWD/data:/app/data" \
  regllm-dqc-api
```

Or the full local stack (Ollama + API + UI) from the repo root:
`docker compose --profile ollama up --build`.

---

## Environment reference

| Variable             | Default                        | Purpose                                   |
|----------------------|--------------------------------|-------------------------------------------|
| `REGLLM_LLM`         | `auto`                         | `auto`\|`ollama`\|`gemini`\|`bedrock`\|`litert`\|`stub` |
| `OLLAMA_URL`         | `http://localhost:11434`       | Ollama server                             |
| `OLLAMA_MODEL`       | `qwen2.5:14b-instruct-q4_K_M`  | Ollama tag or `.gguf` path                |
| `GEMINI_API_KEY`     | —                              | Gemini auth                               |
| `GEMINI_MODEL`       | `gemini-2.5-pro`               | Gemini model                              |
| `BEDROCK_MODEL_ID`   | `eu.amazon.nova-micro-v1:0`    | Bedrock model / inference profile         |
| `BEDROCK_REGION`     | `eu-west-1`                    | Bedrock region                            |
| `REGLLM_LLM_TIMEOUT` | `120`                          | LLM request timeout (s)                   |
| `CHECKS_BACKEND`     | `sqlite`                       | `sqlite` \| `dynamodb`                     |
| `CHECKS_TABLE`       | —                              | DynamoDB table (when `dynamodb`)          |
| `CORS_ORIGINS`       | `http://localhost:3000,http://localhost:4200` | comma-separated allowed origins |

## Endpoints (`/dqc`)

| Method & path                     | Purpose                              |
|-----------------------------------|--------------------------------------|
| `POST /dqc/generate`              | Generate DQCs for a variable (chat)  |
| `POST /dqc/generate/tests`        | One DQC per natural-language test    |
| `GET  /dqc/generate/batch/stream` | SSE batch generation                 |
| `GET  /dqc/checks`                | List checks (`?status=&visible=`)    |
| `GET  /dqc/checks/counts`         | Status/visibility breakdown          |
| `POST /dqc/checks/{id}/status`    | Validate / reject a check            |
| `DELETE /dqc/checks/{id}`         | Delete a check                       |
| `GET  /dqc/dashboard`             | UNION-ALL dashboard query + validated checks |
| `GET  /health`                    | Liveness + active LLM backend        |

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -q          # checks-store parity (SQLite + DynamoDB via moto)
```

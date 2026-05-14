# RegLLM — Spanish Banking Regulation Assistant

**A fine-tuned LLM (Qwen2.5-7B + LoRA) for Spanish banking regulations, with a ChatGPT-style web UI, hybrid RAG, and full CI/CD to AWS.**

RegLLM answers questions about EBA guidelines, CRR/CRD IV, and Basel III/IV by combining a fine-tuned language model with hybrid vector + keyword retrieval over real regulatory documents.

---

## Architecture

```
Browser
  └─► Next.js 14  (port 3000)
        └─► /api/* rewrite
              └─► FastAPI  (port 8000)
                    ├── POST  /auth/register|login   (5 req/min rate limit)
                    ├── GET/POST/DELETE  /conversations
                    └── POST  /chat/stream            (Server-Sent Events)
                          └─► ChatEngine
                                ├── Topic guard  (reject off-topic)
                                ├── Semantic query cache  (cosine ≤ 0.08)
                                ├── Hybrid RAG  (70% pgvector + 30% BM25)
                                ├── Citation RAG  (per-article vectors)
                                └── LLM backend
                                      ├── Groq  llama-3.3-70b  ← default
                                      ├── Ollama  (local server)
                                      └── Local LoRA adapter  (QLoRA 4-bit)

PostgreSQL 16 + pgvector  (single instance — no separate vector DB)
  ├── query_logs            legacy query history
  ├── qa_interactions       QA pairs + 384-d embeddings (semantic cache)
  ├── user_feedback         thumbs up / down ratings
  ├── users                 JWT accounts
  ├── conversations         chat sessions per user
  ├── conversation_messages messages + RAG source citations
  ├── document_chunks       main RAG index  (768-d, HNSW)
  └── citation_chunks       per-article citation index  (384-d, HNSW)

MLflow  (port 5000)
  ├── Tracks SFT / GRPO / DPO training runs
  ├── Model Registry: regllm-lora-adapter
  └── Artifact storage → S3 in production
```

---

## AWS Architecture

```
                        ┌──────────────────────────┐
  Internet ──► Route 53 │  ALB (Application LB)    │
                        │  HTTPS :443               │
                        └──────────────┬───────────┘
                                       │
               ┌───────────────────────┼──────────────────────────┐
               ▼                       ▼                           ▼
   ECS Fargate: frontend     ECS Fargate: fastapi        RDS PostgreSQL 16
   (Next.js, port 3000)      (FastAPI, port 8000)        + pgvector extension
                              reads/writes DB ──────────► (all tables + vectors)

   ECR                        S3 (×2)                  Secrets Manager
   regllm-api:{sha}           mlflow-artifacts         db_password
   regllm-frontend:{sha}      model-weights            jwt_secret
                                                        groq_api_key
```

> **No EFS, no NAT Gateway.** pgvector replaces ChromaDB (eliminates EFS volume). ECS tasks run in public subnets (eliminates NAT Gateway). Both are the biggest cost drivers in a small deployment.

### AWS Cost Estimate — 10 users/month

| Service | Config | $/month |
|---------|--------|---------|
| RDS PostgreSQL t3.micro | 20 GB gp2, single-AZ | ~$15 |
| ECS Fargate — fastapi | 0.5 vCPU / 1 GB RAM, ~8 h/day | ~$5 |
| ECS Fargate — frontend | 0.25 vCPU / 0.5 GB RAM, ~8 h/day | ~$3 |
| ALB | 1 instance | ~$16 |
| ECR | ~2 GB stored | ~$0.20 |
| S3 | MLflow artifacts | ~$0.10 |
| Secrets Manager | 3 secrets | ~$0.12 |
| **Total** | | **~$39/month** |

**Further savings:**
- Scale ECS to 0 tasks outside business hours → ~$8 off
- Switch to Aurora Serverless v2 if traffic is bursty → pay per ACU-second
- Groq free tier comfortably covers 10 users

---

## Local Development

### Prerequisites

- Docker + Docker Compose  
- `GROQ_API_KEY` from [console.groq.com](https://console.groq.com)

### Quick start

```bash
git clone https://github.com/your-org/regllm.git && cd regllm
cp .env.example .env          # fill in GROQ_API_KEY and JWT_SECRET
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| FastAPI docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| PostgreSQL | localhost:5433 |

### Create a user

Register at `http://localhost:3000/register`, or:

```bash
curl -s -X POST http://localhost:8000/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@example.com","password":"changeme"}' | jq .
```

### Load regulation documents

```bash
docker compose exec fastapi python -c "
from src.rag_system import RegulatoryRAGSystem
rag = RegulatoryRAGSystem()
rag.load_from_json('data/raw/regulations.json')
print(rag.collection.count(), 'chunks indexed')
"
```

---

## Database Migrations (Alembic)

`Base.metadata.create_all()` is kept for frictionless local dev. **Production deployments use Alembic exclusively** to avoid accidental schema drift on a persistent RDS instance.

### Apply all migrations (first deploy)

```bash
POSTGRES_HOST=your-rds.rds.amazonaws.com \
POSTGRES_PASSWORD=<from Secrets Manager> \
alembic upgrade head
```

### Create a migration after changing `src/db.py`

```bash
alembic revision --autogenerate -m "add column X to table Y"
# review the generated file in alembic/versions/
alembic upgrade head
```

### Rollback

```bash
alembic downgrade -1     # one revision back
alembic downgrade base   # full teardown
```

---

## Security

| Concern | Implementation |
|---------|---------------|
| Auth rate limiting | 5 req/min per IP on `/auth/login` and `/auth/register` (slowapi) |
| JWT secret | Secrets Manager in production; never embedded in image or task definition |
| CORS | Locked to `CORS_ORIGINS` env var (comma-separated); defaults to `http://localhost:3000` |
| DB passwords | Auto-generated 32-char random by Terraform, injected via Secrets Manager |
| SQL injection | SQLAlchemy ORM + parameterized queries throughout |
| TLS | ALB terminates HTTPS; internal container traffic stays in VPC |

---

## Building Docker Images

### API image

```bash
docker build -f Dockerfile.api -t regllm-api:latest .
```

The build does these things in order:

1. Installs CPU-only PyTorch (prevents sentence-transformers from pulling ~1 GB CUDA packages)
2. Installs `requirements-api.txt`
3. **Bakes both embedding models into the image** — `paraphrase-multilingual-mpnet-base-v2` (768-d) and `MiniLM-L12-v2` (384-d). Without this, ECS health checks fail during the 2-minute cold-start model download.

Expected image size: ~3 GB (models take ~1.5 GB).

### Frontend image

```bash
docker build -f frontend/Dockerfile -t regllm-frontend:latest ./frontend
```

Multi-stage build: `node:20-alpine` builder → `node:20-alpine` runner with only `.next/standalone/` (~50 MB final image).

---

## CI/CD Pipeline

### Overview

```
Every push / PR
  └─► test.yml
        ├── pgvector service container
        ├── pip install (CPU torch + requirements-api.txt)
        └── pytest -m "not slow and not llm_judge"

Push to main (or manual trigger)
  └─► deploy.yml
        ├── OIDC → assume AWS IAM role (no long-lived keys)
        ├── docker buildx → ECR (GHA layer cache speeds up rebuild)
        │     regllm-api:{sha}
        │     regllm-frontend:{sha}
        ├── download current ECS task definition JSON
        ├── swap image digest
        ├── register new task definition revision
        └── aws ecs update-service --force-new-deployment
              └── ECS rolling deploy
                    ├── new tasks start, pass health check → old tasks drain
                    └── if health check fails → auto-rollback (circuit breaker)
```

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `AWS_DEPLOY_ROLE_ARN` | IAM role ARN for OIDC — no access keys stored in GitHub |
| `AWS_REGION` | e.g. `eu-west-1` |

### test.yml

```yaml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env: { POSTGRES_DB: regllm, POSTGRES_USER: regllm, POSTGRES_PASSWORD: changeme }
        ports: ["5432:5432"]
        options: --health-cmd "pg_isready" --health-interval 5s --health-retries 10
    env:
      POSTGRES_HOST: localhost
      POSTGRES_PASSWORD: changeme
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install torch --index-url https://download.pytorch.org/whl/cpu -q
      - run: pip install -r requirements-api.txt pytest pytest-asyncio -q
      - run: pytest -m "not slow and not llm_judge" -x
```

### deploy.yml (key steps)

```yaml
on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      environment: { type: choice, options: [dev, prod], default: dev }

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      id-token: write    # OIDC token
      contents: read
    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_DEPLOY_ROLE_ARN }}
          aws-region: ${{ secrets.AWS_REGION }}

      - uses: aws-actions/amazon-ecr-login@v2

      - uses: docker/setup-buildx-action@v3

      # Build + push API image (GHA layer cache)
      - uses: docker/build-push-action@v5
        with:
          context: .
          file: Dockerfile.api
          push: true
          tags: ${{ env.ECR_REGISTRY }}/regllm-api:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      # Build + push frontend image
      - uses: docker/build-push-action@v5
        with:
          context: ./frontend
          push: true
          tags: ${{ env.ECR_REGISTRY }}/regllm-frontend:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      # Rolling ECS deploy with circuit-breaker auto-rollback
      - name: Deploy API to ECS
        run: |
          TASK_DEF=$(aws ecs describe-task-definition \
            --task-definition regllm-api --query taskDefinition)
          NEW_TASK=$(echo "$TASK_DEF" | jq \
            --arg img "${{ env.ECR_REGISTRY }}/regllm-api:${{ github.sha }}" \
            '.containerDefinitions[0].image = $img
             | del(.taskDefinitionArn,.revision,.status,
                   .requiresAttributes,.placementConstraints,
                   .compatibilities,.registeredAt,.registeredBy)')
          aws ecs register-task-definition --cli-input-json "$NEW_TASK"
          aws ecs update-service \
            --cluster regllm \
            --service regllm-api \
            --task-definition regllm-api \
            --deployment-configuration \
              "deploymentCircuitBreaker={enable=true,rollback=true}" \
            --force-new-deployment
          aws ecs wait services-stable \
            --cluster regllm --services regllm-api
```

---

## Terraform — AWS Infrastructure

```bash
cd infra

# One-time bootstrap: create S3 state bucket + DynamoDB lock table manually
# Then:
terraform init -backend-config=environments/dev.tfvars
terraform plan  -var-file=environments/dev.tfvars
terraform apply -var-file=environments/dev.tfvars
```

### Module map

| Module | Key resources |
|--------|--------------|
| `ecr` | Two ECR repos (api + frontend); lifecycle: keep last 10 images |
| `s3` | `regllm-mlflow-artifacts-{env}` + `regllm-model-weights-{env}` (versioned) |
| `secrets` | Secrets Manager entries for `db_password`, `jwt_secret`, `groq_api_key` |
| `rds` | PostgreSQL 16, `shared_preload_libraries=pg_vector`, encrypted, delete-protection on prod |
| `alb` | ALB + target groups for port 8000 and 3000 |
| `ecs` | Fargate cluster, task definitions, services; IAM roles for Secrets + S3 |

### Post-apply steps

```bash
# 1. Run database migrations
POSTGRES_HOST=$(terraform output -raw rds_endpoint) \
POSTGRES_PASSWORD=$(aws secretsmanager get-secret-value \
  --secret-id regllm/dev/db_password \
  --query SecretString --output text) \
alembic upgrade head

# 2. Set your Groq API key
aws secretsmanager put-secret-value \
  --secret-id regllm/dev/groq_api_key \
  --secret-string "gsk_..."
```

---

## ChromaDB → pgvector Migration

ChromaDB has been removed. All vectors are stored in PostgreSQL.

| Before | After |
|--------|-------|
| `regulacion_bancaria` ChromaDB collection (768-d) | `document_chunks` table + HNSW index |
| `regulation_citations` ChromaDB collection (384-d) | `citation_chunks` table + HNSW index |
| EFS volume (ChromaDB persistence on ECS) | Same RDS instance |
| `./vector_db/chroma_db/` local directory | `postgres_data` Docker volume |

**Benefits:** one fewer service, eliminates EFS cost, transactional consistency with application data, same backup/restore procedure as the rest of the DB.

---

## Model Training

```bash
# SFT
python scripts/train_combined.py --epochs 3 --lr 1e-4

# GRPO (reward from keyword/citation/format quality)
python -m src.rlhf.grpo_trainer --epochs 2

# DPO (from collected thumbs up/down pairs)
python scripts/export_dpo_pairs.py    # builds data/finetuning/dpo_pairs.jsonl
python -m src.rlhf.dpo_trainer
```

Training runs appear in MLflow at `http://localhost:5000`.

### Promote to production

```bash
# MLflow CLI:
mlflow models transition-version \
  --name regllm-lora-adapter --version 3 --stage Production

# The container's entrypoint will pull this version on next cold start
# (when REGLLM_BACKEND=local)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REGLLM_BACKEND` | `groq` | LLM backend: `groq`, `ollama`, `local` |
| `GROQ_API_KEY` | — | Required for groq backend |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model ID |
| `POSTGRES_HOST` | `localhost` | DB hostname |
| `POSTGRES_PORT` | `5432` | DB port |
| `POSTGRES_DB` | `regllm` | Database name |
| `POSTGRES_USER` | `regllm` | DB user |
| `POSTGRES_PASSWORD` | `changeme` | **Change in production** |
| `JWT_SECRET` | `change-me-in-production` | **Change in production** |
| `JWT_EXPIRE_HOURS` | `168` | Token validity (7 days) |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Project Structure

```
regllm/
├── app.py                      Gradio UI entry point (legacy)
├── config.py                   Central config for all components
├── api/
│   ├── main.py                 FastAPI app + lifespan + CORS + rate limiter
│   ├── auth.py                 JWT signing + bcrypt
│   ├── deps.py                 FastAPI dependencies (get_db, get_current_user)
│   ├── models.py               Pydantic request/response schemas
│   └── routers/
│       ├── auth.py             POST /auth/register|login, GET /auth/me
│       ├── conversations.py    CRUD /conversations
│       └── chat.py             POST /chat/stream  (SSE)
├── src/
│   ├── chat_engine.py          Query pipeline: guard → cache → RAG → LLM → parse
│   ├── rag_system.py           Hybrid RAG using pgvector + BM25 (document_chunks)
│   ├── citation_rag.py         Per-article citation vectors (citation_chunks)
│   ├── cache.py                Semantic query cache (cosine similarity on qa_interactions)
│   ├── db.py                   Async SQLAlchemy ORM + pgvector helpers
│   ├── verification.py         Hallucination detection + confidence scoring
│   └── training/ + rlhf/       SFT, GRPO, DPO trainers
├── alembic/                    Database migrations
│   ├── env.py                  Reads DB URL from env vars
│   └── versions/
│       └── 0001_initial_schema.py
├── scripts/
│   ├── train_combined.py       Main training entry point
│   ├── export_dpo_pairs.py     Export thumbs-up/down pairs for DPO
│   └── eval_benchmark.py       Keyword-F1 evaluation
├── frontend/                   Next.js 14 TypeScript app
│   ├── app/                    App Router pages (auth + chat)
│   ├── components/             Sidebar, ChatWindow, SourceDrawer, etc.
│   └── Dockerfile              Multi-stage: builder + minimal runner
├── infra/                      Terraform modules for AWS
├── .github/workflows/
│   ├── test.yml                pytest on every push/PR
│   └── deploy.yml              ECR push + ECS rolling deploy on main
├── Dockerfile.api              FastAPI image (models baked in)
├── docker-compose.yml          Local dev: postgres + mlflow + fastapi + frontend
├── requirements-api.txt        FastAPI dependencies (no torch for training)
└── alembic.ini                 Alembic configuration
```

---

## Tests

```bash
# Fast unit tests (no GPU, no network)
pytest -m "not slow and not llm_judge" -x

# Integration tests (requires PostgreSQL)
POSTGRES_HOST=localhost pytest -m integration

# CI runs both automatically on every PR via .github/workflows/test.yml
```

---

## Disclaimer

RegLLM is for research and educational purposes. Do not use it as the sole basis for regulatory compliance decisions. Always consult official documents and qualified compliance professionals.

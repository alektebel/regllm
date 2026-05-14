# RegLLM

**A fine-tuned language model for Spanish banking regulation.**

RegLLM answers hard questions about EBA guidelines, CRR/CRD IV, and Basel III/IV by combining a fine-tuned Qwen2.5-7B with hybrid RAG over real regulatory documents. It runs fully on your machine, through a cloud API, or deployed to AWS — your choice.

---

## Why it exists

Banking regulation is dense, cross-referenced, and multilingual. A generic LLM will hallucinate article numbers, confuse CRR with CRD, and miss the nuance between IRBA and standardised approaches. RegLLM is trained specifically on the documents that matter — EBA guidelines, CRR/CRD IV, Basel III/IV, and five Spanish bank annual reports — and uses RAG to ground every answer in the actual text.

Ask it things like:

> *"¿Cuál es el tratamiento del riesgo de crédito contraparte bajo CRR para derivados OTC?"*
>
> *"¿Qué requisitos de capital exige Basilea IV para carteras IRBA con LGD estimada?"*
>
> *"Compara el método estándar con el IRB avanzado para el cálculo de APR."*

It returns an answer with citations to the relevant article and paragraph — and rejects off-topic questions outright.

---

## How it works

```
User query
    │
    ▼
Topic Guard ── off-topic ──→  Rejection  (no LLM call wasted)
    │ on-topic
    ▼
Hybrid RAG   70% semantic cosine  +  30% BM25
    │         ChromaDB · paraphrase-multilingual-mpnet · 1 500-char chunks
    ▼
Prompt assembly   system  +  RAG context  +  history  +  question
    │
    ├── local  ──→  Qwen2.5-7B-Instruct  +  LoRA adapter  (4-bit QLoRA)
    ├── groq   ──→  llama-3.3-70b-versatile  via Groq API
    └── ollama ──→  regllm GGUF  on local Ollama server
    │
    ▼
Response parser  +  citation enricher  +  hallucination checker
    │
    ▼
Gradio UI   ←──  PostgreSQL  (query logs · user feedback · embeddings)
```

The three backends share the same RAG pipeline and chat engine. Swapping between them is a single CLI flag.

---

## Quick start

### Option A — Groq API (no GPU, 30-second setup)

```bash
git clone https://github.com/your-org/regllm.git && cd regllm
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu -q
pip install -r requirements-prod.txt

echo "GROQ_API_KEY=gsk_..." > .env
python app.py --backend groq
```

Open **http://localhost:7860**

### Option B — Local model (private, no API key)

On first run the app downloads the LoRA adapter automatically (≈ 300 MB) and the Qwen2.5-7B base model via HuggingFace (≈ 14 GB, cached after the first download):

```bash
pip install -r requirements.txt   # includes torch + bitsandbytes
python app.py --backend local
```

To use a specific adapter:

```bash
python app.py --backend local --adapter models/finetuned/run_20260222_224503/final_model
```

To publish and consume your own adapter from HuggingFace Hub:

```bash
# Set in .env:  ADAPTER_HF_REPO=your-org/regllm-adapter
python app.py --backend local
```

### Option C — Docker Compose (app + MLflow + Postgres)

```bash
cp .env.example .env          # add GROQ_API_KEY and POSTGRES_PASSWORD
docker compose up
```

| Service | URL | Purpose |
|---------|-----|---------|
| app     | http://localhost:7860 | Gradio UI |
| mlflow  | http://localhost:5000 | Experiment tracker |
| postgres | localhost:5432 | Query logs + MLflow backend |

---

## Training your own model

### 1. Gather data

```bash
# Scrape regulatory documents into data/raw/
python src/scraper/regulation_scraper.py

# Generate Q&A pairs (GPU)
python scripts/generate_qa_from_docs.py --input data/raw/

# Or use Ollama if you have no GPU
python scripts/generate_qa_from_docs.py --input data/raw/ --backend ollama --model llama3.2

# Index documents into ChromaDB
python scripts/index_citations.py
```

### 2. Fine-tune (SFT)

`train_combined.py` auto-discovers every JSONL under `data/finetuning/` and resumes from the latest checkpoint automatically:

```bash
python scripts/train_combined.py --epochs 5 --lr 1e-4
```

Every run is tracked in MLflow automatically. Open http://localhost:5000 to compare runs, view loss curves, and browse saved adapters.

### 3. RLHF

```bash
# GRPO — test-based rewards (keyword overlap · citation matching · format quality)
python -m src.rlhf.grpo_trainer --epochs 2

# DPO — learns from thumbs-up / thumbs-down collected in the Gradio UI
python -m src.rlhf.dpo_trainer
```

### 4. Promote a model to production

```bash
# In the MLflow UI (http://localhost:5000) or via CLI:
mlflow models transition-version \
  --name regllm-lora-adapter \
  --version 3 \
  --stage Production

# The app (and Docker container) will serve this version automatically on next startup
```

### 5. Evaluate

```bash
python scripts/eval_qa.py          # keyword F1 + token F1 on ground-truth set
python scripts/validate_model.py   # embedding similarity + citation quality
python scripts/test_with_judge.py  # LLM-as-judge scoring
```

---

## MLflow experiment tracking

Every training run logs automatically:

| What | Detail |
|------|--------|
| Hyperparameters | base model, LoRA rank/alpha, learning rate, epochs, batch size, dataset size |
| Metrics | train loss and eval loss at every step |
| Artifacts | full LoRA adapter (adapter_model.safetensors + config) |
| Model Registry | each adapter registered as a versioned `regllm-lora-adapter` |

On AWS the artifact backend is S3. Locally it's `./mlruns/`.

---

## Model details

| Property | Value |
|----------|-------|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Fine-tuning | LoRA (r=64, α=128) across 7 attention + FFN projection layers |
| Quantisation | 4-bit NF4 QLoRA via BitsAndBytes |
| Post-training | GRPO (test-based rewards) + DPO (user preference pairs) |
| Languages | Spanish (primary), English |
| Training corpus | EBA guidelines · CRR/CRD IV · Basel III/IV · 5 Spanish bank reports (2022–2023) |
| Context window | 4 096 tokens |
| Typical latency | 200–500 ms on GPU · 8–15 s on CPU |

### Hardware requirements

| Use case | VRAM needed |
|----------|------------|
| Inference (4-bit) | 6 GB |
| Training with LoRA (4-bit) | 12 GB |
| CPU inference | 16 GB RAM |

No GPU? Run `--backend groq` or `--backend ollama` for CPU-only operation.

---

## Deployment on AWS

Infrastructure is fully defined in Terraform under `infra/`. One command provisions everything:

```bash
cd infra
terraform init
terraform apply -var-file=environments/dev.tfvars
```

What gets created:

| Resource | Purpose |
|----------|---------|
| ECR | Docker image registry |
| ECS Fargate | Serverless container for the app |
| RDS PostgreSQL 16 | Query logs + MLflow metadata |
| S3 (×2) | MLflow artifacts + raw model weights |
| EFS | ChromaDB persistence across task restarts |
| ALB | Public HTTPS entry point |
| Secrets Manager | API keys and database password |

After applying, every `git push` to `main` triggers a GitHub Actions workflow that builds the image, pushes to ECR, and does a rolling ECS deploy with automatic rollback on failure.

See [infra/README.md](infra/README.md) for the full bootstrap guide.

---

## Project structure

```
regllm/
├── app.py                      # Entry point — Gradio UI (local / groq / ollama)
├── config.py                   # Central config for all components
├── src/
│   ├── chat_engine.py          # Query pipeline: topic guard → RAG → prompt → parse
│   ├── rag_system.py           # ChromaDB + BM25 hybrid retrieval
│   ├── citation_rag.py         # Per-article citation vectors
│   ├── verification.py         # Hallucination detection + confidence scoring
│   ├── db.py                   # Async PostgreSQL (logs, feedback, embeddings)
│   ├── training/
│   │   ├── train.py            # SFT training class
│   │   └── model_setup.py      # Model loading + LoRA setup
│   └── rlhf/
│       ├── grpo_trainer.py     # Group Relative Policy Optimization
│       ├── dpo_trainer.py      # Direct Preference Optimization
│       └── grpo_rewards.py     # Keyword / source / format reward functions
├── scripts/
│   ├── train_combined.py       # Main training entry point (auto-discovers data)
│   ├── entrypoint.sh           # Container startup: pulls adapter from MLflow / S3
│   ├── eval_qa.py              # Evaluation: keyword F1 + token F1
│   └── validate_model.py       # Comprehensive validation suite
├── infra/                      # Terraform: ECR, ECS, RDS, S3, EFS, ALB, Secrets
│   ├── modules/                # One module per AWS service
│   └── environments/           # dev.tfvars  /  prod.tfvars
├── .github/workflows/
│   ├── test.yml                # pytest on every push and PR
│   └── deploy.yml              # ECR push + ECS rolling deploy on main
├── data/
│   ├── raw/                    # Source PDFs and regulatory documents
│   ├── processed/              # Processed Q&A datasets (JSONL)
│   └── finetuning/             # Training datasets (auto-discovered)
└── models/finetuned/           # LoRA adapter checkpoints (run_YYYYMMDD_HHMMSS/)
```

---

## Environment variables

| Variable | Required for | Description |
|----------|-------------|-------------|
| `GROQ_API_KEY` | groq backend | Groq API key (`gsk_...`) |
| `POSTGRES_HOST` | DB logging | PostgreSQL host (default: localhost) |
| `POSTGRES_PASSWORD` | DB logging | Database password |
| `MLFLOW_TRACKING_URI` | MLflow | Tracking server URL (default: http://localhost:5000) |
| `ADAPTER_HF_REPO` | auto-download | HuggingFace Hub repo, e.g. `your-org/regllm-adapter` |
| `MODEL_S3_URI` | auto-download | S3 URI for adapter sync, e.g. `s3://bucket/adapters/latest/` |
| `REGLLM_BACKEND` | Docker | Backend for container startup (default: `groq`) |

Copy `.env.example` to `.env` and fill in what you need.

---

## Tests

```bash
pytest -m "not slow and not llm_judge"   # unit tests, no GPU needed
pytest -m "integration"                  # requires PostgreSQL running
```

CI runs both automatically on every pull request via GitHub Actions.

---

## Disclaimer

RegLLM is for research and educational purposes. Do not use it as the sole basis for regulatory compliance decisions. Always consult official documents and qualified compliance professionals.

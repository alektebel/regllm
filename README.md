# RegLLM — Regulatory LLM for Banking & Credit Risk

A complete pipeline for fine-tuning open-source LLMs on banking regulation and credit risk documents. Includes automatic QA pair generation from PDFs, supervised fine-tuning, RLHF (GRPO/DPO), a RAG system, and a production-ready REST API.

---

## Table of Contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Data Pipeline](#data-pipeline)
6. [QA Pair Generation from Documents](#qa-pair-generation-from-documents)
7. [Fine-tuning](#fine-tuning)
8. [Reinforcement Learning (GRPO)](#reinforcement-learning-grpo)
9. [REST API](#rest-api)
10. [Configuration](#configuration)
11. [Project Structure](#project-structure)
12. [Hardware Requirements](#hardware-requirements)
13. [Troubleshooting](#troubleshooting)

---

## What it does

RegLLM trains language models to answer questions about:

- **Banking regulation**: EBA Guidelines, CRR/CRD, Basel accords
- **Credit risk methodology**: PD/LGD/EAD estimation, IRB, IFRS 9, stress testing
- **Spanish bank financials**: Santander, BBVA, CaixaBank, Sabadell, Kutxabank (2022–2023)
- **SQL methodology review**: validates credit risk SQL code against regulatory standards

Key capabilities:

| Feature | Description |
|---|---|
| **QA Generator** | Automatically creates training pairs from any regulation PDF/text |
| **RAG** | Hybrid semantic + BM25 search over your document corpus |
| **Fine-tuning** | SFT with LoRA on Qwen, Phi, Gemma, or any HF model |
| **RLHF** | GRPO/DPO with domain-specific reward functions |
| **Citation tracking** | Hierarchical regulation → article → paragraph → point |
| **Verification** | LLM judge scoring hallucinations and missing facts |
| **REST API** | FastAPI server with PostgreSQL logging |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
│   PDFs / Text / URLs  →  QA Generator  →  JSONL training data  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      TRAINING PIPELINE                          │
│                                                                 │
│  SFT (LoRA)  →  GRPO (RLHF)  →  DPO  →  Merged model          │
│                                                                 │
│  Reward functions: keyword overlap · source matching · format   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                       INFERENCE LAYER                           │
│                                                                 │
│  RAG (ChromaDB + BM25)  →  LLM  →  Verification  →  Citations  │
│                                                                 │
│  FastAPI  ·  Gradio UI  ·  CLI  ·  Python SDK                   │
└─────────────────────────────────────────────────────────────────┘
```

### Core modules

| Module | Path | Purpose |
|---|---|---|
| **QA Generator** | `scripts/generate_qa_from_docs.py` | Generates training QA pairs from documents |
| **RAG System** | `src/rag_system.py` | ChromaDB + BM25 retrieval |
| **Verification** | `src/verification.py` | Hallucination detection, confidence scoring |
| **Citation Tree** | `src/citation_tree.py` | Hierarchical regulation citation tracker |
| **LLM Judge** | `src/llm_judge.py` | Evaluates response quality vs. ground truth |
| **GRPO Trainer** | `src/rlhf/grpo_trainer.py` | Reinforcement learning from rewards |
| **DPO Trainer** | `src/rlhf/dpo_trainer.py` | Learning from preference pairs |
| **API** | `api/main.py` | FastAPI REST server |
| **DB** | `src/db.py` | Async PostgreSQL query logging |

---

## Quick Start

```bash
# 1 — Install
git clone <repo> && cd regllm
pip install -r requirements.txt

# 2 — Generate QA pairs (GPU, Qwen2.5-7B)
python scripts/generate_qa_from_docs.py --docs-dir data/raw

# 2b — No GPU? Use Ollama  (run once: ollama pull llama3.2)
python scripts/generate_qa_from_docs.py --docs-dir data/raw --backend ollama --model llama3.2

# 3 — Fine-tune on all current datasets
python scripts/train_combined.py

# 4 — Chat with the trained model
python scripts/use_banking_model.py --mode interactive \
  --model-path models/finetuned/run_20260220_191334/final_model

# 5 — Or run the REST API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## Full Workflow (Step by Step)

### Step 1 — Generate QA pairs in Spanish from regulation documents

The generator uses **RAG + hybrid search** (ChromaDB + BM25) to find related passages already indexed, and the **Regulation Citation Tree (RCT)** to resolve exact citation paths (e.g. `CRR > Article 92 > Paragraph 1`). Both are injected into the prompt so answers contain precise regulatory references.

```bash
# With GPU — default Qwen2.5-7B-Instruct, 3 pairs per chunk, Spanish output
python scripts/generate_qa_from_docs.py --docs-dir data/raw

# Lighter model (less VRAM)
python scripts/generate_qa_from_docs.py \
  --docs-dir data/raw \
  --model Qwen/Qwen2.5-3B-Instruct

# Without GPU — Ollama backend (install Ollama from https://ollama.ai first)
ollama pull llama3.2
python scripts/generate_qa_from_docs.py \
  --docs-dir data/raw \
  --backend ollama \
  --model llama3.2

# More pairs per chunk, limit to 5 documents for a quick test
python scripts/generate_qa_from_docs.py \
  --docs-dir data/raw \
  --pairs-per-chunk 5 \
  --max-docs 5

# Output always goes to:
#   data/finetuning/generated_qa.jsonl
```

On first run the script indexes all documents into ChromaDB, then generates QA pairs with RAG-enriched prompts. Progress streams token by token in the terminal. Each extracted pair is printed in formatted Spanish before being saved.

---

### Step 2 — Train the model on all current datasets

`train_combined.py` automatically loads **all four JSONL files** from `data/finetuning/`:

| File | Examples |
|---|---|
| `banking_qa_dataset.jsonl` | 82 |
| `banking_annual_accounts_extra.jsonl` | 12 |
| `sql_methodology_comparison_dataset.jsonl` | 20 |
| `generated_qa.jsonl` | your generated pairs |

```bash
# Full training run  (Qwen2.5-7B, 5 epochs, batch 2, LoRA r=32)
python scripts/train_combined.py

# The model is saved to:
#   models/finetuned/run_<timestamp>/final_model/
```

Training settings (hardcoded in the script, edit to change):
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Epochs: 5, batch size: 2, gradient accumulation: 8 (effective batch: 16)
- Learning rate: 1e-4, LoRA rank: 32, alpha: 64
- Banking data repeated 3× to emphasise factual memorisation
- Saves best checkpoint by eval loss

---

### Step 3 — Run inference

#### Interactive chat (terminal)

```bash
python scripts/use_banking_model.py \
  --mode interactive \
  --model-path models/finetuned/run_20260220_191334/final_model
```

#### Single question

```bash
python scripts/use_banking_model.py \
  --mode single \
  --model-path models/finetuned/run_20260220_191334/final_model \
  --question "¿Cuál fue el beneficio neto de BBVA en 2023?"
```

#### REST API (FastAPI + Swagger UI)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Swagger UI → http://localhost:8000/docs

# Ask a question
curl -X POST http://localhost:8000/consultar \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Qué es el ratio CET1?"}'

# Search documents
curl -X POST http://localhost:8000/buscar \
  -H "Content-Type: application/json" \
  -d '{"query": "requisitos de capital Basilea III", "n_results": 5}'
```

---

## Installation

### Requirements

- Python 3.10+
- CUDA 11.8+ (for local GPU inference/training) — or Ollama for CPU-only use

```bash
# Core dependencies
pip install -r requirements.txt

# Banking data extras (PDF extraction)
pip install -r requirements-banking.txt

# System package for PDF processing
sudo apt-get install poppler-utils

# Copy and edit environment variables
cp .env.example .env
```

### Environment variables (`.env`)

```env
# PostgreSQL (optional — API works without it)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=regllm
DB_USER=postgres
DB_PASSWORD=yourpassword

# API
API_HOST=0.0.0.0
API_PORT=8000

# Override model at runtime
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

---

## Data Pipeline

### 1. Use the ready-made datasets

The repo includes three datasets in `data/finetuning/`:

| File | Examples | Domain |
|---|---|---|
| `banking_qa_dataset.jsonl` | 82 | Spanish bank financials 2022–2023 |
| `sql_methodology_comparison_dataset.jsonl` | 20 | Credit risk SQL vs. EBA guidelines |
| `banking_annual_accounts_extra.jsonl` | 12 | Additional accounting Q&A |

### 2. Generate banking synthetic data

```bash
python scripts/generate_example_data.py
# → data/finetuning/banking_qa_dataset.jsonl
# → data/processed/{bank}/{bank}_{year}.json
```

### 3. Download real financial reports (PDFs)

```bash
python scripts/download_financial_reports.py
# → data/raw/{bank}/  (PDFs)
# → data/processed/{bank}/  (extracted metrics)
```

### 4. Scrape regulation documents

Add URLs to `regurl.txt` then:

```bash
# Via API (must have server running)
curl -X POST http://localhost:8000/scrape \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://www.eba.europa.eu/..."]}'
```

### 5. Inspect and validate datasets

```bash
python scripts/dataset_utils.py validate   # Check JSONL structure
python scripts/dataset_utils.py analyze    # Question type distribution
python scripts/dataset_utils.py stats      # Token counts, lengths
python scripts/dataset_utils.py samples    # Print random examples
```

### 6. Export to CSV

```bash
python scripts/export_datasets.py
# → data/exports/banking_qa.csv
# → data/exports/sql_methodology_comparison.csv
```

---

## QA Pair Generation from Documents

**`scripts/generate_qa_from_docs.py`** reads regulation documents (PDFs, TXT, JSON, Markdown) and uses a local open-source LLM to produce question–answer pairs ready for fine-tuning.

### Basic usage

```bash
# From a folder of documents (GPU, Qwen2.5-7B)
python scripts/generate_qa_from_docs.py \
  --docs-dir data/raw \
  --output data/finetuning/generated_qa.jsonl

# From a single PDF
python scripts/generate_qa_from_docs.py \
  --docs-dir path/to/single_doc \
  --output data/finetuning/generated_qa.jsonl

# Lighter model for less VRAM
python scripts/generate_qa_from_docs.py \
  --docs-dir data/raw \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output data/finetuning/generated_qa.jsonl
```

### Using Ollama (no GPU required)

```bash
# Install Ollama from https://ollama.ai, then:
ollama pull llama3.2          # ~2 GB, fast
# or
ollama pull qwen2.5:7b        # higher quality

python scripts/generate_qa_from_docs.py \
  --docs-dir data/raw \
  --backend ollama \
  --model llama3.2 \
  --output data/finetuning/generated_qa.jsonl
```

### All options

| Option | Default | Description |
|---|---|---|
| `--docs-dir` | `data/raw` | Directory with PDFs / TXTs / JSONs |
| `--output` | `data/finetuning/generated_qa.jsonl` | Output JSONL path |
| `--backend` | `transformers` | `transformers` or `ollama` |
| `--model` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model ID or Ollama model name |
| `--pairs-per-chunk` | `3` | QA pairs to generate per text chunk |
| `--chunk-size` | `800` | Words per chunk |
| `--max-docs` | unlimited | Limit number of documents processed |
| `--lang` | `es` | `es` (Spanish) or `en` (English) |
| `--no-quantize` | off | Disable 4-bit quantization (requires more VRAM) |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL |

### Supported document formats

| Extension | How it's read |
|---|---|
| `.pdf` | PyPDF2 (+ pdfplumber fallback) |
| `.txt` | Plain text |
| `.md` | Markdown as plain text |
| `.json` | Extracts `text`, `content`, or `body` fields |
| `.jsonl` | Each line treated as a document |

### Output format

Each line in the output JSONL is ready for SFT:

```json
{
  "messages": [
    {"role": "system", "content": "Eres un experto en regulación bancaria..."},
    {"role": "user",   "content": "¿Qué establece el artículo 92 del CRR sobre los requisitos de capital?"},
    {"role": "assistant", "content": "El artículo 92 del CRR establece que las entidades deberán mantener..."}
  ],
  "metadata": {
    "source_file": "crr_regulation.pdf",
    "chunk_index": 4,
    "generated_at": "2026-02-20T10:30:00"
  }
}
```

---

## Fine-tuning

### Supervised Fine-Tuning (SFT)

```bash
# Quick overfitting test on a small subset (~2 min)
python scripts/train_combined.py --small-subset

# Full training with defaults (Qwen2.5-7B, 3 epochs)
python scripts/train_combined.py

# Custom run
python scripts/train_combined.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --epochs 3 \
  --batch-size 2 \
  --lr 2e-4 \
  --output models/my_run
```

The script automatically combines all JSONL files found in `data/finetuning/`.

### End-to-end pipeline

```bash
# Downloads data → generates QA → trains → evaluates in one command
python scripts/run_full_pipeline.py
```

### Inference with a trained model

```bash
# Interactive chat
python scripts/use_banking_model.py --mode interactive --model-path models/finetuned/run_1

# Single question
python scripts/use_banking_model.py \
  --mode single \
  --model-path models/finetuned/run_1 \
  --question "¿Cuál fue el beneficio neto de BBVA en 2023?"

# Batch from file
python scripts/use_banking_model.py \
  --mode batch \
  --model-path models/finetuned/run_1 \
  --input questions.txt
```

### LLaMA-Factory (recommended for larger runs)

```bash
pip install llamafactory
llamafactory-cli train examples/llamafactory_config.yaml
```

---

## Reinforcement Learning (GRPO)

GRPO (Group Relative Policy Optimization) improves the model using deterministic rewards.

### Reward functions (`src/rlhf/grpo_rewards.py`)

| Reward | Weight | Measures |
|---|---|---|
| Keyword overlap | 0.5 | Key facts from ground truth in response |
| Source matching | 0.3 | Correct regulation references cited |
| Format quality | 0.2 | Response length, language, structure |

```bash
# Run GRPO training
python scripts/run_grpo.py
# → models/grpo/final_model/
```

### DPO (preference pairs)

```bash
# Collect preference data into data/preferences/feedback.jsonl first, then:
python -c "from src.rlhf.dpo_trainer import train_dpo; train_dpo()"
```

---

## REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Swagger UI: http://localhost:8000/docs
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/stats` | System statistics |
| `POST` | `/consultar` | Ask a regulatory question |
| `POST` | `/buscar` | Search documents by query |
| `POST` | `/documentos/agregar` | Add a single document |
| `POST` | `/documentos/cargar-json` | Load a JSON file of documents |
| `POST` | `/scrape` | Scrape URL(s) and index content |
| `GET` | `/logs` | Query interaction logs |

### Examples

```bash
# Ask a question
curl -X POST http://localhost:8000/consultar \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Qué es el ratio de capital CET1?"}'

# Search documents
curl -X POST http://localhost:8000/buscar \
  -H "Content-Type: application/json" \
  -d '{"query": "requisitos de capital Basilea III", "n_results": 5}'

# Add a document
curl -X POST http://localhost:8000/documentos/agregar \
  -H "Content-Type: application/json" \
  -d '{"texto": "...", "fuente": "EBA/GL/2017/16", "tipo": "guideline"}'
```

### Enable PostgreSQL logging (optional)

```bash
createdb regllm
python -c "from src.db import create_tables; import asyncio; asyncio.run(create_tables())"
# The API will now log every query to the query_logs table
```

---

## Configuration

All settings live in `config.py`. Key sections:

```python
# Model
MODEL_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "use_4bit": True,      # halves VRAM
    "lora_r": 64,
    "lora_alpha": 128,
}

# Training
TRAINING_CONFIG = {
    "num_epochs": 3,
    "batch_size": 2,
    "learning_rate": 2e-4,
    "gradient_accumulation_steps": 4,
}

# RAG
RAG_CONFIG = {
    "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "n_results": 5,
}

# GRPO rewards
GRPO_CONFIG = {
    "group_size": 4,
    "learning_rate": 1e-5,
    "reward_weights": {"keyword": 0.5, "source": 0.3, "format": 0.2},
}
```

---

## Project Structure

```
regllm/
├── api/
│   └── main.py                    # FastAPI REST server
├── data/
│   ├── raw/                       # Downloaded PDFs and scraped text
│   ├── processed/                 # Structured bank data per institution/year
│   ├── finetuning/                # Training JSONL datasets
│   │   ├── banking_qa_dataset.jsonl
│   │   ├── sql_methodology_comparison_dataset.jsonl
│   │   └── generated_qa.jsonl     # Output of QA generator
│   ├── exports/                   # CSV exports for inspection
│   └── preferences/               # DPO preference pairs
├── models/
│   ├── finetuned/                 # SFT checkpoints
│   └── grpo/                      # GRPO-trained models
├── scripts/
│   ├── generate_qa_from_docs.py   # QA pair generator from documents
│   ├── generate_example_data.py   # Synthetic banking data
│   ├── download_financial_reports.py
│   ├── train_combined.py          # Main SFT training script
│   ├── run_grpo.py                # GRPO training
│   ├── run_full_pipeline.py       # End-to-end orchestration
│   ├── use_banking_model.py       # Inference CLI
│   ├── dataset_utils.py           # Validate/inspect datasets
│   ├── export_datasets.py         # Export to CSV
│   └── setup_banking.sh           # Initial setup
├── src/
│   ├── rag_system.py              # RAG with ChromaDB + BM25
│   ├── verification.py            # Hallucination detection
│   ├── citation_tree.py           # Regulation citation hierarchy
│   ├── llm_judge.py               # Automated evaluation
│   ├── db.py                      # Async PostgreSQL layer
│   ├── scraper.py                 # Web scraping
│   └── rlhf/
│       ├── grpo_trainer.py        # GRPO training loop
│       ├── grpo_rewards.py        # Reward functions
│       └── dpo_trainer.py         # DPO training
├── tests/                         # pytest test suite
├── examples/                      # Config examples (LLaMA-Factory, Axolotl)
├── config.py                      # Central configuration
├── app_gradio.py                  # Gradio web UI
├── cli.py                         # Command-line interface
├── requirements.txt               # Core dependencies
└── requirements-banking.txt       # Banking data extras
```

---

## Hardware Requirements

| Model | Inference VRAM (4-bit) | Training VRAM (LoRA 4-bit) |
|---|---|---|
| Qwen2.5-7B | ~6 GB | ~12 GB |
| Qwen2.5-3B | ~3 GB | ~8 GB |
| Phi-3-Mini (3.8B) | ~3 GB | ~8 GB |
| Gemma-2B | ~2 GB | ~6 GB |
| Qwen-1.8B | ~2 GB | ~5 GB |

**No GPU?** Use `--backend ollama` for QA generation and API inference (CPU, ~1–5 tok/s).

---

## Tests

```bash
pytest                          # Run all tests
pytest -m "not requires_gpu"   # Skip GPU tests
pytest tests/test_citation_tree.py
pytest tests/test_llm_judge.py
```

---

## Troubleshooting

**CUDA out of memory during training**
- Reduce `--batch-size` to 1
- Use a smaller model (`Qwen/Qwen2.5-3B-Instruct`)
- Make sure `use_4bit: True` in `config.py`

**PDF text extraction empty**
- Install `poppler-utils`: `sudo apt-get install poppler-utils`
- The script falls back to `pdfplumber` automatically

**Ollama connection refused**
- Start Ollama: `ollama serve`
- Check the URL with `--ollama-url http://localhost:11434`

**Model download fails**
- Authenticate: `huggingface-cli login`
- Accept the model license on the HuggingFace website

**Port already in use**
- Change port: `uvicorn api.main:app --port 8080`

---

## Disclaimer

RegLLM is for research and educational purposes. Do not use it as the sole basis for regulatory compliance decisions. Always consult official documents and qualified compliance professionals.

# RegLLM - Implementation Summary

**RegLLM** (Regulatory LLM) is a fine-tuned language model system specialized in Spanish banking regulation compliance. It integrates RAG (Retrieval-Augmented Generation), agent-based reasoning, RLHF training, and multiple UI interfaces.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Core Components](#2-core-components)
3. [RAG System](#3-rag-system)
4. [Training Pipeline](#4-training-pipeline)
5. [Agent Framework](#5-agent-framework)
6. [RLHF Implementation](#6-rlhf-implementation)
7. [User Interfaces](#7-user-interfaces)
8. [Data Processing](#8-data-processing)
9. [Configuration](#9-configuration)
10. [Testing](#10-testing)

---

## 1. Project Structure

```
regllm/
├── api/                      # FastAPI REST API
│   └── main.py               # API endpoints
├── data/
│   ├── raw/                  # Scraped regulation documents
│   ├── processed/            # Cleaned training data
│   ├── train/ & val/         # Train/validation splits
│   ├── methodology/          # Methodology documents
│   ├── preferences/          # RLHF feedback data
│   └── uploads/              # User uploaded documents
├── src/
│   ├── agents/               # Agent framework
│   │   ├── agent_loop.py     # Agent orchestration
│   │   ├── tool_registry.py  # Tool management
│   │   ├── tool_executor.py  # Tool execution
│   │   └── tools/            # Tool implementations
│   ├── documents/            # Document handling
│   ├── preprocessing/        # Data processing
│   ├── rlhf/                 # RLHF components
│   ├── scraper/              # Web scraping
│   ├── tools/                # Dataset management
│   ├── training/             # Model fine-tuning
│   ├── ui/                   # User interfaces
│   ├── rag_system.py         # RAG implementation
│   └── verification.py       # Response verification
├── tests/                    # Unit tests
├── models/                   # Trained checkpoints
├── logs/                     # Training logs
├── vector_db/                # ChromaDB storage
├── cli.py                    # CLI interface
├── app_gradio.py             # Gradio web UI
├── config.py                 # Configuration
└── requirements.txt          # Dependencies
```

---

## 2. Core Components

### 2.1 Supported Models

| Model | Parameters | GPU Memory | Use Case |
|-------|------------|------------|----------|
| Qwen2.5-7B | 7B | ~6GB | Highest quality |
| Qwen2.5-3B | 3B | ~4GB | Good balance |
| Phi-3-mini | 3.8B | ~4GB | Optimized |
| StableLM-3B | 3B | ~4GB | Fast training |
| Phi-2 | 2.7B | ~3GB | Quick iteration |
| Gemma-2B | 2B | ~2GB | Lightweight |
| Qwen-1.8B | 1.8B | ~2GB | Fastest |

### 2.2 Key Technologies

- **Quantization**: 4-bit BitsAndBytes for memory efficiency
- **LoRA**: Parameter-efficient fine-tuning (rank=64, alpha=128)
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **Vector DB**: ChromaDB with persistent storage
- **Search**: Hybrid semantic + BM25 retrieval

---

## 3. RAG System

**Location**: `src/rag_system.py`

### 3.1 Architecture

```
Query → Embedding → [Semantic Search] ─┐
                                       ├─→ Hybrid Merge → Ranked Results
Query → Tokenize  → [BM25 Search]    ─┘
```

### 3.2 Key Class: `RegulatoryRAGSystem`

**Methods**:
- `procesar_documentos()` - Add documents with embeddings
- `buscar_contexto(query, n)` - Semantic search only
- `buscar_hibrida(query, n)` - Combined semantic + BM25
- `load_from_json(path)` - Load documents from JSON
- `get_stats()` - System statistics

### 3.3 Configuration

```python
RAG = {
    'embedding_model': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'vector_db_path': 'vector_db/chroma_db',
    'default_n_results': 5,
    'hybrid_search_weight': 0.7,  # 70% semantic, 30% keyword
    'max_chunk_size': 1500,
}
```

### 3.4 Vector Database

- **Storage**: `vector_db/chroma_db/`
- **Collection**: `regulacion_bancaria`
- **Metadata**: documento, articulo, source, documento_id

---

## 4. Training Pipeline

### 4.1 Model Setup (`src/training/model_setup.py`)

```
Load Base Model
    ↓
4-bit Quantization (BitsAndBytes)
    ↓
Load Tokenizer
    ↓
Apply LoRA Configuration
    ↓
Ready for Training
```

### 4.2 Training Script (`src/training/train.py`)

**LoRA Configuration**:
```python
lora_config = {
    'r': 64,
    'lora_alpha': 128,
    'lora_dropout': 0.05,
    'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
}
```

**Training Modes**:

| Mode | Epochs | Batch Size | Learning Rate | Purpose |
|------|--------|------------|---------------|---------|
| Small subset | 10 | 2 | 3e-4 | Overfitting test |
| Full training | 3 | 4 | 2e-4 | Production |

### 4.3 Training Data Format

```json
{
  "messages": [
    {"role": "user", "content": "¿Qué es el capital CET1?"},
    {"role": "assistant", "content": "El capital CET1 (Common Equity Tier 1)..."}
  ]
}
```

### 4.4 Outputs

- Checkpoints: `models/finetuned/run_TIMESTAMP/`
- Loss plots: `logs/loss_plot_*.png`
- Training logs: `logs/training_*.log`

---

## 5. Agent Framework

**Location**: `src/agents/`

### 5.1 Architecture

```
Agent
  ├── ToolRegistry (tool discovery)
  ├── ToolExecutor (tool execution)
  └── LLM (reasoning)
```

### 5.2 Agent Types

1. **RegulationAgent** - General regulatory queries
2. **MethodologyConsistencyAgent** - Code vs methodology verification

### 5.3 Registered Tools (14 total)

**Methodology Tools**:
- `read_methodology` - Read .md/.txt files
- `parse_methodology_sections` - Extract document structure
- `extract_formulas` - Find mathematical formulas
- `extract_parameters` - Identify parameter definitions
- `compare_methodologies` - Compare documents

**Code Analysis Tools**:
- `read_code_file` - Read source with line ranges
- `analyze_code_structure` - Parse AST structure
- `extract_functions` - Find function definitions
- `extract_calculations` - Identify math operations
- `find_pattern_in_code` - Regex-based search

**Consistency Tools**:
- `check_formula_consistency` - Verify formulas
- `check_parameter_consistency` - Verify parameters
- `check_implementation_completeness` - Check components
- `generate_consistency_report` - Create reports

### 5.4 Agent Execution Flow

```
Agent.run(query)
    ↓
LLM thinks & calls tool
    ↓
ToolExecutor validates parameters
    ↓
Execute tool function
    ↓
Format result for LLM
    ↓
Repeat until final answer (max 10 steps)
```

---

## 6. RLHF Implementation

**Location**: `src/rlhf/`

### 6.1 Components

- `feedback_collector.py` - Stores user feedback
- `preference_dataset.py` - Creates DPO training pairs
- `dpo_trainer.py` - Direct Preference Optimization training

### 6.2 Feedback Flow

```
User Feedback (Gradio UI)
    ↓ (thumbs up/down)
feedback_collector.record_feedback()
    ↓
data/preferences/feedback.jsonl
    ↓ (accumulate 100+ samples)
PreferenceDataset.load_from_feedback()
    ↓
RegulationDPOTrainer.train()
    ↓
Improved Model
```

### 6.3 DPO Configuration

```python
DPO = {
    'beta': 0.1,              # KL penalty coefficient
    'learning_rate': 5e-7,    # Very low for stability
    'batch_size': 4,
    'max_steps': 500,
}
```

### 6.4 Data Formats

**Feedback Entry** (`feedback.jsonl`):
```json
{
  "query": "¿Qué es el PD?",
  "response": "La Probabilidad de Default...",
  "feedback": "positive",
  "timestamp": "2024-01-15T10:30:00"
}
```

**Preference Pair** (`dpo_pairs.json`):
```json
{
  "prompt": "¿Qué es el LGD?",
  "chosen": "El LGD (Loss Given Default)...",
  "rejected": "No tengo información sobre eso."
}
```

---

## 7. User Interfaces

### 7.1 CLI (`cli.py`)

**Interactive Commands**:
- `/salir` or `/exit` - Quit
- `/fuentes N` - Change number of sources
- `/hibrida on|off` - Toggle hybrid search
- `/stats` - Show statistics
- `/help` - Show help

### 7.2 Gradio Web UI (`app_gradio.py`)

**Tabs**:
- Query - RAG search with verification
- Documents - Upload and manage files
- Compare - Methodology comparison
- Statistics - System dashboard
- Feedback - RLHF feedback collection

**Features**:
- Response verification display
- Source citations with confidence
- Document upload (PDF, DOCX, TXT, MD)
- Feedback collection (thumbs up/down)

### 7.3 FastAPI REST API (`api/main.py`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/consultar` | Query documents |
| POST | `/documentos` | Add documents |
| GET | `/documentos` | List documents |
| POST | `/scrape` | Scrape URLs |
| GET | `/health` | Health check |
| GET | `/stats` | Statistics |
| GET | `/docs` | Swagger UI |

### 7.4 Telegram Bot (`src/ui/telegram_bot.py`)

- Chat interface via Telegram
- Rate limiting (10 msg/min per user)
- Configured via `config.py` TELEGRAM section

### 7.5 Legacy Chat Interface (`src/ui/chat_interface.py`)

- Loads finetuned model with LoRA
- CLI and web modes available

---

## 8. Data Processing

### 8.1 Scraping Pipeline

**Location**: `src/scraper/regulation_scraper.py`

```
regurl.txt (URLs)
    ↓
RegulationScraper
    ├─ Fetch URL (rate limited)
    ├─ Detect PDF vs HTML
    ├─ Extract text (PyPDF2 / BeautifulSoup)
    └─ Extract metadata
    ↓
data/raw/regulation_data_TIMESTAMP.json
```

**Sources**:
- Bank of Spain
- European Central Bank (ECB)
- BOE (Boletín Oficial del Estado)
- CNMV
- Basel Committee

### 8.2 Processing Pipeline

**Location**: `src/preprocessing/data_processor.py`

```
data/raw/*.json
    ↓
DataProcessor.load_raw_data()
    ↓
Clean text:
    ├─ Remove whitespace
    ├─ Normalize characters
    └─ Filter by language
    ↓
Chunk documents:
    ├─ 1000-word chunks
    ├─ 200-word overlap
    └─ Min 50 words
    ↓
Create train/val split (85/15)
    ↓
Output:
    ├─ data/processed/train_data.json
    └─ data/processed/val_data.json
```

### 8.3 Methodology Comparator

**Location**: `src/preprocessing/methodology_comparator.py`

Creates comparison Q&A pairs for:
- IRB Foundation
- IRB Advanced
- Standardized Approach
- Basel Evolution

---

## 9. Configuration

**Location**: `config.py`

### 9.1 Key Sections

```python
# Directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'

# Scraping
SCRAPING = {
    'request_delay': 2,
    'timeout': 30,
}

# Data Processing
DATA_PROCESSING = {
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'val_ratio': 0.15,
    'max_seq_length': 512,
}

# Model
MODEL = {
    'base_model': 'qwen2.5-7b',
    'use_4bit': True,
    'lora': {...}
}

# Training
TRAINING = {
    'small_subset': {'epochs': 10, 'batch_size': 2, 'lr': 3e-4},
    'full_training': {'epochs': 3, 'batch_size': 4, 'lr': 2e-4},
}

# Inference
INFERENCE = {
    'max_new_tokens': 300,
    'temperature': 0.7,
    'top_p': 0.95,
}

# RAG
RAG = {
    'embedding_model': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'hybrid_search_weight': 0.7,
}

# Agent
AGENT = {
    'max_steps': 10,
    'llm': {'model': 'gpt-4', 'temperature': 0.7},
}
```

---

## 10. Testing

**Location**: `tests/`

### 10.1 Test Files

- `conftest.py` - Fixtures and sample data
- `test_agent_loop.py` - Agent orchestration
- `test_tool_registry.py` - Tool management
- `test_tool_executor.py` - Tool execution
- `test_methodology_tools.py` - Methodology tools
- `test_code_analysis_tools.py` - Code analysis
- `test_consistency_tools.py` - Consistency checks
- `test_integration.py` - End-to-end tests

### 10.2 Running Tests

```bash
# All tests
pytest tests/

# Specific file
pytest tests/test_agent_loop.py -v

# By pattern
pytest tests/ -k "consistency"

# With coverage
pytest tests/ --cov=src
```

---

## 11. Verification System

**Location**: `src/verification.py`

### 11.1 Verification Checks

| Check | Weight | Description |
|-------|--------|-------------|
| Citations | 30% | Validates citations exist in sources |
| Coherence | 40% | Semantic similarity question↔response |
| Hallucination | 20% | Detects unsupported claims |
| Language | 10% | Ensures Spanish language |

### 11.2 Output

```json
{
  "confidence_score": 0.85,
  "citation_valid": true,
  "coherence_score": 0.9,
  "hallucination_detected": false,
  "language_ok": true,
  "warnings": []
}
```

---

## 12. Dependencies

### Core ML
- `torch>=2.0.0`
- `transformers>=4.40.0`
- `peft>=0.10.0`
- `bitsandbytes>=0.43.0`
- `trl>=0.7.0`

### RAG
- `sentence-transformers>=2.3.0`
- `chromadb>=0.4.22`
- `langchain>=0.1.0`
- `rank-bm25>=0.2.2`

### Web & APIs
- `fastapi>=0.109.0`
- `gradio>=4.0.0`
- `uvicorn>=0.27.0`

### Data Processing
- `beautifulsoup4>=4.12.0`
- `PyPDF2>=3.0.0`
- `pandas>=2.0.0`

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interfaces                          │
├──────────┬──────────┬──────────┬──────────┬────────────────────┤
│   CLI    │  Gradio  │ FastAPI  │ Telegram │   Chat Interface   │
│ cli.py   │app_gradio│ api/main │telegram_ │ chat_interface.py  │
│          │   .py    │   .py    │  bot.py  │                    │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴────────┬───────────┘
     │          │          │          │              │
     └──────────┴──────────┴──────────┴──────────────┘
                           │
     ┌─────────────────────┼─────────────────────┐
     │                     │                     │
     ▼                     ▼                     ▼
┌─────────┐         ┌───────────┐         ┌──────────┐
│   RAG   │         │  Agents   │         │ Training │
│ System  │         │ Framework │         │ Pipeline │
├─────────┤         ├───────────┤         ├──────────┤
│ChromaDB │         │ Tools (14)│         │  LoRA    │
│ BM25    │         │ Executor  │         │   DPO    │
│Embedding│         │ Registry  │         │ Feedback │
└────┬────┘         └─────┬─────┘         └────┬─────┘
     │                    │                    │
     └────────────────────┼────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    Verification       │
              │ (Citations/Coherence) │
              └───────────────────────┘
```

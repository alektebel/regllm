# RegLLM - Quickstart Guide

Quick reference for all commands to train models, process data, and access interfaces.

---

## Installation

```bash
# Clone and setup
cd /home/diego/Development/regllm
pip install -r requirements.txt
```

---

## 1. Data Collection & Processing

### Scrape Regulatory Documents

```bash
# Scrape from URLs in regurl.txt
python src/scraper/regulation_scraper.py

# Output: data/raw/regulation_data_TIMESTAMP.json
```

### Process Raw Data

```bash
# Clean and chunk documents, create train/val splits
python src/preprocessing/data_processor.py

# Output:
#   data/processed/train_data.json
#   data/processed/val_data.json
```

### Dataset Management

```bash
# View dataset statistics
python src/tools/dataset_cli.py stats

# Launch dataset management UI
python src/tools/dataset_ui.py
```

---

## 2. Training

### Model Setup

```bash
# Download and prepare base model
python src/training/model_setup.py
```

### Training Commands

```bash
# Quick overfitting test (small subset)
python src/training/train.py --small-subset

# Full training with default settings
python src/training/train.py

# Custom training configuration
python src/training/train.py \
    --model phi-2 \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-4

# Available models:
#   qwen2.5-7b, qwen2.5-3b, phi-3-mini, stablelm-3b
#   phi-2, gemma-2b, qwen-1.8b
```

### DPO/RLHF Training

```bash
# Train with preference data (after collecting feedback)
python src/rlhf/dpo_trainer.py \
    --model-path models/finetuned/run_TIMESTAMP/final_model \
    --preferences data/preferences/dpo_pairs.json
```

### Training Outputs

```
models/finetuned/run_TIMESTAMP/
├── final_model/          # Final checkpoint
├── checkpoint-*/         # Intermediate checkpoints
└── training_args.json    # Configuration

logs/
├── loss_plot_*.png       # Loss curves
└── training_*.log        # Training logs
```

---

## 3. Vector Database

### Initialize/Load Documents

```bash
# Load documents into ChromaDB via CLI
python cli.py --load-json data/raw/regulation_data.json

# Or via Python:
python -c "
from src.rag_system import RegulatoryRAGSystem
rag = RegulatoryRAGSystem()
rag.load_from_json('data/raw/regulation_data.json')
print(rag.get_stats())
"
```

### Database Location

```
vector_db/chroma_db/    # ChromaDB persistent storage
```

### Reset Database

```bash
# Delete and reinitialize
rm -rf vector_db/chroma_db
python cli.py --load-json data/raw/regulation_data.json
```

---

## 4. User Interfaces

### CLI (Interactive Mode)

```bash
# Start interactive session
python cli.py --interactive

# Commands inside CLI:
#   /salir or /exit    - Quit
#   /fuentes N         - Set number of sources (default: 5)
#   /hibrida on|off    - Toggle hybrid search
#   /stats             - Show statistics
#   /help              - Show help
```

### CLI (Single Query)

```bash
# Single query mode
python cli.py "¿Qué es el capital CET1?"

# With custom data
python cli.py --load-json data/custom.json "¿Qué es el PD?"
```

### Gradio Web Interface

```bash
# Start web UI on localhost
python app_gradio.py --port 7860

# Create public shareable link
python app_gradio.py --share

# Access at: http://localhost:7860
```

### FastAPI REST API

```bash
# Start API server
python -m api.main --host 0.0.0.0 --port 8000

# Access:
#   Swagger UI: http://localhost:8000/docs
#   ReDoc:      http://localhost:8000/redoc
#   Health:     http://localhost:8000/health
```

### API Usage Examples

```bash
# Query endpoint
curl -X POST http://localhost:8000/consultar \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Qué es el CET1?", "n_fuentes": 5}'

# Get statistics
curl http://localhost:8000/stats

# Health check
curl http://localhost:8000/health
```

### Telegram Bot

```bash
# Configure in config.py:
# TELEGRAM = {'bot_token': 'YOUR_TOKEN', ...}

# Start bot
python src/ui/telegram_bot.py
```

### Chat Interface (Finetuned Model)

```bash
# CLI mode with finetuned model
python src/ui/chat_interface.py \
    --model-path models/finetuned/run_TIMESTAMP/final_model

# Web mode
python src/ui/chat_interface.py \
    --model-path models/finetuned/run_TIMESTAMP/final_model \
    --web
```

---

## 5. Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_agent_loop.py -v

# Run tests matching pattern
pytest tests/ -k "consistency"

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# View coverage
open htmlcov/index.html
```

---

## 6. Agents & Tools

### Run Consistency Check

```bash
# Demo consistency checking
python demo_consistency_check.py

# Or programmatically:
python -c "
from src.agents.agent_loop import MethodologyConsistencyAgent

agent = MethodologyConsistencyAgent()
result = agent.run(
    methodology_path='data/methodology/irb_foundation.md',
    code_path='src/calculations/pd_model.py'
)
print(result)
"
```

### Available Tools

```
Methodology Tools:
  - read_methodology
  - parse_methodology_sections
  - extract_formulas
  - extract_parameters
  - compare_methodologies

Code Analysis Tools:
  - read_code_file
  - analyze_code_structure
  - extract_functions
  - extract_calculations
  - find_pattern_in_code

Consistency Tools:
  - check_formula_consistency
  - check_parameter_consistency
  - check_implementation_completeness
  - generate_consistency_report
```

---

## 7. Quick Reference

### Environment Variables

```bash
# Optional: Set in .env or export
export CUDA_VISIBLE_DEVICES=0      # GPU selection
export TRANSFORMERS_CACHE=/path    # Model cache
export HF_TOKEN=your_token         # HuggingFace access
```

### Common Workflows

**Complete Pipeline (Data → Train → Deploy)**:
```bash
# 1. Scrape data
python src/scraper/regulation_scraper.py

# 2. Process data
python src/preprocessing/data_processor.py

# 3. Load into vector DB
python cli.py --load-json data/raw/regulation_data_*.json

# 4. Train model (optional)
python src/training/train.py --model phi-2 --epochs 3

# 5. Start interface
python app_gradio.py --port 7860
```

**Quick RAG Setup (No Training)**:
```bash
# 1. Load existing data
python cli.py --load-json data/raw/regulation_data.json

# 2. Start CLI
python cli.py --interactive
```

**Development/Testing**:
```bash
# Run tests
pytest tests/ -v

# Start with small subset
python src/training/train.py --small-subset
```

---

## 8. Configuration Quick Reference

Edit `config.py` to modify:

| Setting | Location | Default |
|---------|----------|---------|
| Base model | `MODEL['base_model']` | `qwen2.5-7b` |
| 4-bit quantization | `MODEL['use_4bit']` | `True` |
| LoRA rank | `MODEL['lora']['r']` | `64` |
| Training epochs | `TRAINING['full_training']['epochs']` | `3` |
| Batch size | `TRAINING['full_training']['batch_size']` | `4` |
| Learning rate | `TRAINING['full_training']['learning_rate']` | `2e-4` |
| Chunk size | `DATA_PROCESSING['chunk_size']` | `1000` |
| Hybrid search weight | `RAG['hybrid_search_weight']` | `0.7` |
| Max agent steps | `AGENT['max_steps']` | `10` |

---

## 9. Troubleshooting

### CUDA Out of Memory

```bash
# Use smaller model
python src/training/train.py --model phi-2

# Reduce batch size
python src/training/train.py --batch-size 2

# Enable gradient checkpointing (in config.py)
TRAINING['gradient_checkpointing'] = True
```

### ChromaDB Issues

```bash
# Reset database
rm -rf vector_db/chroma_db
python cli.py --load-json data/raw/regulation_data.json
```

### Model Loading Errors

```bash
# Clear cache and redownload
rm -rf ~/.cache/huggingface/hub/models--*
python src/training/model_setup.py
```

### Import Errors

```bash
# Ensure you're in project root
cd /home/diego/Development/regllm

# Install dependencies
pip install -r requirements.txt
```

---

## 10. File Locations Summary

| Component | Path |
|-----------|------|
| Raw data | `data/raw/` |
| Processed data | `data/processed/` |
| Methodology docs | `data/methodology/` |
| Feedback data | `data/preferences/` |
| Trained models | `models/finetuned/` |
| Vector database | `vector_db/chroma_db/` |
| Training logs | `logs/` |
| Tests | `tests/` |
| Configuration | `config.py` |

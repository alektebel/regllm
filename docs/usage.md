# RegLLM Usage Guide

## Quick Start (3 Commands)

```bash
# 1. Launch the chat interface
./launch_ui.sh

# 2. Open browser - Go to: http://localhost:7860

# 3. Ask questions!
# Example: "¿Qué es la probabilidad de default (PD)?"
```

## Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- GPU with 4GB+ VRAM (optional but recommended)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Authenticate with Hugging Face for gated models
huggingface-cli login
```

## Launching the UI

```bash
# Web interface (recommended)
./launch_ui.sh

# CLI interface
python src/ui/chat_interface.py \
    --model-path models/finetuned/run_20260118_220204/final_model \
    --interface cli

# With RAG (retrieval augmented generation)
python app_gradio.py
```

## Training

### Quick Training Test
```bash
python src/training/train.py --small-subset --epochs 10
```

### Full Training
```bash
python src/training/train.py --epochs 3 --batch-size 4
```

## Example Questions

### Spanish Questions
```
¿Qué es la probabilidad de default (PD)?
¿Cómo se calcula el LGD para carteras retail?
¿Qué regulación aplica al método IRB?
Explica el factor de apoyo a las PYME
¿Cuáles son los requisitos de capital para riesgo de crédito?
```

### English Questions
```
What is the probability of default?
How is LGD calculated for retail portfolios?
What regulation applies to the IRB method?
Explain the SME supporting factor
```

## Dataset Management

Tools are available in `src/tools/`:

```bash
# Analyze dataset quality
python src/tools/analyze_quality.py

# Clean and regenerate dataset
python src/tools/clean_dataset.py

# CLI for dataset operations
python src/tools/dataset_cli.py stats
python src/tools/dataset_cli.py search "PD"
python src/tools/dataset_cli.py add --question "..." --answer "..."

# Web UI for dataset management
python src/tools/dataset_ui.py

# Process local PDFs
python src/tools/process_pdfs.py
```

## Troubleshooting

### UI Won't Start
```bash
# Check if model exists
ls -lah models/finetuned/

# Try different port
python src/ui/chat_interface.py --port 8080
```

### Out of Memory
```bash
# Use CPU (slower)
export CUDA_VISIBLE_DEVICES=""
./launch_ui.sh

# Or reduce batch size during training
python src/training/train.py --batch-size 2
```

### Model Quality Low
1. Collect more training data
2. Train for more epochs
3. Use the full dataset (not small subset)

## File Locations

```
regllm/
├── models/finetuned/          # Trained models
├── data/processed/            # Training data
├── data/raw/                  # Raw scraped data
├── logs/                      # Training logs and plots
├── src/                       # Source code
│   ├── tools/                 # Dataset utilities
│   ├── training/              # Training scripts
│   ├── preprocessing/         # Data processing
│   └── ui/                    # User interfaces
└── vector_db/                 # RAG vector database
```

# Spanish Banking Regulation LLM (RegLLM)

A finetuned language model for Spanish banking regulation compliance, specialized in credit risk parameters and regulatory requirements. Now with **RAG (Retrieval-Augmented Generation)** for more accurate, source-grounded responses.

## Overview

This project finetunes a lightweight 7B parameter model on Spanish banking regulation documents to provide accurate, source-cited responses to compliance questions. It uses a RAG system with ChromaDB for semantic search and hybrid retrieval.

**Key Features:**
- Trained on official documents from Bank of Spain, ECB, BOE, CNMV, and Basel Committee
- Specialized in credit risk parameters (PD, LGD, EAD, IRB methods)
- **RAG System** with ChromaDB for accurate source retrieval
- **Hybrid Search** combining semantic embeddings with BM25 keywords
- **Response Verification** with citation checking and confidence scoring
- **Enhanced Scraper** with LinkedIn and JavaScript-heavy site support
- Always cites sources and admits uncertainty when appropriate
- REST API (FastAPI) for integration
- Interactive web and CLI interfaces

## Project Structure

```
regllm/
├── api/                  # FastAPI REST API
│   └── main.py          # API endpoints
├── data/
│   ├── raw/             # Scraped regulation documents
│   ├── processed/       # Cleaned and formatted data
│   ├── train/           # Training set
│   └── val/             # Validation set
├── src/
│   ├── scraper/         # Web scraping tools (incl. LinkedIn)
│   ├── preprocessing/   # Data cleaning pipeline
│   ├── training/        # Model training scripts
│   ├── ui/              # User interfaces
│   ├── rag_system.py    # RAG with ChromaDB
│   └── verification.py  # Response verification
├── vector_db/           # ChromaDB vector database
├── models/              # Finetuned models
├── logs/                # Training logs and plots
├── cli.py               # Interactive CLI
├── app_gradio.py        # Enhanced Gradio Web UI
├── regurl.txt           # URLs for regulation sources
└── requirements.txt     # Python dependencies
```

## Installation

### 1. Clone and Setup

```bash
cd regllm
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Authenticate with Hugging Face (if needed)

Some models require accepting license agreements on Hugging Face:

```bash
pip install huggingface_hub
huggingface-cli login
```

Visit the model page (e.g., https://huggingface.co/microsoft/phi-2) and accept the license if prompted.

## Usage

### Step 1: Scrape Regulation Documents

Collect banking regulation documents from official sources:

```bash
python src/scraper/regulation_scraper.py
```

This will:
- Read URLs from `regurl.txt`
- Scrape HTML pages and PDFs
- Extract regulatory text
- Save to `data/raw/regulation_data_*.json`

**Note:** Scraping may take time. Be respectful of rate limits.

### Step 2: Preprocess Data

Clean and prepare data for training:

```bash
python src/preprocessing/data_processor.py
```

This creates:
- `data/processed/train_data.json` - Full training set
- `data/processed/val_data.json` - Validation set
- `data/processed/train_data_small.json` - Small subset for overfitting test

### Step 2.5: Manage and Clean Dataset (Optional)

Review, edit, and improve your training data using the dataset management tools:

#### Web UI (Interactive)

```bash
./launch_dataset_manager.sh
# or
python dataset_manager_ui.py
```

Access at http://localhost:7861

Features:
- Browse and search all samples
- Add new Q&A pairs manually
- Edit existing samples
- Delete low-quality samples
- View statistics and distribution
- Automatic backups before changes

#### CLI (Programmatic)

```bash
# View statistics
python dataset_manager_cli.py stats

# Search for samples
python dataset_manager_cli.py search "PD" --field answer

# Add new sample
python dataset_manager_cli.py add \
  --question "¿Qué es el PD floor?" \
  --answer "El PD floor es..." \
  --source "EBA"

# Validate dataset
python dataset_manager_cli.py validate
```

**Use Cases:**
- Manual quality control and editing
- LLM-assisted dataset cleaning
- Batch processing with scripts
- Automated validation

See [DATASET_MANAGEMENT.md](DATASET_MANAGEMENT.md) for complete guide.

### Step 3: Test Model Setup

Verify the base model loads correctly:

```bash
python src/training/model_setup.py
```

Available models:
- `phi-2` (2.7B params, recommended)
- `phi-3-mini` (3.8B params)
- `stablelm-3b` (3B params)
- `gemma-2b` (2B params)
- `qwen-1.8b` (1.8B params)

### Step 4: Overfit on Small Subset

First, verify the training pipeline works by overfitting on a small subset:

```bash
python src/training/train.py \
    --model phi-2 \
    --small-subset \
    --epochs 10 \
    --batch-size 2 \
    --lr 3e-4
```

**Expected behavior:** Training loss should decrease significantly, showing the model can learn from the data.

### Step 5: Train on Full Dataset

Once overfitting succeeds, train on the full dataset:

```bash
python src/training/train.py \
    --model phi-2 \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-4
```

**Training parameters:**
- Uses LoRA (Low-Rank Adaptation) for efficient finetuning
- 4-bit quantization to reduce memory usage
- Gradient accumulation for larger effective batch sizes
- Automatic GPU utilization if available

**Monitoring:**
- Training progress logged to console
- Loss plots saved to `logs/training_plot_*.png`
- Model checkpoints saved to `models/finetuned/run_*/`

### Step 6: Launch the UI

#### Enhanced Gradio Interface (Recommended)

```bash
python app_gradio.py --port 7860
```

Then open http://localhost:7860 in your browser.

Features:
- Query regulatory documents with RAG
- View source citations and confidence scores
- Manage documents and view statistics
- Hybrid search (semantic + keywords)

To create a public link:
```bash
python app_gradio.py --share
```

#### Interactive CLI

```bash
python cli.py --interactive
```

Or for a single query:
```bash
python cli.py "What is the minimum CET1 ratio?"
```

#### REST API (FastAPI)

```bash
python -m api.main --host 0.0.0.0 --port 8000
```

Then access the API:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

Example API call:
```bash
curl -X POST "http://localhost:8000/consultar" \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "Que es el capital CET1?", "n_fuentes": 5}'
```

#### Legacy UI (Original)

```bash
python src/ui/chat_interface.py \
    --model-path models/finetuned/run_XXXXXX/final_model \
    --interface web \
    --port 7861
```

### Step 7: Scrape LinkedIn and Other Sites (Optional)

Use the enhanced scraper for JavaScript-heavy sites:

```python
from src.scraper import EnhancedScraper

scraper = EnhancedScraper()

# Scrape LinkedIn articles
urls = [
    "https://www.linkedin.com/pulse/banking-regulation-article...",
]

results = scraper.scrape_multiple(urls)
scraper.save_results(results)
```

Or via the API:
```bash
curl -X POST "http://localhost:8000/scrape" \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://linkedin.com/..."], "include_linkedin": true}'
```

## Example Questions

Spanish:
- ¿Qué es la probabilidad de default (PD)?
- ¿Qué regulación se aplica al cálculo de capital para riesgo de crédito?
- Explica el método IRB para carteras retail
- ¿Cuáles son los requisitos para el cálculo de LGD en carteras corporativas?
- ¿Qué dice el Banco de España sobre provisiones IFRS 9?

English:
- What regulation applies to credit risk capital calculation?
- Explain the IRB method for retail portfolios
- What are the requirements for calculating LGD?

## Training Tips

### GPU Acceleration

Training is much faster with a GPU. If you don't have one:
- Use Google Colab (free GPU): [colab.research.google.com](https://colab.research.google.com)
- Use cloud providers (AWS, GCP, Azure)
- Rent GPU time from vast.ai or similar services

### Memory Optimization

If you encounter out-of-memory errors:
- Reduce batch size: `--batch-size 2` or `--batch-size 1`
- Increase gradient accumulation: Add `gradient_accumulation_steps=8` in train.py
- Use a smaller model: `--model qwen-1.8b`
- Reduce max sequence length in `RegulationDataset.__init__()`

### Improving Model Quality

1. **More Data**: Scrape additional regulatory sources
2. **Better QA Pairs**: Manually create high-quality examples
3. **Longer Training**: Increase epochs (but watch for overfitting)
4. **Hyperparameter Tuning**: Experiment with learning rate, LoRA rank, etc.
5. **Data Augmentation**: Create variations of existing questions

## Model Behavior

The model is trained to:
- ✅ Always cite sources (Bank of Spain, ECB, etc.)
- ✅ Admit uncertainty when it doesn't have information
- ✅ Provide specific regulatory references
- ✅ Focus on credit risk parameters and Spanish regulations
- ❌ Never invent information
- ❌ Never provide answers without source attribution

## Troubleshooting

### Scraper Issues
- **403/404 Errors**: Some URLs may be outdated. Update `regurl.txt`
- **Rate Limiting**: Increase delay in `RegulationScraper.__init__()` (self.delay)
- **PDF Extraction Fails**: Install poppler-utils: `apt-get install poppler-utils`

### Training Issues
- **Model download fails**: Accept license on Hugging Face and authenticate
- **CUDA out of memory**: Reduce batch size or use CPU (slower)
- **No training progress**: Check data files exist in `data/processed/`

### UI Issues
- **Model not loading**: Check model path is correct
- **Slow responses**: Normal on CPU; use GPU for faster inference
- **Port already in use**: Change port with `--port 8080`

## Evaluation

To evaluate model performance:

1. **Validation Loss**: Check `logs/training_plot_*.png`
   - Should decrease over epochs
   - Gap between train/val indicates overfitting

2. **Manual Testing**: Ask domain-specific questions
   - Does it cite sources?
   - Are answers accurate?
   - Does it admit uncertainty appropriately?

3. **Compare Before/After**: Test base model vs finetuned model

## Citation

If you use this project, please cite:
```
Spanish Banking Regulation LLM (RegLLM)
https://github.com/yourusername/regllm
```

## License

This project is for educational and research purposes. Ensure compliance with:
- Model licenses (Phi-2, etc.)
- Data usage terms from regulatory sources
- Applicable banking and data protection regulations

## Disclaimer

This model is a tool for research and educational purposes. It should NOT be used as the sole basis for regulatory compliance decisions. Always consult official regulatory documents and qualified compliance professionals.

## Contributing

Contributions welcome! Areas for improvement:
- Additional data sources
- Better QA pair generation
- Model optimization
- Evaluation metrics
- Multi-language support

## Contact

For questions or issues, please open a GitHub issue.

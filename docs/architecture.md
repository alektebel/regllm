# RegLLM Architecture

## Overview

RegLLM is a Spanish banking regulation chatbot that uses fine-tuned LLMs with optional RAG (Retrieval Augmented Generation).

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Gradio Web   │  │     CLI      │  │ Telegram Bot │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      RAG System (Optional)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Embeddings │  │  Vector DB   │  │   Retrieval  │         │
│  │  (Sentence   │  │  (ChromaDB)  │  │    Engine    │         │
│  │ Transformers)│  │              │  │              │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                         LLM Engine                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Base Model  │  │ LoRA Adapter │  │   4-bit      │         │
│  │  (Qwen2.5)   │  │  (Fine-tuned)│  │ Quantization │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
regllm/
├── src/
│   ├── training/           # Model training
│   │   ├── train.py        # Main training script
│   │   └── model_setup.py  # Model configuration
│   ├── preprocessing/      # Data processing
│   │   └── data_processor.py
│   ├── scraper/            # Web scraping
│   │   └── regulation_scraper.py
│   ├── rag/                # RAG system
│   │   └── rag_system.py
│   ├── ui/                 # User interfaces
│   │   ├── chat_interface.py
│   │   └── telegram_bot.py
│   └── tools/              # Utilities
│       ├── dataset_cli.py
│       ├── dataset_ui.py
│       ├── analyze_quality.py
│       ├── clean_dataset.py
│       └── process_pdfs.py
├── data/
│   ├── raw/                # Scraped documents
│   ├── processed/          # Training data
│   ├── pdf/                # Local PDFs
│   └── backups/            # Dataset backups
├── models/
│   └── finetuned/          # Trained models
├── vector_db/              # ChromaDB storage
├── logs/                   # Training logs
├── docs/                   # Documentation
├── config.py               # Configuration
├── app_gradio.py           # Main Gradio app
└── cli.py                  # Main CLI
```

## Training Pipeline

1. **Data Collection**: Scrape or load regulatory documents
2. **Preprocessing**: Extract Q&A pairs, clean text
3. **Training**: Fine-tune with LoRA on 4-bit quantized model
4. **Evaluation**: Test on validation set

## Model Configuration

- **Base Model**: Qwen2.5-7B-Instruct (configurable)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit (BitsAndBytes)
- **Memory**: ~4GB GPU VRAM

## RAG Integration

When enabled, the system:
1. Embeds user query using Sentence Transformers
2. Retrieves relevant document chunks from ChromaDB
3. Augments prompt with retrieved context
4. Generates response with grounded information

## Key Technologies

- **Transformers**: Model loading and inference
- **PEFT**: Parameter-efficient fine-tuning (LoRA)
- **BitsAndBytes**: 4-bit quantization
- **Gradio**: Web interface
- **ChromaDB**: Vector database for RAG
- **Sentence Transformers**: Document embeddings

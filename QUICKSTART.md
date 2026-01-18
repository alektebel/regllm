# Quick Start Guide

Get started with RegLLM in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- GPU with 4GB+ VRAM (optional but recommended)
- Internet connection for downloading models

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Authenticate with Hugging Face
huggingface-cli login
# Get your token from https://huggingface.co/settings/tokens
```

## Option 1: Automated Pipeline (Recommended)

Run the complete pipeline with a single command:

```bash
# Run everything: scrape, preprocess, and train
python run_pipeline.py --all
```

Or run steps individually:

```bash
# Just scraping and preprocessing
python run_pipeline.py --scrape --preprocess

# Just overfitting test
python run_pipeline.py --train-small

# Just full training
python run_pipeline.py --train-full
```

## Option 2: Manual Step-by-Step

### Step 1: Scrape Data (Optional - can skip if you have data)

```bash
python src/scraper/regulation_scraper.py
```

**Note:** This may take 30-60 minutes depending on network speed.

### Step 2: Preprocess Data

```bash
python src/preprocessing/data_processor.py
```

### Step 3: Verify Model Setup

```bash
python src/training/model_setup.py
```

If you see a license error:
1. Visit https://huggingface.co/microsoft/phi-2
2. Click "Access repository"
3. Run `huggingface-cli login`

### Step 4: Overfitting Test (Important!)

```bash
python src/training/train.py --small-subset --epochs 10
```

This should complete in 10-30 minutes. Check `logs/training_plot_*.png` - the loss should go near zero.

### Step 5: Full Training

```bash
python src/training/train.py --epochs 3 --batch-size 4
```

This may take 1-3 hours depending on your hardware and dataset size.

### Step 6: Launch UI

```bash
# Find your model path
ls -la models/finetuned/

# Launch web interface (replace XXXXXX with your timestamp)
python src/ui/chat_interface.py \
    --model-path models/finetuned/run_XXXXXX/final_model \
    --interface web
```

Open http://localhost:7860 in your browser!

## Quick Test

Try these questions:
- "¿Qué es la probabilidad de default (PD)?"
- "¿Qué regulación se aplica al cálculo de capital para riesgo de crédito?"
- "Explica el método IRB para carteras retail"

The model should cite sources and provide regulatory information!

## Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
python src/training/train.py --batch-size 2
# Or use CPU (slower)
export CUDA_VISIBLE_DEVICES=""
```

### "Model not found" or "Permission denied"
```bash
# Login to Hugging Face
huggingface-cli login
# Accept model license at https://huggingface.co/microsoft/phi-2
```

### "No training data found"
```bash
# Run preprocessing first
python src/preprocessing/data_processor.py
# Check that files exist
ls -la data/processed/
```

## Performance Tips

### Speed up training:
- Use a GPU (10-50x faster than CPU)
- Try Google Colab for free GPU access
- Increase batch size if you have more VRAM

### Improve model quality:
- Collect more data (scrape additional sources)
- Train for more epochs (3-5)
- Create better QA pairs manually
- Increase LoRA rank in `config.py`

## Next Steps

1. **Evaluate**: Test the model with domain-specific questions
2. **Iterate**: If quality is low, collect more data or train longer
3. **Deploy**: Share the model with your team
4. **Monitor**: Track which questions work well and which don't

## Need Help?

- Check the main README.md for detailed documentation
- Review config.py for all configurable parameters
- Open an issue on GitHub

## Example Complete Workflow

```bash
# 1. Setup
pip install -r requirements.txt
huggingface-cli login

# 2. Get data (or skip if you already have it)
python run_pipeline.py --scrape --preprocess

# 3. Quick test (optional but recommended)
python run_pipeline.py --train-small

# 4. Full training
python run_pipeline.py --train-full --epochs 3

# 5. Launch UI
python src/ui/chat_interface.py \
    --model-path models/finetuned/run_*/final_model \
    --interface web

# Done! 🎉
```

That's it! You now have a working Spanish banking regulation chatbot.

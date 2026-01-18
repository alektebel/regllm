# RegLLM - Quick Usage Guide

## 🚀 Quick Start (3 Commands)

```bash
# 1. Launch the chat interface
./launch_ui.sh

# 2. Open browser
# Go to: http://localhost:7860

# 3. Ask questions!
# Example: "¿Qué es la probabilidad de default (PD)?"
```

That's it! Your Spanish banking regulation chatbot is running.

---

## 📋 Common Commands

### Test the Trained Model
```bash
# Web interface (recommended)
./launch_ui.sh

# CLI interface
python src/ui/chat_interface.py \
    --model-path models/finetuned/run_20260116_212356/final_model \
    --interface cli

# Public sharing (creates public URL)
python src/ui/chat_interface.py \
    --model-path models/finetuned/run_20260116_212356/final_model \
    --share
```

### Improve the Model

#### Option 1: Collect More Real Data
```bash
# Add URLs to regurl.txt, then:
python run_pipeline.py --scrape --preprocess --train-full --epochs 5
```

#### Option 2: Use Sample Data for Quick Testing
```bash
# Create sample data
python create_sample_data.py

# Train quickly
python train_quick.py --epochs 5 --batch-size 4
```

### View Training Results
```bash
# Check logs
ls -lah logs/

# View training plot
xdg-open logs/training_plot_*.png  # Linux
open logs/training_plot_*.png      # Mac
```

---

## 💬 Example Questions

### Spanish Questions
```
¿Qué es la probabilidad de default (PD)?
¿Cómo se calcula el LGD para carteras retail?
¿Qué regulación aplica al método IRB?
Explica el factor de apoyo a las PYME
¿Cuáles son los requisitos de capital para riesgo de crédito?
¿Qué dice el Banco de España sobre provisiones?
```

### English Questions
```
What is the probability of default?
How is LGD calculated for retail portfolios?
What regulation applies to the IRB method?
Explain the SME supporting factor
What are the capital requirements for credit risk?
```

### Technical Questions
```
¿Cuál es la diferencia entre IRB básico y avanzado?
¿Qué parámetros se necesitan para calcular RWA?
¿Cómo se segmentan las carteras minoristas?
What is the through-the-cycle PD?
```

---

## 🛠️ Troubleshooting

### UI Won't Start
```bash
# Check if model exists
ls -lah models/finetuned/run_20260116_212356/final_model/

# Try different port
python src/ui/chat_interface.py \
    --model-path models/finetuned/run_20260116_212356/final_model \
    --port 8080
```

### Out of Memory
```bash
# The model uses 4-bit quantization and should work with 4GB GPU
# If still issues, try CPU (slower):
export CUDA_VISIBLE_DEVICES=""
./launch_ui.sh
```

### Model Quality Low
The current model is trained on only 19 documents. To improve:

1. **Collect more data**:
   ```bash
   # Edit regurl.txt to add more URLs
   python run_pipeline.py --scrape --preprocess
   ```

2. **Train longer**:
   ```bash
   python train_quick.py --epochs 10 --batch-size 4
   ```

3. **Use full dataset** (not small subset):
   ```bash
   # Edit train_quick.py and remove --small-subset flag
   python train_quick.py --epochs 5
   ```

---

## 📊 Check Training Progress

### View Current Models
```bash
# List all trained models
ls -lah models/finetuned/

# Each run_* directory contains:
#   - checkpoint-* : Intermediate checkpoints
#   - final_model  : Best model (use this!)
```

### View Training Metrics
```bash
# Training plots
ls -lah logs/training_plot_*.png

# Training logs
cat logs/regllm.log
```

### Model Info
```bash
# Best model location
echo "models/finetuned/run_20260116_212356/final_model"

# Model size
du -sh models/finetuned/run_20260116_212356/final_model/

# Training data used
wc -l data/processed/train_data.json
```

---

## 🔄 Retrain from Scratch

```bash
# Complete pipeline (scrape + preprocess + train)
python run_pipeline.py --all

# Just training (if you have data)
python train_quick.py --small-subset --epochs 10  # Quick test
python train_quick.py --epochs 5                  # Full training
```

---

## 📚 File Locations

```
regllm/
├── models/finetuned/run_20260116_212356/final_model/  ← YOUR MODEL
├── data/processed/train_data.json                     ← TRAINING DATA
├── logs/training_plot_*.png                           ← TRAINING PLOT
├── launch_ui.sh                                       ← UI LAUNCHER
└── README.md                                          ← FULL DOCS
```

---

## 🎯 Advanced Usage

### Custom Training
```bash
python train_quick.py \
    --epochs 10 \
    --batch-size 8 \
    --lr 1e-4 \
    --small-subset  # Remove for full dataset
```

### Different Model
Edit `config.py` and change:
```python
MODEL = {
    'base_model': 'phi-3-mini',  # or 'gemma-2b', 'qwen-1.8b'
}
```

### Export for Deployment
```bash
# Model is already in HuggingFace format
# Can be uploaded to HuggingFace Hub or deployed as-is

# To share:
# 1. Zip the model
cd models/finetuned/
tar -czf regllm_model.tar.gz run_20260116_212356/final_model/

# 2. Share regllm_model.tar.gz
```

---

## 💡 Tips

1. **Start Simple**: Test with current model first
2. **Iterate**: Collect more data → retrain → test
3. **Monitor**: Check training plots for overfitting
4. **Evaluate**: Try edge cases and unusual questions
5. **Document**: Note what works and what doesn't

---

## 🆘 Need Help?

1. **Setup Issues**: Check `verify_setup.py`
2. **Training Issues**: See `RESULTS.md`
3. **Usage Issues**: See `README.md`
4. **Architecture**: See `PROJECT_SUMMARY.md`
5. **Quick Start**: See `QUICKSTART.md`

---

## ⚡ One-Liners

```bash
# Launch UI
./launch_ui.sh

# Quick retrain
python train_quick.py --small-subset --epochs 5

# Check status
python verify_setup.py

# Collect new data
python run_pipeline.py --scrape --preprocess

# View results
xdg-open logs/training_plot_*.png
```

---

## 🎉 You're Ready!

Your Spanish banking regulation chatbot is trained and ready to use.

**Next Step**: Run `./launch_ui.sh` and start asking questions!

---

*RegLLM - Banking Regulation Language Model*
*Built with Transformers, PEFT, and Gradio*

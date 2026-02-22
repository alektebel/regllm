# Fine-Tuning Quickstart Guide

Complete guide to fine-tune a model on credit risk, SQL methodology review, and Spanish banking data.

---

## Overview

This project provides **3 training datasets** for building a credit risk expert LLM:

| Dataset | Examples | Purpose |
|---------|----------|---------|
| SQL Methodology Comparison | 20 | Review SQL code against EBA/IRB methodology |
| Banking Q&A | 82 | Spanish bank financial data (2022-2023) |
| Regulation Training | 61 | EBA guidelines and banking regulation |
| **Combined Total** | **163** | All datasets merged |

---

## 1. Quick Setup

```bash
# Install dependencies
cd /home/diego/Development/regllm
pip install -r requirements.txt

# Generate/refresh all datasets
python scripts/generate_example_data.py
python scripts/export_datasets.py
```

---

## 2. Dataset Locations

```
data/
├── finetuning/
│   ├── sql_methodology_comparison_dataset.jsonl  # SQL vs methodology (20)
│   └── banking_qa_dataset.jsonl                  # Banking Q&A (82)
├── processed/
│   └── train_data.json                           # Regulation (61)
└── exports/
    ├── combined_training_dataset.json            # All 163 examples
    ├── sql_methodology_comparison.csv            # For LibreOffice/Excel
    ├── banking_qa.csv
    └── spanish_banks_financials.csv
```

---

## 3. Dataset Formats

### SQL Methodology Comparison (JSONL)

Reviews SQL code for compliance with EBA/GL/2017/16 and CRR:

```json
{
  "messages": [
    {"role": "system", "content": "Eres un experto en riesgo de crédito que revisa código SQL..."},
    {"role": "user", "content": "Revisa este SQL que calcula la PD:\n```sql\nSELECT cliente_id, COUNT(CASE WHEN default_flag = 1...\n```"},
    {"role": "assistant", "content": "**ANÁLISIS DE CONFORMIDAD METODOLÓGICA**\n\n**Hallazgos:**\n1. ❌ **Ventana temporal insuficiente**..."}
  ]
}
```

**Topics covered:**
- PD estimation and calibration
- LGD workout and downturn
- EAD and CCF calculation
- RWA formulas (IRB)
- Default definition (CRR Art. 178)
- IFRS9 ECL provisioning
- Backtesting and validation
- Stress testing
- Margin of conservatism

### Banking Q&A (JSONL)

Spanish bank financial data questions:

```json
{
  "messages": [
    {"role": "user", "content": "¿Cuánto ganó Banco Santander en 2023?"},
    {"role": "assistant", "content": "Banco Santander obtuvo un beneficio neto de 11.076 millones EUR en 2023."}
  ]
}
```

**Banks covered:** Santander, CaixaBank, BBVA, Sabadell, Kutxabank
**Years:** 2022, 2023
**Metrics:** Total assets, net profit, capital ratio, ROE, NPL ratio

### Regulation Training (JSON)

EBA guidelines Q&A with source citations:

```json
{
  "messages": [
    {"role": "system", "content": "Eres un asistente experto en regulación bancaria..."},
    {"role": "user", "content": "¿Cuál es el objetivo principal de las directrices EBA/GL/2017/16?"},
    {"role": "assistant", "content": "Las directrices... [EBA/GL/2017/16, párrafo X]"}
  ],
  "metadata": {"source": "EBA", "title": "EBA Guidelines Q&A"}
}
```

---

## 4. Fine-Tuning Options

### Option A: Using This Project's Training Script

```bash
# Quick test (small subset)
python src/training/train.py --small-subset

# Full training with combined dataset
python src/training/train.py \
    --data-path data/exports/combined_training_dataset.json \
    --model qwen2.5-7b \
    --epochs 3 \
    --batch-size 2 \
    --lr 2e-4

# Available models:
#   qwen2.5-7b (default), qwen2.5-3b, phi-3-mini
#   stablelm-3b, phi-2, gemma-2b, qwen-1.8b
```

### Option B: Using LLaMA-Factory (Recommended for Production)

```bash
# 1. Clone LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .

# 2. Prepare combined dataset
cat ../data/finetuning/sql_methodology_comparison_dataset.jsonl \
    ../data/finetuning/banking_qa_dataset.jsonl \
    > data/credit_risk_combined.jsonl

# 3. Add to data/dataset_info.json:
cat >> data/dataset_info.json << 'EOF'
{
  "credit_risk_combined": {
    "file_name": "credit_risk_combined.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"}
  }
}
EOF

# 4. Train with LoRA
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --stage sft \
    --do_train \
    --dataset credit_risk_combined \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --output_dir ../models/credit-risk-llm \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --fp16
```

### Option C: Using Hugging Face Transformers Directly

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
tokenizer.pad_token = tokenizer.eos_token

# Add LoRA adapters
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load combined dataset
dataset = load_dataset("json", data_files={
    "train": [
        "data/finetuning/sql_methodology_comparison_dataset.jsonl",
        "data/finetuning/banking_qa_dataset.jsonl"
    ]
})

# Format function
def format_example(example):
    messages = example["messages"]
    text = ""
    for msg in messages:
        if msg["role"] == "system":
            text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "user":
            text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "assistant":
            text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
    return tokenizer(text, truncation=True, max_length=2048, padding="max_length")

tokenized_dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="models/finetuned/credit-risk",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=50,
    warmup_ratio=0.1,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)
trainer.train()

# Save
model.save_pretrained("models/finetuned/credit-risk/final_model")
tokenizer.save_pretrained("models/finetuned/credit-risk/final_model")
```

---

## 5. Viewing and Editing Data

### LibreOffice Calc (Spreadsheet View)

```bash
# Open all CSV exports
libreoffice --calc data/exports/*.csv

# Individual files:
libreoffice --calc data/exports/sql_methodology_comparison.csv
libreoffice --calc data/exports/banking_qa.csv
libreoffice --calc data/exports/spanish_banks_financials.csv
```

### VS Code (JSON/JSONL Editing)

```bash
code data/finetuning/sql_methodology_comparison_dataset.jsonl
```

### Command Line (jq)

```bash
# View first example
jq -s '.[0]' data/finetuning/sql_methodology_comparison_dataset.jsonl

# Count examples
jq -s 'length' data/finetuning/sql_methodology_comparison_dataset.jsonl

# Extract all user queries
jq -s '.[].messages[] | select(.role=="user") | .content' \
    data/finetuning/sql_methodology_comparison_dataset.jsonl
```

### Python/Pandas

```python
import pandas as pd
import json

# Load JSONL
with open("data/finetuning/sql_methodology_comparison_dataset.jsonl") as f:
    data = [json.loads(line) for line in f]

# Convert to DataFrame
df = pd.DataFrame([{
    "user": next(m["content"] for m in ex["messages"] if m["role"] == "user"),
    "assistant": next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
} for ex in data])

print(df.head())
```

---

## 6. Regenerating Datasets

```bash
# Regenerate banking Q&A (82 examples)
python scripts/generate_example_data.py

# Export to CSV/JSON formats
python scripts/export_datasets.py

# Validate dataset format
python scripts/dataset_utils.py validate

# View statistics
python scripts/dataset_utils.py stats
```

---

## 7. Adding More Examples

### Add SQL Methodology Examples

Edit `data/finetuning/sql_methodology_comparison_dataset.jsonl`:

```json
{"messages": [{"role": "system", "content": "Eres un experto en riesgo de crédito..."}, {"role": "user", "content": "Revisa este SQL:\n```sql\nYOUR_SQL_HERE\n```"}, {"role": "assistant", "content": "**ANÁLISIS DE CONFORMIDAD METODOLÓGICA**\n\n**Hallazgos:**\n1. ..."}]}
```

### Add Banking Q&A Examples

```bash
# Add new bank data to data/processed/<bank>/<bank>_<year>.json
# Then regenerate:
python scripts/generate_example_data.py
```

### Combine All Datasets

```bash
# Regenerate combined export
python scripts/export_datasets.py

# Or manually:
cat data/finetuning/sql_methodology_comparison_dataset.jsonl \
    data/finetuning/banking_qa_dataset.jsonl \
    > data/finetuning/combined_dataset.jsonl
```

---

## 8. Model Inference

### Using the Trained Model

```bash
# Interactive mode
python scripts/use_banking_model.py \
    --model-path models/finetuned/credit-risk/final_model \
    --mode interactive

# Single query
python scripts/use_banking_model.py \
    --model-path models/finetuned/credit-risk/final_model \
    --mode single \
    --query "Revisa este SQL: SELECT SUM(ead * pd * lgd) FROM cartera"
```

### Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base + LoRA
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="auto")
model = PeftModel.from_pretrained(base_model, "models/finetuned/credit-risk/final_model")
tokenizer = AutoTokenizer.from_pretrained("models/finetuned/credit-risk/final_model")

# Generate
prompt = """<|im_start|>system
Eres un experto en riesgo de crédito que revisa código SQL.<|im_end|>
<|im_start|>user
Revisa este SQL: SELECT AVG(1 - recuperacion/exposicion) as lgd FROM defaults<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 9. Hardware Requirements

| Model | VRAM (4-bit) | VRAM (FP16) | Training Time* |
|-------|--------------|-------------|----------------|
| Qwen2.5-7B | 8-12 GB | 16-24 GB | ~30 min |
| Qwen2.5-3B | 4-6 GB | 8-12 GB | ~15 min |
| Phi-3-Mini | 4-6 GB | 8-12 GB | ~15 min |
| Phi-2 | 3-4 GB | 6-8 GB | ~10 min |

*For 163 examples, 3 epochs, batch size 2, on RTX 3090/4090

---

## 10. Troubleshooting

### CUDA Out of Memory

```bash
# Use smaller model
python src/training/train.py --model phi-2

# Reduce batch size
python src/training/train.py --batch-size 1

# Enable gradient checkpointing in config.py
```

### Dataset Format Errors

```bash
# Validate JSONL format
python -c "
import json
with open('data/finetuning/sql_methodology_comparison_dataset.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except:
            print(f'Error on line {i}')
"
```

### Missing Dependencies

```bash
pip install torch transformers peft accelerate bitsandbytes datasets
```

---

## 11. File Reference

| File | Description |
|------|-------------|
| `data/finetuning/sql_methodology_comparison_dataset.jsonl` | SQL review training data |
| `data/finetuning/banking_qa_dataset.jsonl` | Banking Q&A training data |
| `data/processed/train_data.json` | Regulation training data |
| `data/exports/combined_training_dataset.json` | All datasets combined |
| `scripts/export_datasets.py` | Export to CSV/JSON |
| `scripts/generate_example_data.py` | Generate banking examples |
| `src/training/train.py` | Training script |
| `config.py` | Model and training configuration |

---

## 12. Next Steps

1. **Expand SQL methodology dataset** - Add more examples covering edge cases
2. **Download real bank PDFs** - Update URLs in `data/banks_urls.json`
3. **Add validation set** - Split 20% for evaluation
4. **Implement evaluation metrics** - Accuracy on held-out test set
5. **Try DPO/RLHF** - Collect preference data for alignment

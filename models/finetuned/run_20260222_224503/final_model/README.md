---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
pipeline_tag: text-generation
language:
- es
- en
tags:
- lora
- transformers
- banking
- regulation
- credit-risk
- finance
- spanish
license: apache-2.0
---

# RegLLM — Banking & Credit Risk Expert (LoRA adapter)

A LoRA adapter fine-tuned on top of [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) for banking regulation and credit risk expertise.

## What it does

This model specialises in:
- **Spanish banking regulation**: EBA Guidelines, CRR/CRD, Basel accords
- **Credit risk methodology**: PD/LGD/EAD estimation, IRB models, IFRS 9, stress testing
- **SQL methodology review**: validates credit risk SQL code against regulatory standards
- **Spanish bank financials**: Santander, BBVA, CaixaBank, Sabadell, Kutxabank (2022–2023)

## Training Details

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Method | LoRA (SFT) |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Training examples | ~163 (SQL + Banking Q&A + Regulation) |
| Adapter size | ~309MB |
| Run | 2026-02-22 |

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = "Qwen/Qwen2.5-7B-Instruct"
adapter = "kabesaml/regllm-qwen25-7b-banking-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter)

messages = [
    {"role": "system", "content": "Eres un asistente experto en regulación bancaria y el sector bancario español. Responde con datos precisos y cita la normativa cuando sea posible."},
    {"role": "user", "content": "¿Qué es la tasa de impago bajo el estándar CRR Art. 178?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## System Prompt

```
Eres un asistente experto en regulación bancaria y el sector bancario español.
Responde con datos precisos y cita la normativa cuando sea posible.
```

## Limitations

- Trained on a relatively small dataset (~163 examples); best used as a specialised augmentation over the base model
- Primarily focused on Spanish/EU banking regulation
- Not a substitute for professional regulatory advice

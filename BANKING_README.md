# Análisis de Cuentas Anuales - Bancos Españoles

Proyecto para recopilar, procesar y preparar cuentas anuales de los principales bancos españoles para fine-tuning de modelos LLM.

## Bancos Incluidos

- **Banco Santander**
- **CaixaBank**
- **BBVA**
- **Banco Sabadell**
- **Kutxabank**

## Estructura del Proyecto

```
regllm/
├── data/
│   ├── banks_urls.json           # URLs de cuentas anuales
│   ├── raw/                       # PDFs descargados
│   │   ├── santander/
│   │   ├── caixabank/
│   │   ├── bbva/
│   │   ├── sabadell/
│   │   └── kutxabank/
│   ├── processed/                 # Datos extraídos en JSON
│   │   ├── consolidated_data.json
│   │   └── [banco]/[año].json
│   └── finetuning/                # Dataset para entrenamiento
│       └── banking_qa_dataset.jsonl
├── scripts/
│   ├── setup_banking.sh           # Configuración del entorno
│   └── download_financial_reports.py  # Script principal
└── requirements-banking.txt       # Dependencias Python
```

## Instalación

### 1. Configurar el entorno

```bash
cd /home/diego/Development/regllm
./scripts/setup_banking.sh
```

Este script instalará:
- Dependencias Python (requests, PyPDF2, pdfplumber)
- poppler-utils (pdftotext) para extracción de PDFs

### 2. Instalación manual (alternativa)

```bash
# Instalar dependencias Python
pip3 install -r requirements-banking.txt

# Instalar poppler-utils
# En Ubuntu/Debian:
sudo apt-get install poppler-utils

# En Fedora:
sudo dnf install poppler-utils

# En macOS:
brew install poppler
```

## Uso

### 1. Descargar y procesar cuentas anuales

```bash
python3 scripts/download_financial_reports.py
```

Este script realizará:
1. Descarga de PDFs desde las URLs configuradas
2. Extracción de datos financieros (activos, beneficios, ratios, etc.)
3. Guardado de datos en formato JSON
4. Generación de dataset para fine-tuning

### 2. Resultados

Después de ejecutar el script, encontrarás:

- **`data/raw/[banco]/`**: PDFs descargados originales
- **`data/processed/[banco]/`**: JSON individuales por banco y año
- **`data/processed/consolidated_data.json`**: Todos los datos consolidados
- **`data/finetuning/banking_qa_dataset.jsonl`**: Dataset para fine-tuning

## Estructura de Datos JSON

### Archivo individual (ejemplo: `data/processed/santander/santander_2023.json`)

```json
{
  "banco": "santander",
  "año": "2023",
  "archivo_fuente": "santander_2023.pdf",
  "metricas_financieras": {
    "activo_total": "1,720,000",
    "beneficio_neto": "11,076",
    "patrimonio_neto": "127,213",
    "ratio_capital": "12.8",
    "morosidad": "3.14"
  },
  "texto_resumen": "Resumen del informe anual...",
  "datos_clave": [
    "Balance de situación...",
    "Cuenta de pérdidas y ganancias..."
  ]
}
```

## Fine-tuning con Qwen2.5-7B

### Dataset Generado

El archivo `data/finetuning/banking_qa_dataset.jsonl` contiene ejemplos en formato conversacional:

```jsonl
{"messages": [{"role": "user", "content": "¿Cuál fue el beneficio neto de Banco Santander en 2023?"}, {"role": "assistant", "content": "El beneficio neto de Banco Santander en 2023 fue de 11,076 millones de euros."}]}
{"messages": [{"role": "user", "content": "¿Cuáles fueron las principales métricas financieras de BBVA en 2022?"}, {"role": "assistant", "content": "Las principales métricas financieras de BBVA en 2022 fueron..."}]}
```

### Comandos para Fine-tuning

#### Opción 1: Usar Hugging Face Transformers

```bash
# Instalar dependencias
pip install transformers datasets accelerate bitsandbytes

# Script de fine-tuning
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Cargar modelo base
model_name = 'Qwen/Qwen2.5-7B'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    load_in_4bit=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Cargar dataset
dataset = load_dataset('json', data_files='data/finetuning/banking_qa_dataset.jsonl')

# Configurar entrenamiento
training_args = TrainingArguments(
    output_dir='./models/qwen2.5-7b-banking',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=100,
)

# Entrenar
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    tokenizer=tokenizer,
)

trainer.train()
"
```

#### Opción 2: Usar LLaMA-Factory (Recomendado)

```bash
# Clonar LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# Instalar dependencias
pip install -e .

# Copiar dataset
cp ../data/finetuning/banking_qa_dataset.jsonl data/

# Configurar dataset en data/dataset_info.json
# Añadir:
# "banking_qa": {
#   "file_name": "banking_qa_dataset.jsonl",
#   "formatting": "sharegpt",
#   "columns": {
#     "messages": "messages"
#   }
# }

# Entrenar con LoRA (eficiente en memoria)
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --stage sft \
    --do_train \
    --dataset banking_qa \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --output_dir ../models/qwen2.5-7b-banking-lora \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --plot_loss \
    --fp16
```

#### Opción 3: Usar Axolotl

```bash
# Instalar Axolotl
pip install axolotl

# Crear configuración (config.yml)
cat > config.yml << EOF
base_model: Qwen/Qwen2.5-7B
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

load_in_8bit: false
load_in_4bit: true
strict: false

datasets:
  - path: data/finetuning/banking_qa_dataset.jsonl
    type: sharegpt

dataset_prepared_path:
val_set_size: 0.1
output_dir: ./models/qwen2.5-7b-banking

sequence_len: 2048
sample_packing: true

adapter: lora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true

wandb_project:
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

gradient_accumulation_steps: 8
micro_batch_size: 2
num_epochs: 3
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002

train_on_inputs: false
group_by_length: false
bf16: auto
fp16:
tf32: false

gradient_checkpointing: true
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

warmup_steps: 10
evals_per_epoch: 4
eval_table_size:
saves_per_epoch: 1
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
EOF

# Entrenar
accelerate launch -m axolotl.cli.train config.yml
```

### Usar el Modelo Fine-tuneado

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Cargar modelo fine-tuneado
model = AutoModelForCausalLM.from_pretrained(
    "./models/qwen2.5-7b-banking",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# Hacer preguntas
prompt = "¿Cuál fue el beneficio neto de BBVA en 2023?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Personalización

### Añadir más bancos

Edita `data/banks_urls.json` y añade nuevos bancos siguiendo la estructura:

```json
{
  "bancos": {
    "nuevo_banco": {
      "nombre_completo": "Nombre Completo del Banco",
      "web_oficial": "https://...",
      "cuentas_anuales": {
        "2023": {
          "url": "https://.../informe-2023.pdf",
          "descripcion": "Informe Anual 2023"
        }
      }
    }
  }
}
```

### Modificar métricas extraídas

Edita los patrones en `scripts/download_financial_reports.py`, clase `FinancialDataExtractor`, atributo `METRICS_PATTERNS`:

```python
METRICS_PATTERNS = {
    'nueva_metrica': r'patrón_regex_aquí',
    # ...
}
```

## Solución de Problemas

### Error: "No se pudieron extraer datos"

- Verifica que poppler-utils esté instalado: `pdftotext -v`
- Algunos PDFs pueden estar escaneados como imágenes. Considera usar OCR:
  ```bash
  pip install pytesseract pdf2image
  sudo apt-get install tesseract-ocr
  ```

### URLs de PDFs no funcionan

Las URLs de los informes pueden cambiar. Verifica en las páginas oficiales:
- Santander: https://www.santander.com/es/accionistas-e-inversores
- CaixaBank: https://www.caixabank.com/es/accionistas-inversores
- BBVA: https://shareholders.bbva.com/
- Sabadell: https://www.grupbancsabadell.com/
- Kutxabank: https://www.kutxabank.es/

### Memoria insuficiente para fine-tuning

- Usa cuantización de 4 bits (`load_in_4bit=True`)
- Reduce `per_device_train_batch_size`
- Aumenta `gradient_accumulation_steps`
- Usa LoRA en lugar de full fine-tuning
- Considera usar DeepSpeed o FSDP para entrenamiento distribuido

## Notas Legales

Los informes financieros son documentos públicos disponibles en las páginas web oficiales de los bancos. Este proyecto es solo para propósitos educativos y de investigación.

## Contribuciones

Para contribuir:
1. Añade más bancos en `banks_urls.json`
2. Mejora los patrones de extracción en el script
3. Añade más tipos de preguntas para el dataset de fine-tuning

## Licencia

Este proyecto es de código abierto. Los datos financieros pertenecen a sus respectivos bancos.

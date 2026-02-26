# Inicio Rápido - Análisis Bancario

## Resumen del Proyecto

Sistema completo para recopilar, procesar y preparar cuentas anuales de bancos españoles para fine-tuning de LLMs.

## Estructura Creada

```
regllm/
├── data/
│   ├── banks_urls.json                    # URLs de informes bancarios
│   ├── processed/                         # Datos extraídos
│   │   ├── santander/
│   │   ├── caixabank/
│   │   ├── bbva/
│   │   ├── sabadell/
│   │   ├── kutxabank/
│   │   └── consolidated_data.json        # Todos los datos
│   ├── finetuning/
│   │   └── banking_qa_dataset.jsonl      # Dataset para entrenar
│   └── raw/                               # PDFs descargados
│
├── scripts/
│   ├── setup_banking.sh                   # Configuración inicial
│   ├── download_financial_reports.py      # Descarga y procesa PDFs
│   ├── generate_example_data.py           # Genera datos de ejemplo
│   └── dataset_utils.py                   # Utilidades de análisis
│
├── requirements-banking.txt               # Dependencias
└── BANKING_README.md                     # Documentación completa
```

## Pasos de Instalación

### 1. Configurar entorno

```bash
./scripts/setup_banking.sh
```

O manualmente:

```bash
pip3 install -r requirements-banking.txt
sudo apt-get install poppler-utils  # Linux
brew install poppler                # macOS
```

### 2. Generar datos de ejemplo (opcional)

Para probar el sistema inmediatamente:

```bash
python3 scripts/generate_example_data.py
```

Esto creará 82 ejemplos de entrenamiento basados en datos públicos.

### 3. Descargar datos reales (cuando estés listo)

```bash
python3 scripts/download_financial_reports.py
```

Este script:
- Descarga PDFs de cuentas anuales
- Extrae datos financieros
- Genera archivos JSON
- Crea dataset para fine-tuning

## Análisis del Dataset

```bash
# Validar formato
python3 scripts/dataset_utils.py validate

# Analizar contenido
python3 scripts/dataset_utils.py analyze

# Ver ejemplos
python3 scripts/dataset_utils.py samples 10

# Estadísticas
python3 scripts/dataset_utils.py stats

# Todo junto
python3 scripts/dataset_utils.py all
```

## Resultados Actuales

**Dataset generado:**
- ✓ 82 ejemplos de entrenamiento
- ✓ 5 bancos (Santander, CaixaBank, BBVA, Sabadell, Kutxabank)
- ✓ 2 años (2022, 2023)
- ✓ 8 tipos de preguntas (beneficios, solvencia, ROE, morosidad, etc.)
- ✓ Formato validado para Qwen2.5-7B

**Distribución:**
- Preguntas sobre beneficios: 14.6%
- Preguntas generales: 12.2%
- Balance/Activos: 12.2%
- Solvencia: 12.2%
- Rentabilidad: 12.2%
- Morosidad: 12.2%
- Resumen: 12.2%
- Aspectos destacados: 12.2%

## Fine-tuning con Qwen2.5-7B

### Opción 1: LLaMA-Factory (Recomendado)

```bash
# Clonar repositorio
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# Instalar
pip install -e .

# Copiar dataset
cp ../data/finetuning/banking_qa_dataset.jsonl data/

# Añadir a data/dataset_info.json:
{
  "banking_qa": {
    "file_name": "banking_qa_dataset.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    }
  }
}

# Entrenar (LoRA)
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --stage sft \
    --do_train \
    --dataset banking_qa \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --output_dir ../models/qwen2.5-7b-banking \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --fp16
```

### Opción 2: Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from datasets import load_dataset

# Cargar modelo
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-7B',
    device_map='auto',
    load_in_4bit=True
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')

# Cargar dataset
dataset = load_dataset('json', data_files='data/finetuning/banking_qa_dataset.jsonl')

# Entrenar (ver BANKING_README.md para código completo)
```

## Próximos Pasos

1. **Expandir dataset:**
   - Añadir más años (2020, 2021, 2024)
   - Incluir más bancos españoles
   - Añadir bancos internacionales

2. **Mejorar extracción:**
   - Implementar OCR para PDFs escaneados
   - Extraer gráficos y tablas
   - Añadir más métricas financieras

3. **Fine-tuning avanzado:**
   - Experimentar con diferentes hiperparámetros
   - Probar con otros modelos (Mistral, Llama, etc.)
   - Implementar validación cruzada

4. **Evaluación:**
   - Crear conjunto de test
   - Métricas de evaluación (perplexity, accuracy)
   - Comparar con modelo base

## Archivos Importantes

- **BANKING_README.md**: Documentación completa
- **data/finetuning/banking_qa_dataset.jsonl**: Dataset listo para entrenar
- **data/processed/consolidated_data.json**: Todos los datos estructurados
- **data/banks_urls.json**: URLs de fuentes oficiales

## Soporte

Para problemas comunes, consulta BANKING_README.md sección "Solución de Problemas".

## Notas

- Los datos de ejemplo son estimaciones basadas en información pública
- Las URLs de PDFs pueden cambiar; verificar en sitios oficiales
- El fine-tuning requiere GPU con al menos 12GB VRAM (con cuantización 4-bit)

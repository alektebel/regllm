# Uso del Modelo Fine-tuneado

## Descripción

Script para interactuar con el modelo Qwen2.5-7B fine-tuneado con datos bancarios.

## Requisitos Previos

1. Haber completado el fine-tuning del modelo (ver BANKING_README.md)
2. Tener el modelo guardado en `./models/qwen2.5-7b-banking/`

## Instalación de Dependencias

```bash
pip install transformers torch bitsandbytes accelerate
```

## Modos de Uso

### 1. Modo Interactivo (Recomendado)

Permite hacer preguntas en tiempo real:

```bash
python3 scripts/use_banking_model.py --mode interactive
```

Ejemplo de sesión:
```
💬 Tu pregunta: ¿Cuál fue el beneficio neto de BBVA en 2023?
🤖 Respuesta: El beneficio neto de BBVA en 2023 fue de 8.019 millones EUR.

💬 Tu pregunta: Compara el ROE de Santander y CaixaBank
🤖 Respuesta: En 2023, Banco Santander tuvo un ROE del 14.8%, mientras que CaixaBank alcanzó un 12.5%.
```

### 2. Modo Batch

Ejecuta un conjunto predefinido de preguntas de prueba:

```bash
python3 scripts/use_banking_model.py --mode batch
```

### 3. Modo Single

Para una pregunta única:

```bash
python3 scripts/use_banking_model.py --mode single \
    --question "¿Cuál fue la tasa de morosidad de Kutxabank en 2023?"
```

## Opciones Avanzadas

### Usar modelo de otra ubicación

```bash
python3 scripts/use_banking_model.py \
    --model /ruta/a/tu/modelo \
    --mode interactive
```

### Desactivar cuantización 4-bit

Si tienes suficiente VRAM (>24GB):

```bash
python3 scripts/use_banking_model.py \
    --no-4bit \
    --mode interactive
```

## Ejemplos de Preguntas

El modelo ha sido entrenado para responder preguntas sobre:

### Beneficios y Resultados
- "¿Cuánto ganó Banco Santander en 2023?"
- "¿Cuál fue el resultado neto de CaixaBank en 2022?"

### Balance y Activos
- "¿Cuál es el tamaño del balance de BBVA en 2023?"
- "¿Cuál fue el activo total de Sabadell en 2022?"

### Solvencia y Capital
- "¿Cómo está la solvencia de Kutxabank en 2023?"
- "¿Qué ratio de capital tiene CaixaBank?"

### Rentabilidad
- "¿Qué ROE tuvo BBVA en 2023?"
- "¿Cuál fue la rentabilidad sobre recursos propios de Santander?"

### Morosidad
- "¿Cuál fue la tasa de morosidad de Kutxabank en 2023?"
- "¿Cómo está la mora en Sabadell?"

### Comparativas
- "Compara los beneficios de Santander y BBVA en 2023"
- "¿Qué banco tiene mejor ratio de capital, CaixaBank o Kutxabank?"

### Resúmenes
- "Dame un resumen del desempeño de BBVA en 2023"
- "¿Cuáles fueron los aspectos más destacados de Santander en 2022?"

## Integración en Código Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Cargar modelo
model = AutoModelForCausalLM.from_pretrained(
    "./models/qwen2.5-7b-banking",
    device_map="auto",
    load_in_4bit=True
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# Hacer pregunta
prompt = "Usuario: ¿Cuál fue el beneficio neto de BBVA en 2023?\nAsistente:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generar respuesta
outputs = model.generate(
    **inputs,
    max_length=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## API REST (Opcional)

Para crear una API REST con FastAPI:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Cargar modelo al inicio (código arriba)
model, tokenizer = load_model("./models/qwen2.5-7b-banking")

class Question(BaseModel):
    text: str

@app.post("/ask")
async def ask_question(question: Question):
    response = generate_response(model, tokenizer, question.text)
    return {"question": question.text, "answer": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Ejecutar:
```bash
pip install fastapi uvicorn
python api.py
```

Usar:
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"text": "¿Cuál fue el beneficio de BBVA en 2023?"}'
```

## Troubleshooting

### Error: CUDA out of memory

Soluciones:
1. Usar cuantización 4-bit (por defecto)
2. Reducir `max_length` en la generación
3. Usar una GPU con más VRAM
4. Usar CPU (muy lento):
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       "./models/qwen2.5-7b-banking",
       device_map="cpu"
   )
   ```

### El modelo responde mal o de forma incoherente

Posibles causas:
1. Insufficient training (aumentar epochs)
2. Learning rate demasiado alto/bajo
3. Dataset pequeño (añadir más ejemplos)
4. Evaluar con validation set

### Respuestas muy cortas

Ajustar parámetros de generación:
```python
outputs = model.generate(
    **inputs,
    max_length=512,        # Aumentar
    min_length=50,         # Añadir mínimo
    temperature=0.8,       # Aumentar creatividad
    top_p=0.95,           # Aumentar diversidad
)
```

## Métricas de Evaluación

Para evaluar el modelo:

```python
from datasets import load_dataset
from transformers import pipeline

# Cargar test set
test_data = load_dataset('json', data_files='data/test.jsonl')

# Crear pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Evaluar
correct = 0
total = len(test_data['train'])

for example in test_data['train']:
    question = example['messages'][0]['content']
    expected = example['messages'][1]['content']
    
    generated = pipe(f"Usuario: {question}\nAsistente:")
    
    # Comparar (implementar métrica apropiada)
    # Por ejemplo, BLEU, ROUGE, o exact match
    
print(f"Accuracy: {correct/total*100:.2f}%")
```

## Mejoras Futuras

1. **Añadir más datos**: Expandir dataset con más años y bancos
2. **Fine-tuning incremental**: Reentrenar con datos actualizados
3. **RAG (Retrieval-Augmented Generation)**: Combinar con búsqueda de documentos
4. **Multi-turn conversations**: Soporte para diálogos largos
5. **Validación de respuestas**: Verificar coherencia de números

## Recursos Adicionales

- [Documentación Qwen2.5](https://github.com/QwenLM/Qwen2.5)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

# Guía de Implementación: Asistente de Consultoría en Regulación Bancaria

## Resumen Ejecutivo

Esta guía describe la implementación completa de un asistente de consultoría especializado en regulación bancaria (Basilea III, estimación de LGD, PD, etc.) usando un modelo de lenguaje de 7B parámetros con RAG (Retrieval-Augmented Generation) configurado para responder **exclusivamente en español**.

**Hardware objetivo:** NVIDIA RTX 5060 16GB VRAM

**Capacidades principales:**
- Respuestas a preguntas sobre regulación bancaria en español
- Citación de fuentes regulatorias específicas
- Auto-verificación de respuestas
- Explicaciones en español de conceptos técnicos

---

## Tabla de Contenidos

1. [Arquitectura del Sistema](#arquitectura-del-sistema)
2. [Selección del Modelo](#selección-del-modelo)
3. [Preparación de Datos](#preparación-de-datos)
4. [Fine-tuning para Español](#fine-tuning-para-español)
5. [Implementación RAG](#implementación-rag)
6. [Sistema de Verificación](#sistema-de-verificación)
7. [Configuración Técnica](#configuración-técnica)
8. [Testing y Validación](#testing-y-validación)
9. [Despliegue](#despliegue)

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    Usuario (Consultor)                       │
│              Pregunta en español sobre regulación            │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              ETAPA 1: Recuperación (Retrieval)               │
│  • Embedding de la pregunta (modelo multilingüe)            │
│  • Búsqueda en Vector DB (documentos regulatorios)          │
│  • Recuperación de top-k fragmentos relevantes              │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│         ETAPA 2: Generación (7B LLM Fine-tuned)             │
│  • Modelo configurado para responder SOLO en español        │
│  • Genera respuesta basada en fragmentos recuperados        │
│  • Incluye citaciones específicas a artículos/párrafos      │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              ETAPA 3: Auto-verificación                      │
│  • Re-lectura de fuentes citadas                            │
│  • Verificación de coherencia entre respuesta y fuente      │
│  • Detección de citaciones inventadas                       │
│  • Asignación de nivel de confianza                         │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  ETAPA 4: Presentación                       │
│  • Respuesta en español                                     │
│  • Citas textuales de la regulación                         │
│  • Enlaces a documentos fuente                              │
│  • Indicadores de confianza y advertencias                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Selección del Modelo

### Modelos Recomendados para Español

Para 16GB VRAM, estos modelos tienen buen soporte de español:

#### **Opción 1: Llama 3.1 8B (Recomendado)**
```python
modelo = "meta-llama/Meta-Llama-3.1-8B-Instruct"
```
**Ventajas:**
- Excelente capacidad multilingüe (entrenado con datos en español)
- Buen razonamiento para tareas técnicas
- Ampliamente adoptado, mucha documentación
- Formatos de chat bien definidos

**Desventajas:**
- Necesita cuantización a 4-bit para 16GB VRAM

#### **Opción 2: Qwen2.5 7B**
```python
modelo = "Qwen/Qwen2.5-7B-Instruct"
```
**Ventajas:**
- Rendimiento superior en tareas técnicas/financieras
- Buen soporte multilingüe
- Menor uso de memoria

**Desventajas:**
- Menos recursos en español comparado con Llama

#### **Opción 3: Mistral 7B v0.3**
```python
modelo = "mistralai/Mistral-7B-Instruct-v0.3"
```
**Ventajas:**
- Excelente seguimiento de instrucciones
- Buena capacidad en español (empresa francesa)
- Eficiente en memoria

**Comparativa de rendimiento en español:**

| Modelo | Calidad ES | Memoria | Velocidad | Técnico/Financiero |
|--------|-----------|---------|-----------|-------------------|
| Llama 3.1 8B | ★★★★★ | 8GB (4-bit) | ★★★★☆ | ★★★★☆ |
| Qwen2.5 7B | ★★★★☆ | 7GB (4-bit) | ★★★★★ | ★★★★★ |
| Mistral 7B | ★★★★☆ | 6GB (4-bit) | ★★★★★ | ★★★★☆ |

**Recomendación:** Llama 3.1 8B para mejor calidad en español, Qwen2.5 7B si priorizas rendimiento técnico.

---

## Preparación de Datos

### Estructura de Datos para Fine-tuning en Español

Los datos deben estar en formato JSONL con el siguiente esquema:

```jsonl
{
  "messages": [
    {
      "role": "system",
      "content": "Eres un asistente experto en regulación bancaria y parámetros de riesgo de crédito. Siempre respondes en español, citas fuentes específicas y proporcionas explicaciones claras y precisas."
    },
    {
      "role": "user",
      "content": "¿Cuál es el ratio mínimo de capital CET1 según Basilea III?\n\nFuente: [Reglamento CRR Art. 92(1)(a)]: 'Las entidades deberán cumplir en todo momento los siguientes requisitos de fondos propios: (a) un ratio de capital de nivel 1 ordinario del 4,5%'"
    },
    {
      "role": "assistant",
      "content": "Según Basilea III, el ratio mínimo de capital CET1 (Common Equity Tier 1) es del **4,5%** de los activos ponderados por riesgo [Reglamento CRR Artículo 92(1)(a)].\n\nAdicionalmente, debe considerarse el colchón de conservación de capital del 2,5%, lo que resulta en un requisito efectivo del 7% en condiciones normales.\n\n📚 **Fuente regulatoria:**\nReglamento (UE) n.º 575/2013 (CRR), Artículo 92, apartado 1, letra a: \"un ratio de capital de nivel 1 ordinario del 4,5%\""
    }
  ]
}
```

### Creación del Dataset de Entrenamiento

#### **1. Recopilación de Documentos Regulatorios en Español**

**Fuentes principales:**
- Reglamento CRR (UE) 575/2013 (versión consolidada en español)
- Directrices EBA sobre estimación de PD y LGD (traducción oficial)
- Normativa local del Banco de España
- Orientaciones del Comité de Basilea (documentos en español)
- Normas técnicas de implementación (ITS/RTS)

**Script de descarga y procesamiento:**

```python
import requests
from bs4 import BeautifulSoup
import PyPDF2
import re

def extraer_regulacion_pdf(pdf_path):
    """
    Extrae texto de PDFs regulatorios manteniendo estructura
    """
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        texto_completo = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            texto = page.extract_text()
            
            # Limpiar texto
            texto = re.sub(r'\s+', ' ', texto)
            texto = re.sub(r'(\d+)\s*\n\s*(\d+)', r'\1\2', texto)  # Unir números separados
            
            texto_completo.append({
                'pagina': page_num + 1,
                'texto': texto
            })
    
    return texto_completo

def segmentar_por_articulos(texto):
    """
    Divide regulación en artículos/secciones
    """
    # Patrón para detectar artículos: "Artículo 92", "Art. 92", etc.
    patron_articulo = r'(Artículo|Art\.?)\s+(\d+[a-z]?)'
    
    segmentos = []
    matches = list(re.finditer(patron_articulo, texto, re.IGNORECASE))
    
    for i, match in enumerate(matches):
        inicio = match.start()
        fin = matches[i+1].start() if i+1 < len(matches) else len(texto)
        
        articulo_num = match.group(2)
        contenido = texto[inicio:fin].strip()
        
        segmentos.append({
            'articulo': articulo_num,
            'contenido': contenido,
            'tipo': 'articulo'
        })
    
    return segmentos

# Ejemplo de uso
regulacion = extraer_regulacion_pdf('CRR_español.pdf')
articulos = segmentar_por_articulos(regulacion)
```

#### **2. Generación de Pares Q&A en Español**

**Estrategia híbrida:**
1. Crear manualmente 100-200 ejemplos de alta calidad
2. Usar LLM grande (GPT-4/Claude) para generar ejemplos sintéticos
3. Validar y corregir ejemplos generados

**Script para generación sintética:**

```python
from anthropic import Anthropic

client = Anthropic(api_key="tu-api-key")

def generar_qa_desde_articulo(articulo_texto, metadata):
    """
    Usa Claude para generar pares Q&A en español desde artículos regulatorios
    """
    
    prompt = f"""Eres un experto en regulación bancaria. Genera 3 pares de pregunta-respuesta en español basados en este fragmento regulatorio.

FRAGMENTO REGULATORIO:
{articulo_texto}

METADATA:
- Documento: {metadata['documento']}
- Artículo: {metadata['articulo']}

REQUISITOS:
1. Las preguntas deben ser naturales, como las que haría un consultor
2. Las respuestas deben citar específicamente el artículo/párrafo
3. Incluir la cita textual relevante entre comillas
4. Todo en español profesional

FORMATO DE SALIDA (JSON):
{{
  "qa_pairs": [
    {{
      "pregunta": "...",
      "respuesta": "...",
      "cita_textual": "...",
      "referencia": "..."
    }}
  ]
}}
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# Ejemplo de uso
articulo_92 = {
    'texto': 'Artículo 92: Las entidades deberán cumplir en todo momento...',
    'metadata': {
        'documento': 'Reglamento CRR (UE) 575/2013',
        'articulo': '92'
    }
}

qa_pairs = generar_qa_desde_articulo(articulo_92['texto'], articulo_92['metadata'])
```

#### **3. Formateo para Fine-tuning**

```python
import json

def crear_dataset_entrenamiento(qa_pairs, output_file='training_data.jsonl'):
    """
    Convierte pares Q&A a formato de entrenamiento con system prompt en español
    """
    
    system_prompt = """Eres un asistente experto en regulación bancaria, especializado en Basilea III, parámetros de riesgo de crédito (PD, LGD, EAD) y normativa prudencial.

REGLAS OBLIGATORIAS:
1. Respondes SIEMPRE en español
2. Citas fuentes regulatorias específicas (artículos, párrafos, secciones)
3. Incluyes citas textuales relevantes entre comillas
4. Si no tienes información en las fuentes proporcionadas, lo indicas claramente
5. Mantienes precisión técnica y claridad en explicaciones
6. No inventas referencias o artículos que no existen"""

    with open(output_file, 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            # Construir contexto con fuentes
            contexto_fuentes = f"\n\nFuente: [{qa['referencia']}]: \"{qa['cita_textual']}\""
            
            entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": qa['pregunta'] + contexto_fuentes},
                    {"role": "assistant", "content": qa['respuesta']}
                ]
            }
            
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Crear dataset
crear_dataset_entrenamiento(parsed_qa_pairs)
```

#### **4. Calidad del Dataset**

**Checklist de calidad:**
- [ ] Mínimo 500 ejemplos de alta calidad
- [ ] Cobertura de temas: CET1, ratios, LGD, PD, EAD, Basilea III
- [ ] Variedad de tipos de preguntas (definiciones, cálculos, procedimientos)
- [ ] Todas las respuestas en español correcto
- [ ] Citaciones verificadas manualmente
- [ ] Balance entre preguntas simples (30%), medias (50%), complejas (20%)

**Distribución recomendada:**

```python
distribucion_temas = {
    'Requisitos de capital (CET1, Tier 1, etc.)': 150,  # 30%
    'Parámetros de riesgo (PD, LGD, EAD)': 150,        # 30%
    'Metodologías de estimación': 100,                  # 20%
    'Definiciones y conceptos': 50,                     # 10%
    'Procedimientos y procesos': 50                     # 10%
}
```

---

## Fine-tuning para Español

### Configuración de LoRA para Español

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch

# Configuración de cuantización (4-bit para 16GB VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Cargar modelo base
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Preparar modelo para entrenamiento con QLoRA
model = prepare_model_for_kbit_training(model)

# Configuración LoRA optimizada para dominio técnico
lora_config = LoraConfig(
    r=64,  # Rank aumentado para capturar complejidad regulatoria
    lora_alpha=128,  # Scaling factor (típicamente 2*r)
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Mostrar parámetros entrenables
model.print_trainable_parameters()
# Salida esperada: ~1-2% de parámetros totales (~80-160M de 8B)
```

### Argumentos de Entrenamiento

```python
training_args = TrainingArguments(
    output_dir="./llama3-regulatory-spanish",
    
    # Hiperparámetros críticos
    num_train_epochs=3,  # 2-3 épocas para evitar overfitting
    per_device_train_batch_size=4,  # Ajustar según VRAM disponible
    gradient_accumulation_steps=4,  # Batch efectivo = 4*4 = 16
    
    learning_rate=2e-5,  # LR conservador para dominio técnico
    lr_scheduler_type="cosine",  # Decaimiento suave
    warmup_ratio=0.03,  # 3% de pasos de calentamiento
    
    # Regularización
    weight_decay=0.01,
    max_grad_norm=0.3,  # Clip gradientes
    
    # Optimización de memoria
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",  # Optimizador eficiente en memoria
    
    # Logging y guardado
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=3,
    
    # Otros
    fp16=True,  # Precisión mixta
    report_to="none",  # O "wandb" para tracking
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)
```

### Entrenamiento con SFTTrainer

```python
from datasets import load_dataset

# Cargar dataset
dataset = load_dataset('json', data_files='training_data.jsonl', split='train')

# Split train/validation
train_test = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test['train']
eval_dataset = train_test['test']

# Formatear para chat template de Llama
def format_chat_template(example):
    """
    Formatea usando el chat template de Llama 3.1
    """
    messages = example['messages']
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {'text': text}

train_dataset = train_dataset.map(format_chat_template)
eval_dataset = eval_dataset.map(format_chat_template)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=2048,  # Longitud máxima de secuencia
    packing=False  # No empaquetar para mantener conversaciones intactas
)

# Entrenar
trainer.train()

# Guardar modelo final
trainer.save_model("./llama3-regulatory-spanish-final")
```

### Forzar Respuestas en Español

**Método 1: System Prompt Reforzado**

```python
SYSTEM_PROMPT_ESPAÑOL = """Eres un asistente experto en regulación bancaria que SIEMPRE responde en español.

REGLA CRÍTICA: Todas tus respuestas deben estar en español. Nunca respondas en inglés u otro idioma.

Especialización:
- Basilea III y normativa prudencial
- Parámetros de riesgo de crédito (PD, LGD, EAD, CCF)
- Regulación europea (CRR, CRD) y directrices EBA
- Normativa del Banco de España

Estilo de respuesta:
- Citas específicas a artículos y párrafos regulatorios
- Explicaciones claras en español técnico pero accesible
- Incluir citas textuales relevantes entre comillas
- Indicar nivel de confianza cuando sea apropiado"""
```

**Método 2: Configuración de Generación**

```python
def generar_respuesta_español(pregunta, contexto_fuentes, model, tokenizer):
    """
    Genera respuesta forzando español mediante configuración
    """
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_ESPAÑOL},
        {"role": "user", "content": f"{pregunta}\n\nFuentes:\n{contexto_fuentes}"}
    ]
    
    # Aplicar chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    # Generar con configuración específica
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,  # Creatividad moderada
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,  # Evitar repeticiones
        pad_token_id=tokenizer.eos_token_id
    )
    
    respuesta = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                  skip_special_tokens=True)
    
    return respuesta
```

**Método 3: Validación Post-generación**

```python
import langdetect

def validar_idioma_respuesta(respuesta):
    """
    Valida que la respuesta esté en español
    """
    try:
        idioma = langdetect.detect(respuesta)
        if idioma != 'es':
            return False, f"Idioma detectado: {idioma}"
        return True, "Español confirmado"
    except:
        return False, "No se pudo detectar idioma"

# Uso en pipeline
respuesta = generar_respuesta_español(pregunta, fuentes, model, tokenizer)
es_español, mensaje = validar_idioma_respuesta(respuesta)

if not es_español:
    print(f"⚠️ ADVERTENCIA: {mensaje}")
    # Regenerar o rechazar respuesta
```

---

## Implementación RAG

### Arquitectura del Sistema RAG

```python
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np

class RegulatoryRAGSystem:
    """
    Sistema RAG completo para consultas regulatorias en español
    """
    
    def __init__(self, 
                 embedding_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 persist_directory="./chroma_db"):
        
        # Modelo de embeddings multilingüe (español incluido)
        self.embedder = SentenceTransformer(embedding_model_name)
        
        # Base de datos vectorial
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Colección para documentos regulatorios
        self.collection = self.client.get_or_create_collection(
            name="regulacion_bancaria",
            metadata={"description": "Documentos regulatorios en español"}
        )
        
    def procesar_documentos(self, documentos):
        """
        Procesa y almacena documentos regulatorios
        
        Args:
            documentos: Lista de dicts con 'texto', 'metadata'
        """
        textos = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(documentos):
            # Segmentar documento en chunks
            chunks = self._segmentar_documento(doc['texto'], doc['metadata'])
            
            for j, chunk in enumerate(chunks):
                textos.append(chunk['texto'])
                metadatas.append(chunk['metadata'])
                ids.append(f"{doc['metadata'].get('documento_id', i)}_{j}")
        
        # Generar embeddings
        embeddings = self.embedder.encode(textos, show_progress_bar=True)
        
        # Almacenar en ChromaDB
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=textos,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✓ Procesados {len(textos)} chunks de {len(documentos)} documentos")
    
    def _segmentar_documento(self, texto, metadata):
        """
        Segmenta documento por artículos/párrafos manteniendo contexto
        """
        import re
        
        chunks = []
        
        # Detectar artículos
        patron = r'(Artículo\s+\d+[a-z]?[.\s])'
        partes = re.split(patron, texto)
        
        articulo_actual = None
        for parte in partes:
            if re.match(patron, parte):
                articulo_actual = parte.strip()
            elif articulo_actual and len(parte.strip()) > 50:
                # Dividir artículos largos en párrafos
                parrafos = parte.split('\n')
                
                for parrafo in parrafos:
                    if len(parrafo.strip()) > 100:  # Párrafos significativos
                        chunk_metadata = metadata.copy()
                        chunk_metadata['articulo'] = articulo_actual
                        chunk_metadata['longitud'] = len(parrafo)
                        
                        chunks.append({
                            'texto': f"{articulo_actual}\n{parrafo.strip()}",
                            'metadata': chunk_metadata
                        })
        
        return chunks
    
    def buscar_contexto(self, pregunta, n_resultados=5):
        """
        Busca fragmentos relevantes para una pregunta
        
        Args:
            pregunta: Pregunta del usuario en español
            n_resultados: Número de chunks a recuperar
            
        Returns:
            Lista de chunks con texto y metadata
        """
        # Embed pregunta
        query_embedding = self.embedder.encode([pregunta])[0]
        
        # Buscar en ChromaDB
        resultados = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_resultados
        )
        
        chunks = []
        for i in range(len(resultados['documents'][0])):
            chunks.append({
                'texto': resultados['documents'][0][i],
                'metadata': resultados['metadatas'][0][i],
                'distancia': resultados['distances'][0][i] if 'distances' in resultados else None
            })
        
        return chunks
    
    def responder_pregunta(self, pregunta, model, tokenizer):
        """
        Pipeline completo: recuperar + generar + verificar
        """
        # 1. Recuperar contexto
        chunks_relevantes = self.buscar_contexto(pregunta, n_resultados=5)
        
        # 2. Construir contexto para el modelo
        contexto = self._formatear_contexto(chunks_relevantes)
        
        # 3. Generar respuesta
        respuesta = generar_respuesta_español(pregunta, contexto, model, tokenizer)
        
        # 4. Verificar respuesta
        verificacion = self._verificar_respuesta(respuesta, chunks_relevantes)
        
        return {
            'pregunta': pregunta,
            'respuesta': respuesta,
            'fuentes': chunks_relevantes,
            'verificacion': verificacion
        }
    
    def _formatear_contexto(self, chunks):
        """
        Formatea chunks recuperados para el prompt
        """
        contexto_partes = []
        
        for i, chunk in enumerate(chunks, 1):
            meta = chunk['metadata']
            fuente = f"[{meta.get('documento', 'Desconocido')} {meta.get('articulo', '')}]"
            
            contexto_partes.append(
                f"Fuente {i}: {fuente}\n\"{chunk['texto']}\"\n"
            )
        
        return "\n".join(contexto_partes)
    
    def _verificar_respuesta(self, respuesta, chunks_fuente):
        """
        Verifica que la respuesta esté fundamentada en las fuentes
        """
        import re
        
        # Extraer citaciones de la respuesta
        patron_cita = r'\[([^\]]+)\]'
        citaciones = re.findall(patron_cita, respuesta)
        
        verificaciones = []
        
        for cita in citaciones:
            # Buscar si la citación existe en los chunks
            encontrada = False
            for chunk in chunks_fuente:
                if cita.lower() in chunk['texto'].lower() or \
                   cita.lower() in str(chunk['metadata']).lower():
                    encontrada = True
                    break
            
            verificaciones.append({
                'citacion': cita,
                'verificada': encontrada
            })
        
        # Calcular nivel de confianza
        if not citaciones:
            confianza = "BAJA - Sin citaciones"
        elif all(v['verificada'] for v in verificaciones):
            confianza = "ALTA - Todas las citaciones verificadas"
        else:
            tasa_verificacion = sum(v['verificada'] for v in verificaciones) / len(verificaciones)
            confianza = f"MEDIA - {tasa_verificacion:.0%} citaciones verificadas"
        
        return {
            'citaciones': verificaciones,
            'nivel_confianza': confianza,
            'num_fuentes_usadas': len(chunks_fuente)
        }
```

### Preparación de la Base de Datos Vectorial

```python
# Script de inicialización

# 1. Cargar documentos regulatorios
documentos_regulatorios = [
    {
        'texto': open('CRR_español.txt', 'r', encoding='utf-8').read(),
        'metadata': {
            'documento': 'Reglamento CRR (UE) 575/2013',
            'documento_id': 'CRR',
            'tipo': 'reglamento',
            'fecha_vigencia': '2014-01-01',
            'jurisdiccion': 'UE'
        }
    },
    {
        'texto': open('EBA_GL_PD_LGD.txt', 'r', encoding='utf-8').read(),
        'metadata': {
            'documento': 'Directrices EBA sobre PD y LGD',
            'documento_id': 'EBA_PD_LGD',
            'tipo': 'directriz',
            'fecha_publicacion': '2017-11-21',
            'jurisdiccion': 'UE'
        }
    },
    # ... más documentos
]

# 2. Inicializar sistema RAG
rag_system = RegulatoryRAGSystem()

# 3. Procesar y almacenar documentos
rag_system.procesar_documentos(documentos_regulatorios)

# 4. Verificar
print(f"Total de chunks en la base de datos: {rag_system.collection.count()}")
```

### Optimización de Embeddings para Español

```python
# Comparativa de modelos de embeddings multilingües

modelos_embeddings = {
    'paraphrase-multilingual-mpnet-base-v2': {
        'dimensiones': 768,
        'velocidad': '★★★★☆',
        'calidad_español': '★★★★★',
        'memoria': '~500MB'
    },
    'paraphrase-multilingual-MiniLM-L12-v2': {
        'dimensiones': 384,
        'velocidad': '★★★★★',
        'calidad_español': '★★★★☆',
        'memoria': '~200MB'
    },
    'distiluse-base-multilingual-cased-v2': {
        'dimensiones': 512,
        'velocidad': '★★★★★',
        'calidad_español': '★★★★☆',
        'memoria': '~300MB'
    }
}

# Recomendación: paraphrase-multilingual-mpnet-base-v2 para mejor calidad
# O MiniLM si necesitas mayor velocidad
```

### Búsqueda Híbrida (Semántica + Palabras Clave)

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearch:
    """
    Combina búsqueda semántica (embeddings) con BM25 (palabras clave)
    """
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.bm25 = None
        self.corpus = []
        
    def indexar_corpus(self):
        """
        Indexa todos los documentos para BM25
        """
        # Obtener todos los documentos
        all_docs = self.rag_system.collection.get()
        
        self.corpus = all_docs['documents']
        
        # Tokenizar para BM25
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"✓ Corpus indexado: {len(self.corpus)} documentos")
    
    def buscar_hibrida(self, pregunta, n_resultados=5, peso_semantico=0.7):
        """
        Búsqueda híbrida con re-ranking
        
        Args:
            pregunta: Query del usuario
            n_resultados: Número de resultados finales
            peso_semantico: 0-1, peso de búsqueda semántica vs. BM25
        """
        # 1. Búsqueda semántica
        resultados_semanticos = self.rag_system.buscar_contexto(
            pregunta, 
            n_resultados=n_resultados*2
        )
        
        # 2. Búsqueda BM25
        tokenized_query = pregunta.lower().split()
        scores_bm25 = self.bm25.get_scores(tokenized_query)
        
        # 3. Combinar scores (normalizar antes)
        scores_combinados = []
        
        for chunk in resultados_semanticos:
            idx = self.corpus.index(chunk['texto'])
            
            # Normalizar scores entre 0-1
            score_sem = 1 - (chunk['distancia'] / max(c['distancia'] for c in resultados_semanticos))
            score_bm25_norm = scores_bm25[idx] / max(scores_bm25)
            
            score_final = (peso_semantico * score_sem + 
                          (1 - peso_semantico) * score_bm25_norm)
            
            scores_combinados.append({
                'chunk': chunk,
                'score': score_final
            })
        
        # 4. Re-rankear y devolver top-N
        scores_combinados.sort(key=lambda x: x['score'], reverse=True)
        
        return [s['chunk'] for s in scores_combinados[:n_resultados]]
```

---

## Sistema de Verificación

### Auto-verificación de Citaciones

```python
class SistemaVerificacion:
    """
    Verifica la exactitud de respuestas generadas
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def verificar_respuesta(self, pregunta, respuesta, fuentes):
        """
        Verifica que la respuesta esté fundamentada en las fuentes
        """
        verificaciones = {
            'citaciones': self._verificar_citaciones(respuesta, fuentes),
            'coherencia': self._verificar_coherencia(respuesta, fuentes),
            'hallucination': self._detectar_hallucination(respuesta, fuentes),
            'idioma': self._verificar_español(respuesta)
        }
        
        # Calcular score global
        score_global = self._calcular_score_confianza(verificaciones)
        
        return {
            'verificaciones': verificaciones,
            'score_confianza': score_global,
            'aprobada': score_global >= 0.7
        }
    
    def _verificar_citaciones(self, respuesta, fuentes):
        """
        Verifica que las citaciones existan en las fuentes
        """
        import re
        
        # Extraer citaciones: [Artículo X], [CRR Art. Y], etc.
        patron = r'\[((?:Art(?:ículo|\.)?|CRR|EBA|Reglamento)[^\]]+)\]'
        citaciones = re.findall(patron, respuesta, re.IGNORECASE)
        
        resultados = []
        for cita in citaciones:
            # Buscar en fuentes
            encontrada = False
            fuente_origen = None
            
            for fuente in fuentes:
                if cita.lower() in fuente['texto'].lower() or \
                   cita.lower() in str(fuente['metadata']).lower():
                    encontrada = True
                    fuente_origen = fuente['metadata'].get('documento', 'Desconocido')
                    break
            
            resultados.append({
                'citacion': cita,
                'verificada': encontrada,
                'fuente': fuente_origen
            })
        
        tasa_verificacion = (sum(r['verificada'] for r in resultados) / len(resultados) 
                            if resultados else 0)
        
        return {
            'citaciones_encontradas': len(citaciones),
            'citaciones_verificadas': sum(r['verificada'] for r in resultados),
            'tasa_verificacion': tasa_verificacion,
            'detalles': resultados
        }
    
    def _verificar_coherencia(self, respuesta, fuentes):
        """
        Usa el LLM para verificar coherencia entre respuesta y fuentes
        """
        # Concatenar fuentes
        contexto_fuentes = "\n\n".join([f['texto'][:500] for f in fuentes[:3]])
        
        prompt_verificacion = f"""Eres un verificador de exactitud. Compara esta respuesta con las fuentes proporcionadas.

RESPUESTA A VERIFICAR:
{respuesta}

FUENTES ORIGINALES:
{contexto_fuentes}

¿La respuesta es coherente con las fuentes? Responde solo: COHERENTE, PARCIALMENTE_COHERENTE, o INCOHERENTE.
Luego explica brevemente por qué."""

        messages = [
            {"role": "system", "content": "Eres un verificador de exactitud que responde en español."},
            {"role": "user", "content": prompt_verificacion}
        ]
        
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=150, temperature=0.3)
        
        verificacion = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parsear resultado
        if "COHERENTE" in verificacion.upper() and "PARCIALMENTE" not in verificacion.upper():
            nivel = "COHERENTE"
            score = 1.0
        elif "PARCIALMENTE" in verificacion.upper():
            nivel = "PARCIALMENTE_COHERENTE"
            score = 0.6
        else:
            nivel = "INCOHERENTE"
            score = 0.2
        
        return {
            'nivel': nivel,
            'score': score,
            'explicacion': verificacion
        }
    
    def _detectar_hallucination(self, respuesta, fuentes):
        """
        Detecta posibles hallucinations (información inventada)
        """
        import re
        
        # Extraer afirmaciones numéricas específicas
        patron_numeros = r'(\d+(?:\.\d+)?)\s*%'
        numeros_respuesta = re.findall(patron_numeros, respuesta)
        
        hallucination_detectada = False
        detalles = []
        
        for numero in numeros_respuesta:
            # Verificar si este número aparece en las fuentes
            encontrado = any(numero in f['texto'] for f in fuentes)
            
            if not encontrado:
                hallucination_detectada = True
                detalles.append(f"Número {numero}% no encontrado en fuentes")
        
        return {
            'hallucination_detectada': hallucination_detectada,
            'confianza': 0.3 if hallucination_detectada else 0.9,
            'detalles': detalles
        }
    
    def _verificar_español(self, respuesta):
        """
        Verifica que la respuesta esté en español
        """
        try:
            import langdetect
            idioma = langdetect.detect(respuesta)
            es_español = (idioma == 'es')
            
            return {
                'es_español': es_español,
                'idioma_detectado': idioma,
                'confianza': 1.0 if es_español else 0.0
            }
        except:
            return {
                'es_español': True,  # Asumir español si falla detección
                'idioma_detectado': 'desconocido',
                'confianza': 0.5
            }
    
    def _calcular_score_confianza(self, verificaciones):
        """
        Calcula score global de confianza (0-1)
        """
        scores = [
            verificaciones['citaciones']['tasa_verificacion'] * 0.3,  # 30%
            verificaciones['coherencia']['score'] * 0.4,              # 40%
            verificaciones['hallucination']['confianza'] * 0.2,       # 20%
            verificaciones['idioma']['confianza'] * 0.1               # 10%
        ]
        
        return sum(scores)
```

### Presentación de Resultados al Usuario

```python
def presentar_respuesta(resultado_verificado):
    """
    Formatea respuesta verificada para mostrar al usuario
    """
    verificacion = resultado_verificado['verificaciones']
    score = resultado_verificado['score_confianza']
    
    # Determinar nivel de confianza visual
    if score >= 0.8:
        badge_confianza = "✓ ALTA CONFIANZA"
        color = "verde"
    elif score >= 0.6:
        badge_confianza = "⚠ CONFIANZA MEDIA"
        color = "amarillo"
    else:
        badge_confianza = "⚠️ BAJA CONFIANZA - REVISAR"
        color = "rojo"
    
    # Formatear salida
    output = f"""
{'='*70}
PREGUNTA: {resultado_verificado['pregunta']}
{'='*70}

RESPUESTA:
{resultado_verificado['respuesta']}

{'='*70}
VERIFICACIÓN: {badge_confianza}
{'='*70}

📊 Score de Confianza: {score:.2%}

Citaciones:
  - Encontradas: {verificacion['citaciones']['citaciones_encontradas']}
  - Verificadas: {verificacion['citaciones']['citaciones_verificadas']}
  - Tasa de verificación: {verificacion['citaciones']['tasa_verificacion']:.0%}

Coherencia: {verificacion['coherencia']['nivel']}
{verificacion['coherencia']['explicacion']}

Idioma: {'✓ Español' if verificacion['idioma']['es_español'] else '✗ No español'}

{'='*70}
FUENTES CONSULTADAS:
{'='*70}
"""
    
    for i, fuente in enumerate(resultado_verificado['fuentes'], 1):
        meta = fuente['metadata']
        output += f"""
[{i}] {meta.get('documento', 'Desconocido')} - {meta.get('articulo', '')}
    {fuente['texto'][:200]}...
"""
    
    return output

# Uso
resultado = {
    'pregunta': '¿Cuál es el ratio mínimo CET1?',
    'respuesta': 'El ratio mínimo de CET1 es 4.5% [CRR Art. 92]...',
    'fuentes': [...],
    'verificaciones': {...},
    'score_confianza': 0.85
}

print(presentar_respuesta(resultado))
```

---

## Configuración Técnica

### Instalación de Dependencias

```bash
# requirements.txt

# Transformers y entrenamiento
transformers>=4.38.0
torch>=2.2.0
accelerate>=0.27.0
peft>=0.8.0
bitsandbytes>=0.42.0
trl>=0.7.0

# Datasets y procesamiento
datasets>=2.16.0
pandas>=2.0.0
numpy>=1.24.0

# RAG y embeddings
sentence-transformers>=2.3.0
chromadb>=0.4.22
langchain>=0.1.0

# Búsqueda
rank-bm25>=0.2.2

# Utilidades
langdetect>=1.0.9
pypdf2>=3.0.0
beautifulsoup4>=4.12.0
tqdm>=4.66.0

# Monitoreo (opcional)
wandb>=0.16.0

# API (opcional)
fastapi>=0.109.0
uvicorn>=0.27.0
```

### Script de Instalación

```bash
#!/bin/bash
# install.sh

echo "🚀 Instalando dependencias para asistente regulatorio..."

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip

# Instalar PyTorch con CUDA 12.1 (ajustar según tu versión)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalar dependencias
pip install -r requirements.txt

# Descargar modelo de embeddings
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')"

echo "✓ Instalación completada"
echo ""
echo "Para activar el entorno: source venv/bin/activate"
```

### Configuración de CUDA y Memoria

```python
# config.py

import torch
import os

# Configuración de GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Usar GPU 0

# Optimizaciones de memoria
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def verificar_gpu():
    """
    Verifica disponibilidad y memoria de GPU
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memoria_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memoria_libre = (torch.cuda.get_device_properties(0).total_memory - 
                        torch.cuda.memory_allocated(0)) / 1024**3
        
        print(f"✓ GPU detectada: {gpu_name}")
        print(f"  Memoria total: {memoria_total:.2f} GB")
        print(f"  Memoria libre: {memoria_libre:.2f} GB")
        
        if memoria_total < 14:
            print("⚠️ ADVERTENCIA: Menos de 16GB VRAM. Considera reducir batch size.")
        
        return True
    else:
        print("✗ No se detectó GPU. El entrenamiento será muy lento.")
        return False

# Verificar al importar
verificar_gpu()
```

### Estructura de Directorios del Proyecto

```
regulatory-assistant/
│
├── data/
│   ├── raw/                    # Documentos PDF/TXT originales
│   │   ├── CRR_español.pdf
│   │   ├── EBA_guidelines.pdf
│   │   └── ...
│   │
│   ├── processed/              # Documentos procesados
│   │   ├── chunks/
│   │   └── cleaned/
│   │
│   └── training/               # Datasets de entrenamiento
│       ├── training_data.jsonl
│       ├── validation_data.jsonl
│       └── test_data.jsonl
│
├── models/
│   ├── base/                   # Modelos base descargados
│   ├── finetuned/             # Modelos fine-tuneados
│   │   ├── checkpoint-100/
│   │   ├── checkpoint-200/
│   │   └── final/
│   └── embeddings/            # Modelos de embeddings
│
├── vector_db/
│   └── chroma_db/             # Base de datos vectorial
│
├── scripts/
│   ├── 01_process_documents.py
│   ├── 02_create_training_data.py
│   ├── 03_finetune_model.py
│   ├── 04_build_vector_db.py
│   └── 05_evaluate_model.py
│
├── src/
│   ├── __init__.py
│   ├── rag_system.py          # Sistema RAG
│   ├── verification.py        # Sistema de verificación
│   ├── utils.py               # Utilidades
│   └── config.py              # Configuración
│
├── tests/
│   ├── test_rag.py
│   ├── test_verification.py
│   └── test_cases.json        # Casos de prueba
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_evaluation.ipynb
│
├── api/                       # API opcional
│   ├── main.py
│   └── schemas.py
│
├── requirements.txt
├── install.sh
├── README.md
└── IMPLEMENTATION_DETAIL.md   # Este documento
```

---

## Testing y Validación

### Suite de Tests

```python
# tests/test_cases.json

{
  "test_cases": [
    {
      "id": "TC001",
      "categoria": "definiciones",
      "pregunta": "¿Qué es el capital CET1?",
      "respuesta_esperada_contiene": [
        "Common Equity Tier 1",
        "capital de nivel 1 ordinario",
        "acciones ordinarias"
      ],
      "fuente_esperada": "CRR",
      "dificultad": "facil"
    },
    {
      "id": "TC002",
      "categoria": "calculos",
      "pregunta": "¿Cómo se calcula el ratio de apalancamiento según Basilea III?",
      "respuesta_esperada_contiene": [
        "capital de nivel 1",
        "exposición total",
        "3%"
      ],
      "fuente_esperada": "CRR Art. 429",
      "dificultad": "media"
    },
    {
      "id": "TC003",
      "categoria": "procedimientos",
      "pregunta": "¿Qué metodologías están permitidas para la estimación de PD bajo IRB?",
      "respuesta_esperada_contiene": [
        "through-the-cycle",
        "point-in-time",
        "mínimo un año"
      ],
      "fuente_esperada": "Directrices EBA",
      "dificultad": "dificil"
    }
  ]
}
```

### Script de Evaluación Automática

```python
# tests/test_rag.py

import json
import pytest
from src.rag_system import RegulatoryRAGSystem
from src.verification import SistemaVerificacion

class TestRAGSystem:
    """
    Tests del sistema RAG completo
    """
    
    @classmethod
    def setup_class(cls):
        """Setup ejecutado una vez antes de todos los tests"""
        cls.rag = RegulatoryRAGSystem()
        cls.verificador = SistemaVerificacion(model, tokenizer)
        
        with open('tests/test_cases.json', 'r') as f:
            cls.test_cases = json.load(f)['test_cases']
    
    def test_respuestas_en_español(self):
        """Verifica que todas las respuestas estén en español"""
        import langdetect
        
        for caso in self.test_cases[:10]:  # Primeros 10 casos
            resultado = self.rag.responder_pregunta(
                caso['pregunta'],
                model,
                tokenizer
            )
            
            idioma = langdetect.detect(resultado['respuesta'])
            assert idioma == 'es', f"Respuesta no en español: {caso['id']}"
    
    def test_citaciones_presentes(self):
        """Verifica que las respuestas incluyan citaciones"""
        import re
        
        for caso in self.test_cases:
            resultado = self.rag.responder_pregunta(caso['pregunta'], model, tokenizer)
            
            # Buscar citaciones [...]
            citaciones = re.findall(r'\[[^\]]+\]', resultado['respuesta'])
            
            assert len(citaciones) > 0, f"Sin citaciones: {caso['id']}"
    
    def test_contenido_esperado(self):
        """Verifica que las respuestas contengan términos clave"""
        resultados = []
        
        for caso in self.test_cases:
            resultado = self.rag.responder_pregunta(caso['pregunta'], model, tokenizer)
            respuesta = resultado['respuesta'].lower()
            
            # Verificar presencia de términos esperados
            terminos_encontrados = [
                term for term in caso['respuesta_esperada_contiene']
                if term.lower() in respuesta
            ]
            
            tasa_acierto = len(terminos_encontrados) / len(caso['respuesta_esperada_contiene'])
            
            resultados.append({
                'id': caso['id'],
                'tasa_acierto': tasa_acierto,
                'aprobado': tasa_acierto >= 0.7
            })
        
        # Al menos 80% de casos deben pasar
        tasa_aprobacion = sum(r['aprobado'] for r in resultados) / len(resultados)
        
        assert tasa_aprobacion >= 0.8, f"Tasa de aprobación: {tasa_aprobacion:.1%}"
    
    def test_verificacion_citaciones(self):
        """Verifica que las citaciones sean válidas"""
        for caso in self.test_cases[:5]:
            resultado = self.rag.responder_pregunta(caso['pregunta'], model, tokenizer)
            
            verificacion = self.verificador.verificar_respuesta(
                caso['pregunta'],
                resultado['respuesta'],
                resultado['fuentes']
            )
            
            # Al menos 70% de citaciones deben estar verificadas
            assert verificacion['verificaciones']['citaciones']['tasa_verificacion'] >= 0.7

# Ejecutar tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### Métricas de Evaluación

```python
# scripts/05_evaluate_model.py

import json
import pandas as pd
from tqdm import tqdm

def evaluar_modelo_completo(rag_system, test_cases):
    """
    Evaluación completa del modelo con múltiples métricas
    """
    
    resultados = []
    
    for caso in tqdm(test_cases, desc="Evaluando casos"):
        # Generar respuesta
        resultado = rag_system.responder_pregunta(
            caso['pregunta'],
            model,
            tokenizer
        )
        
        # Calcular métricas
        metricas = {
            'id': caso['id'],
            'categoria': caso['categoria'],
            'dificultad': caso['dificultad'],
            
            # Precisión de contenido
            'terminos_presentes': calcular_precision_terminos(
                resultado['respuesta'],
                caso['respuesta_esperada_contiene']
            ),
            
            # Calidad de citaciones
            'num_citaciones': contar_citaciones(resultado['respuesta']),
            'citaciones_verificadas': resultado['verificacion']['citaciones']['tasa_verificacion'],
            
            # Coherencia
            'score_coherencia': resultado['verificacion']['coherencia']['score'],
            
            # Confianza general
            'score_confianza': resultado['verificacion']['score_confianza'],
            
            # Idioma
            'es_español': resultado['verificacion']['idioma']['es_español']
        }
        
        resultados.append(metricas)
    
    # Convertir a DataFrame para análisis
    df = pd.DataFrame(resultados)
    
    # Generar reporte
    reporte = generar_reporte(df)
    
    return df, reporte

def calcular_precision_terminos(respuesta, terminos_esperados):
    """Calcula qué % de términos esperados están presentes"""
    respuesta_lower = respuesta.lower()
    encontrados = sum(1 for t in terminos_esperados if t.lower() in respuesta_lower)
    return encontrados / len(terminos_esperados) if terminos_esperados else 0

def contar_citaciones(respuesta):
    """Cuenta número de citaciones en respuesta"""
    import re
    return len(re.findall(r'\[[^\]]+\]', respuesta))

def generar_reporte(df):
    """
    Genera reporte de evaluación
    """
    reporte = f"""
{'='*70}
REPORTE DE EVALUACIÓN - ASISTENTE REGULATORIO
{'='*70}

RESUMEN GENERAL:
----------------
Total de casos evaluados: {len(df)}
Idioma español: {df['es_español'].sum()}/{len(df)} ({df['es_español'].mean():.1%})

MÉTRICAS PROMEDIO:
------------------
Precisión de términos: {df['terminos_presentes'].mean():.1%}
Citaciones por respuesta: {df['num_citaciones'].mean():.1f}
Tasa de citaciones verificadas: {df['citaciones_verificadas'].mean():.1%}
Score de coherencia: {df['score_coherencia'].mean():.2f}
Score de confianza: {df['score_confianza'].mean():.2f}

POR CATEGORÍA:
--------------
"""
    
    for categoria in df['categoria'].unique():
        df_cat = df[df['categoria'] == categoria]
        reporte += f"\n{categoria.upper()}:"
        reporte += f"\n  - Casos: {len(df_cat)}"
        reporte += f"\n  - Score medio: {df_cat['score_confianza'].mean():.2f}"
        reporte += f"\n  - Precisión: {df_cat['terminos_presentes'].mean():.1%}\n"
    
    reporte += f"""
POR DIFICULTAD:
---------------
"""
    for dif in ['facil', 'media', 'dificil']:
        df_dif = df[df['dificultad'] == dif]
        if len(df_dif) > 0:
            reporte += f"\n{dif.upper()}:"
            reporte += f"\n  - Casos: {len(df_dif)}"
            reporte += f"\n  - Score medio: {df_dif['score_confianza'].mean():.2f}\n"
    
    reporte += f"""
{'='*70}
CASOS CON BAJA CONFIANZA (< 0.6):
{'='*70}
"""
    
    bajos = df[df['score_confianza'] < 0.6]
    for _, caso in bajos.iterrows():
        reporte += f"\n{caso['id']} - {caso['categoria']} - Score: {caso['score_confianza']:.2f}"
    
    return reporte

# Ejecutar evaluación
if __name__ == '__main__':
    with open('tests/test_cases.json') as f:
        test_cases = json.load(f)['test_cases']
    
    df_resultados, reporte = evaluar_modelo_completo(rag_system, test_cases)
    
    print(reporte)
    
    # Guardar resultados
    df_resultados.to_csv('evaluation_results.csv', index=False)
    
    with open('evaluation_report.txt', 'w') as f:
        f.write(reporte)
```

---

## Despliegue

### API REST con FastAPI

```python
# api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="Asistente Regulatorio Bancario",
    description="API para consultas sobre regulación bancaria en español",
    version="1.0.0"
)

# Modelos de datos
class PreguntaRequest(BaseModel):
    pregunta: str
    n_fuentes: Optional[int] = 5
    
class FuenteResponse(BaseModel):
    documento: str
    articulo: str
    texto: str
    relevancia: float

class RespuestaResponse(BaseModel):
    pregunta: str
    respuesta: str
    fuentes: List[FuenteResponse]
    score_confianza: float
    nivel_confianza: str
    advertencias: List[str]

# Inicializar sistema (hacer esto una sola vez)
@app.on_event("startup")
async def startup_event():
    global rag_system, verificador
    
    print("🚀 Inicializando sistema RAG...")
    rag_system = RegulatoryRAGSystem()
    
    print("🔧 Cargando modelo fine-tuneado...")
    # Cargar modelo aquí
    
    print("✓ Sistema listo")

# Endpoint principal
@app.post("/consultar", response_model=RespuestaResponse)
async def consultar(request: PreguntaRequest):
    """
    Responde una pregunta sobre regulación bancaria
    """
    try:
        # Generar respuesta
        resultado = rag_system.responder_pregunta(
            request.pregunta,
            model,
            tokenizer
        )
        
        # Formatear fuentes
        fuentes = [
            FuenteResponse(
                documento=f['metadata'].get('documento', 'Desconocido'),
                articulo=f['metadata'].get('articulo', ''),
                texto=f['texto'][:300],
                relevancia=1 - f['distancia'] if f['distancia'] else 0.5
            )
            for f in resultado['fuentes'][:request.n_fuentes]
        ]
        
        # Determinar nivel de confianza
        score = resultado['verificacion']['score_confianza']
        if score >= 0.8:
            nivel = "ALTA"
        elif score >= 0.6:
            nivel = "MEDIA"
        else:
            nivel = "BAJA"
        
        # Generar advertencias
        advertencias = []
        if score < 0.7:
            advertencias.append("Confianza por debajo del umbral recomendado. Verificar respuesta.")
        if not resultado['verificacion']['idioma']['es_español']:
            advertencias.append("ADVERTENCIA: Respuesta no detectada como español.")
        
        return RespuestaResponse(
            pregunta=request.pregunta,
            respuesta=resultado['respuesta'],
            fuentes=fuentes,
            score_confianza=score,
            nivel_confianza=nivel,
            advertencias=advertencias
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint de salud
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "modelo": "operativo",
        "vector_db": "conectada"
    }

# Endpoint de estadísticas
@app.get("/stats")
async def get_stats():
    return {
        "total_documentos": rag_system.collection.count(),
        "modelo": model_name,
        "idioma": "español"
    }

# Ejecutar servidor
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Cliente de Línea de Comandos

```python
# cli.py

import argparse
import sys
from src.rag_system import RegulatoryRAGSystem

def main():
    parser = argparse.ArgumentParser(
        description="Asistente de Consultoría Regulatoria en Español"
    )
    
    parser.add_argument(
        'pregunta',
        type=str,
        nargs='?',
        help='Pregunta sobre regulación bancaria'
    )
    
    parser.add_argument(
        '--interactive',
        '-i',
        action='store_true',
        help='Modo interactivo'
    )
    
    parser.add_argument(
        '--fuentes',
        '-f',
        type=int,
        default=5,
        help='Número de fuentes a recuperar (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Inicializar sistema
    print("🔧 Inicializando sistema RAG...")
    rag = RegulatoryRAGSystem()
    print("✓ Sistema listo\n")
    
    if args.interactive:
        # Modo interactivo
        print("Modo interactivo activado. Escribe 'salir' para terminar.\n")
        
        while True:
            pregunta = input("📋 Pregunta: ").strip()
            
            if pregunta.lower() in ['salir', 'exit', 'quit']:
                print("¡Hasta luego!")
                break
            
            if not pregunta:
                continue
            
            # Procesar pregunta
            procesar_y_mostrar(rag, pregunta, args.fuentes)
            print("\n" + "="*70 + "\n")
    
    elif args.pregunta:
        # Modo single query
        procesar_y_mostrar(rag, args.pregunta, args.fuentes)
    
    else:
        parser.print_help()

def procesar_y_mostrar(rag, pregunta, n_fuentes):
    """Procesa pregunta y muestra resultado formateado"""
    
    print(f"\n🔍 Buscando respuesta...\n")
    
    resultado = rag.responder_pregunta(pregunta, model, tokenizer)
    
    # Mostrar respuesta
    print("="*70)
    print("RESPUESTA:")
    print("="*70)
    print(resultado['respuesta'])
    print()
    
    # Mostrar verificación
    verif = resultado['verificacion']
    score = verif['score_confianza']
    
    if score >= 0.8:
        badge = "✓ ALTA CONFIANZA"
    elif score >= 0.6:
        badge = "⚠ CONFIANZA MEDIA"
    else:
        badge = "⚠️ BAJA CONFIANZA"
    
    print("="*70)
    print(f"VERIFICACIÓN: {badge}")
    print("="*70)
    print(f"Score: {score:.1%}")
    print(f"Citaciones verificadas: {verif['citaciones']['tasa_verificacion']:.0%}")
    print(f"Coherencia: {verif['coherencia']['nivel']}")
    print()
    
    # Mostrar fuentes
    print("="*70)
    print("FUENTES CONSULTADAS:")
    print("="*70)
    
    for i, fuente in enumerate(resultado['fuentes'][:n_fuentes], 1):
        meta = fuente['metadata']
        print(f"\n[{i}] {meta.get('documento', 'Desconocido')} - {meta.get('articulo', '')}")
        print(f"    {fuente['texto'][:200]}...")

if __name__ == '__main__':
    main()
```

### Interfaz Web Simple con Gradio

```python
# app_gradio.py

import gradio as gr
from src.rag_system import RegulatoryRAGSystem

# Inicializar sistema
rag_system = RegulatoryRAGSystem()

def responder_consulta(pregunta, num_fuentes):
    """
    Función llamada por Gradio
    """
    if not pregunta.strip():
        return "Por favor, ingresa una pregunta.", "", ""
    
    # Generar respuesta
    resultado = rag_system.responder_pregunta(
        pregunta,
        model,
        tokenizer
    )
    
    # Formatear respuesta
    respuesta = resultado['respuesta']
    
    # Formatear verificación
    verif = resultado['verificacion']
    score = verif['score_confianza']
    
    verificacion = f"""
**Score de Confianza:** {score:.0%}

**Citaciones:**
- Encontradas: {verif['citaciones']['citaciones_encontradas']}
- Verificadas: {verif['citaciones']['citaciones_verificadas']}

**Coherencia:** {verif['coherencia']['nivel']}

**Idioma:** {'✓ Español' if verif['idioma']['es_español'] else '✗ No español'}
"""
    
    # Formatear fuentes
    fuentes_texto = ""
    for i, fuente in enumerate(resultado['fuentes'][:num_fuentes], 1):
        meta = fuente['metadata']
        fuentes_texto += f"""
**[{i}] {meta.get('documento', 'Desconocido')} - {meta.get('articulo', '')}**

{fuente['texto'][:300]}...

---
"""
    
    return respuesta, verificacion, fuentes_texto

# Crear interfaz
with gr.Blocks(title="Asistente Regulatorio Bancario") as demo:
    gr.Markdown("""
    # 🏦 Asistente de Regulación Bancaria
    
    Sistema especializado en Basilea III, parámetros de riesgo de crédito y normativa prudencial.
    **Todas las respuestas son en español.**
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            pregunta_input = gr.Textbox(
                label="Tu pregunta sobre regulación bancaria",
                placeholder="Ej: ¿Cuál es el ratio mínimo de capital CET1?",
                lines=3
            )
            
            num_fuentes = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Número de fuentes a mostrar"
            )
            
            submit_btn = gr.Button("Consultar", variant="primary")
    
    with gr.Row():
        with gr.Column():
            respuesta_output = gr.Markdown(label="Respuesta")
    
    with gr.Row():
        with gr.Column():
            verificacion_output = gr.Markdown(label="Verificación")
        
        with gr.Column():
            fuentes_output = gr.Markdown(label="Fuentes")
    
    # Ejemplos
    gr.Examples(
        examples=[
            ["¿Qué es el capital CET1?"],
            ["¿Cómo se calcula el ratio de apalancamiento?"],
            ["¿Qué metodologías existen para estimar la PD?"],
            ["¿Cuál es el tratamiento de las exposiciones hipotecarias?"]
        ],
        inputs=pregunta_input
    )
    
    # Conectar evento
    submit_btn.click(
        fn=responder_consulta,
        inputs=[pregunta_input, num_fuentes],
        outputs=[respuesta_output, verificacion_output, fuentes_output]
    )

# Lanzar aplicación
if __name__ == "__main__":
    demo.launch(share=True)
```

---

## Próximos Pasos y Mejoras

### Roadmap de Desarrollo

**Fase 1: MVP (2-4 semanas)**
- [ ] Procesar documentos regulatorios clave
- [ ] Crear dataset inicial (500 ejemplos)
- [ ] Fine-tune modelo base
- [ ] Implementar RAG básico
- [ ] Testing con casos de uso principales

**Fase 2: Refinamiento (4-6 semanas)**
- [ ] Ampliar dataset a 1000+ ejemplos
- [ ] Implementar sistema de verificación completo
- [ ] Optimizar búsqueda híbrida
- [ ] Crear suite de tests comprehensiva
- [ ] Desarrollar CLI y API básica

**Fase 3: Producción (6-8 semanas)**
- [ ] Interfaz web con Gradio/Streamlit
- [ ] Monitoreo y logging
- [ ] Gestión de versiones de documentos
- [ ] Integración con flujos de trabajo existentes
- [ ] Documentación completa para usuarios

**Fase 4: Mejoras Avanzadas (ongoing)**
- [ ] Fine-tuning continuo con feedback de usuarios
- [ ] Soporte multimodal (gráficos, tablas en PDFs)
- [ ] Comparación automática entre jurisdicciones
- [ ] Alertas de cambios regulatorios
- [ ] Integración con sistemas de gestión de riesgos

### Posibles Mejoras Técnicas

1. **Re-ranking avanzado**: Usar modelos cross-encoder para re-rankear resultados
2. **Decomposición de preguntas**: Dividir preguntas complejas en sub-preguntas
3. **Graph RAG**: Modelar relaciones entre conceptos regulatorios
4. **Actualización incremental**: Sistema para incorporar nuevas regulaciones sin reentrenar
5. **Explicabilidad**: Generar explicaciones de por qué se recuperaron ciertas fuentes

---

## Conclusión

Este documento proporciona una guía completa para implementar un asistente de consultoría regulatoria especializado en español. Los componentes clave son:

1. **Modelo 7B fine-tuneado** en español con LoRA
2. **Sistema RAG** con búsqueda híbrida y vector DB
3. **Auto-verificación** de citaciones y coherencia
4. **Pipeline completo** desde datos hasta despliegue

**Expectativas realistas:**
- ✓ 70-80% de consultas rutinarias respondidas correctamente
- ✓ Respuestas siempre en español con citaciones verificables
- ✓ Reducción significativa en tiempo de búsqueda de información
- ⚠️ Revisión humana necesaria para decisiones críticas
- ⚠️ No reemplaza expertise regulatorio, sino que lo complementa

**Tiempo estimado de implementación:** 8-12 semanas para MVP funcional

---

## Referencias y Recursos

### Documentación Técnica
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- PEFT (LoRA): https://huggingface.co/docs/peft
- LangChain: https://python.langchain.com/docs/get_started/introduction
- ChromaDB: https://docs.trychroma.com/

### Papers Relevantes
- LoRA: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- RAG: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- RAFT: "RAFT: Adapting Language Model to Domain Specific RAG" (Zhang et al., 2024)

### Datasets y Recursos en Español
- Corpus legal español: https://www.boe.es/
- EBA en español: https://www.eba.europa.eu/languages/home_es
- Banco de España: https://www.bde.es/

---

**Última actualización:** Enero 2026
**Versión:** 1.0
**Autor:** Implementación para consultoría regulatoria bancaria

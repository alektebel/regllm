import anthropic
import json
from tqdm import tqdm

client = anthropic.Anthropic(api_key="your-key")

def generate_qa_from_regulation(article_text, metadata, num_questions=3):
    """
    Generate Q&A pairs from a regulatory article
    """

    prompt = f"""Eres un experto en regulación bancaria. A partir del siguiente fragmento regulatorio, genera {num_questions} pares de pregunta-respuesta en español de alta calidad.

FRAGMENTO REGULATORIO:
{article_text}

METADATA:
- Documento: {metadata['documento']}
- Artículo: {metadata['articulo']}

REQUISITOS CRÍTICOS:
1. Preguntas naturales que haría un consultor bancario
2. Respuestas deben citar específicamente el artículo/párrafo
3. Incluir cita textual relevante entre comillas
4. Variedad: una pregunta de definición, una de aplicación, una de cálculo/procedimiento
5. TODO en español profesional y técnico
6. Formato JSON exacto

FORMATO DE SALIDA:
{{
  "qa_pairs": [
    {{
      "pregunta": "pregunta natural del usuario",
      "respuesta": "respuesta completa con citaciones [Art. X] y explicación",
      "cita_textual": "fragmento exacto del texto regulatorio",
      "nivel_dificultad": "facil|medio|dificil"
    }}
  ]
}}

SOLO devuelve el JSON, sin texto adicional."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse JSON response
    try:
        result = json.loads(response.content[0].text)
        return result['qa_pairs']
    except:
        print(f"Error parsing response for {metadata['articulo']}")
        return []

# Example: Process your regulatory documents
regulatory_articles = [
    {
        'text': """Artículo 92: Requisitos de fondos propios
        1. Las entidades deberán cumplir en todo momento los siguientes requisitos de fondos propios:
        (a) un ratio de capital de nivel 1 ordinario del 4,5 %;
        (b) un ratio de capital de nivel 1 del 6 %;
        (c) un ratio de fondos propios totales del 8 %.""",
        'metadata': {
            'documento': 'Reglamento CRR (UE) 575/2013',
            'articulo': 'Artículo 92'
        }
    },
    # ... more articles
]

all_qa_pairs = []

for article in tqdm(regulatory_articles, desc="Generating Q&A pairs"):
    qa_pairs = generate_qa_from_regulation(
        article['text'],
        article['metadata'],
        num_questions=3
    )

    # Add metadata to each pair
    for qa in qa_pairs:
        qa['source_document'] = article['metadata']['documento']
        qa['source_article'] = article['metadata']['articulo']

    all_qa_pairs.extend(qa_pairs)

    # Save incrementally
    with open('synthetic_qa.jsonl', 'a', encoding='utf-8') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')

print(f"✓ Generated {len(all_qa_pairs)} Q&A pairs")

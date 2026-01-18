#!/usr/bin/env python3
"""
Clean the training dataset by removing problematic samples and regenerating
high-quality training data from raw documents.
"""

import json
import re
from pathlib import Path
from urllib.parse import unquote
import logging
from datetime import datetime
from typing import List, Dict, Optional
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decode_url_title(title: str) -> str:
    """Decode URL-encoded titles and clean them."""
    if not title:
        return ""

    # URL decode
    decoded = unquote(title)

    # Remove file extensions
    decoded = re.sub(r'\.(pdf|html|htm)$', '', decoded, flags=re.IGNORECASE)

    # Replace special patterns
    decoded = decoded.replace('_', ' ')
    decoded = decoded.replace('-', ' ')
    decoded = decoded.replace('  ', ' ')

    return decoded.strip()


def extract_readable_title(title: str, url: str = "", source: str = "") -> str:
    """Extract a human-readable title from document metadata."""
    decoded = decode_url_title(title)

    # Common patterns to clean up
    patterns_to_remove = [
        r'^BOE A \d{4} \d+\s*',  # BOE reference prefix
        r'^EBA\s*(BS|OP|GL)\s*\d{4}\s*\d+\s*',  # EBA reference prefix
        r'\s*\|.*$',  # Remove pipe and everything after
        r'^\d+\s*',  # Leading numbers
    ]

    for pattern in patterns_to_remove:
        decoded = re.sub(pattern, '', decoded, flags=re.IGNORECASE)

    # If still looks like a filename or too short, create descriptive title
    if len(decoded) < 10 or '.' in decoded:
        # Try to create from source and keywords in URL
        if 'irb' in url.lower() or 'irb' in decoded.lower():
            decoded = "enfoque basado en calificaciones internas IRB"
        elif 'lgd' in url.lower() or 'lgd' in decoded.lower():
            decoded = "pérdida en caso de impago LGD"
        elif 'pd' in url.lower() and 'dpd' not in url.lower():
            decoded = "probabilidad de impago PD"
        elif 'sme' in url.lower() or 'pyme' in url.lower():
            decoded = "factor de apoyo a PYMES"
        elif 'dpd' in url.lower() or '180' in decoded:
            decoded = "criterio de días de mora 180 DPD"
        elif 'capital' in url.lower() or 'crr' in url.lower():
            decoded = "requisitos de capital"
        elif 'supervisory' in url.lower() or 'ssm' in url.lower():
            decoded = "manual de supervisión del MUS"
        elif 'transparencia' in decoded.lower():
            decoded = "ley de transparencia"
        else:
            decoded = f"regulación de {source}" if source else "regulación bancaria"

    return decoded.strip()


def is_spanish_text(text: str) -> bool:
    """Check if text is primarily in Spanish using unambiguous indicators."""
    if not text or len(text) < 50:
        return False

    # Check first 2000 chars for language detection
    sample = text[:2000].lower()

    # UNAMBIGUOUS Spanish-specific words (NOT found in English)
    # These words only exist in Spanish and have no English equivalent
    spanish_only_words = [
        # Spanish articles and prepositions (unique conjugations)
        'según', 'además', 'también', 'mediante', 'respecto',
        # Spanish verbs (conjugated forms not in English)
        'será', 'serán', 'debe', 'deben', 'puede', 'pueden',
        'siendo', 'sido', 'están', 'estará', 'han', 'hemos',
        # Spanish words with accents (definitively Spanish)
        'artículo', 'crédito', 'información', 'regulación',
        'supervisión', 'definición', 'decisión', 'participación',
        # Common Spanish function words not in English
        'entidades', 'así', 'índice', 'áreas', 'través',
        # Spanish verb endings (-ción, -ado, -ido)
        'normativa', 'actividad', 'solvencia', 'actuación',
    ]

    # English-specific words (common but NOT Spanish)
    english_only_words = [
        'the', 'and', 'for', 'that', 'with', 'this', 'from', 'which',
        'shall', 'should', 'must', 'will', 'would', 'could', 'have',
        'been', 'were', 'being', 'their', 'they', 'them', 'those',
        'these', 'where', 'when', 'what', 'pursuant', 'regarding',
        'institution', 'institutions', 'framework', 'approach',
        'implementation', 'requirements', 'assessment', 'methodology',
        'provides', 'provides', 'listed', 'below', 'tool'
    ]

    spanish_count = sum(1 for w in spanish_only_words if w in sample)
    english_count = sum(1 for w in english_only_words if w in sample)

    # Require strong Spanish presence: at least 2 unambiguous Spanish words
    # AND more Spanish words than English words
    if spanish_count >= 2 and spanish_count > english_count:
        return True

    # If very strong Spanish presence (4+ words), accept even with some English
    if spanish_count >= 4:
        return True

    return False


def is_valid_question(question: str) -> bool:
    """Check if the question is well-formed and conceptual."""
    question_lower = question.lower()

    # Reject questions about filenames
    if '.pdf' in question_lower or '%20' in question_lower:
        return False

    # Reject questions with URL-encoded content
    if '%' in question and re.search(r'%[0-9a-fA-F]{2}', question):
        return False

    # Reject questions in English
    if question.startswith(('What ', 'How ', 'Which ', 'When ', 'Where ', 'Why ')):
        return False

    # Reject questions that are just about document titles
    if re.match(r'^¿Qué dice la regulación sobre [^?]+\?$', question):
        # Check if it's asking about a filename/title rather than a concept
        subject = re.search(r'sobre (.+)\?', question)
        if subject:
            subject_text = subject.group(1)
            # If subject looks like a filename or URL component, reject
            if '.' in subject_text or '|' in subject_text:
                return False

    return True


def clean_response_text(text: str) -> str:
    """Clean and improve response text."""
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Fix common PDF extraction issues
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Fix hyphenation
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)   # Fix split numbers

    return text.strip()


def generate_conceptual_questions(keywords: List[str], source: str) -> List[str]:
    """Generate conceptual questions in Spanish based on keywords."""
    questions = []

    # Spanish-only question templates
    templates = [
        "¿Cuáles son los requisitos regulatorios para {keyword}?",
        "¿Cómo se debe calcular {keyword} según la normativa?",
        "¿Qué establece la regulación sobre {keyword}?",
        "¿Cuál es la metodología para estimar {keyword}?",
        "Explica los requisitos de {keyword} según la regulación bancaria",
        "¿Qué criterios aplican para {keyword}?",
        "¿Cuáles son las directrices sobre {keyword}?",
    ]

    # Keyword-specific question mappings
    keyword_questions = {
        'pd': [
            "¿Cómo se calcula la probabilidad de impago (PD)?",
            "¿Cuáles son los requisitos para estimar la PD bajo el enfoque IRB?",
            "¿Qué metodología se usa para calibrar la probabilidad de impago?",
        ],
        'lgd': [
            "¿Cómo se estima la pérdida en caso de impago (LGD)?",
            "¿Cuáles son los requisitos regulatorios para el cálculo de LGD?",
            "¿Qué factores afectan la estimación de la severidad?",
        ],
        'ead': [
            "¿Cómo se calcula la exposición en caso de impago (EAD)?",
            "¿Qué requisitos establece la normativa para estimar EAD?",
        ],
        'irb': [
            "¿Cuáles son los requisitos del enfoque basado en calificaciones internas?",
            "¿Qué condiciones debe cumplir una entidad para usar el método IRB?",
            "¿Cómo funciona el enfoque IRB para riesgo de crédito?",
        ],
        'capital': [
            "¿Cuáles son los requisitos mínimos de capital según Basilea III?",
            "¿Cómo se calculan los activos ponderados por riesgo?",
            "¿Qué establece la normativa sobre ratios de capital?",
        ],
        'crr': [
            "¿Qué establece el Reglamento de Requisitos de Capital (CRR)?",
            "¿Cuáles son los principales requisitos del CRR?",
        ],
        'crd': [
            "¿Qué establece la Directiva de Requisitos de Capital (CRD)?",
        ],
        'pyme': [
            "¿Qué es el factor de apoyo a PYMES?",
            "¿Cómo se aplica el tratamiento preferencial para exposiciones a PYMES?",
        ],
        'sme': [
            "¿Qué es el factor de apoyo a PYMES?",
            "¿Cómo se aplica el tratamiento preferencial para exposiciones a PYMES?",
        ],
        'default': [
            "¿Cuál es la definición de impago según la normativa?",
            "¿Qué criterios determinan que un deudor está en situación de impago?",
        ],
        'mora': [
            "¿Cuántos días de mora se consideran para definir el impago?",
            "¿Qué establece la regulación sobre el criterio de días de mora?",
        ],
        'provisión': [
            "¿Cómo se calculan las provisiones por riesgo de crédito?",
            "¿Qué requisitos establece la normativa para las provisiones?",
        ],
        'riesgo de crédito': [
            "¿Cuáles son los principales componentes del riesgo de crédito?",
            "¿Qué metodologías existen para medir el riesgo de crédito?",
        ],
    }

    for keyword in keywords:
        keyword_lower = keyword.lower().strip()

        # Use specific questions if available
        if keyword_lower in keyword_questions:
            questions.extend(keyword_questions[keyword_lower])
        else:
            # Generate from templates
            for template in random.sample(templates, min(2, len(templates))):
                questions.append(template.format(keyword=keyword))

    return list(set(questions))  # Remove duplicates


def process_raw_document(doc: Dict) -> List[Dict]:
    """Process a raw document and generate clean QA pairs."""
    qa_pairs = []

    text = doc.get('text', '')
    source = doc.get('source', 'Unknown')
    url = doc.get('url', '')
    title = doc.get('title', '')
    keywords = doc.get('keywords', [])

    # Skip if text is too short
    if len(text) < 200:
        return []

    # Clean the title
    readable_title = extract_readable_title(title, url, source)

    # Clean the text
    cleaned_text = clean_response_text(text)

    # Skip transparency law (not banking regulation)
    if 'transparencia' in cleaned_text.lower() and 'buen gobierno' in cleaned_text.lower():
        if 'ley 19/2013' in cleaned_text.lower() or 'boe-a-2013-12887' in url.lower():
            logger.debug(f"Skipping transparency law document: {title}")
            return []

    # Create a focused excerpt that will be used as the answer
    excerpt = cleaned_text[:1500]
    if len(cleaned_text) > 1500:
        last_period = excerpt.rfind('.')
        if last_period > 1000:
            excerpt = excerpt[:last_period + 1]

    # Check if the excerpt (answer) is in Spanish - this is the critical check
    if not is_spanish_text(excerpt):
        logger.debug(f"Skipping document with English content: {title}")
        return []

    # Generate conceptual questions
    questions = generate_conceptual_questions(keywords, source)

    # Create QA pairs with Spanish-only content
    for question in questions[:5]:  # Limit to 5 questions per doc
        if not is_valid_question(question):
            continue

        # Format the response with proper citation
        response = f"Según {source}:\n\n{excerpt}\n\nFuente: {readable_title} ({url})"

        qa_pairs.append({
            'question': question,
            'response': response,
            'source': source,
            'url': url,
            'title': readable_title
        })

    return qa_pairs


def format_for_training(qa_pair: Dict) -> Dict:
    """Format a QA pair in the instruction-response format."""
    system_prompt = """Eres un experto en regulación bancaria española y europea. Tu tarea es responder preguntas sobre normativa prudencial, especialmente sobre parámetros de riesgo de crédito (PD, LGD, EAD), Basilea III, CRR y directrices de la EBA.

REGLAS IMPORTANTES:
- Responde SIEMPRE en español
- Cita la fuente específica de tu información (artículo, directriz, documento)
- Si no tienes información suficiente, indícalo claramente
- Sé preciso y técnico en tus explicaciones
- Nunca inventes información regulatoria"""

    return {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': qa_pair['question']},
            {'role': 'assistant', 'content': qa_pair['response']}
        ],
        'metadata': {
            'source': qa_pair['source'],
            'url': qa_pair['url'],
            'title': qa_pair['title']
        }
    }


def clean_and_regenerate_dataset():
    """Main function to clean and regenerate the dataset."""
    raw_data_dir = Path("data/raw")
    processed_data_dir = Path("data/processed")
    backup_dir = Path("data/backups")

    # Create backup of existing data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)

    existing_train = processed_data_dir / "train_data.json"
    if existing_train.exists():
        import shutil
        backup_path = backup_dir / f"train_data_backup_{timestamp}.json"
        shutil.copy(existing_train, backup_path)
        logger.info(f"Backed up existing data to {backup_path}")

    # Load raw data
    json_files = list(raw_data_dir.glob("regulation_data_*.json"))
    if not json_files:
        logger.error("No raw data files found!")
        return

    # Use the most recent file
    raw_file = max(json_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading raw data from {raw_file}")

    with open(raw_file, 'r', encoding='utf-8') as f:
        raw_documents = json.load(f)

    logger.info(f"Loaded {len(raw_documents)} raw documents")

    # Process each document
    all_qa_pairs = []
    spanish_doc_count = 0

    for doc in raw_documents:
        qa_pairs = process_raw_document(doc)
        if qa_pairs:
            spanish_doc_count += 1
            all_qa_pairs.extend(qa_pairs)

    logger.info(f"Generated {len(all_qa_pairs)} QA pairs from {spanish_doc_count} Spanish documents")

    if not all_qa_pairs:
        logger.error("No valid QA pairs generated!")
        return

    # Format for training
    formatted_examples = [format_for_training(qa) for qa in all_qa_pairs]

    # Remove duplicates based on question
    seen_questions = set()
    unique_examples = []
    for ex in formatted_examples:
        q = ex['messages'][1]['content']
        if q not in seen_questions:
            seen_questions.add(q)
            unique_examples.append(ex)

    logger.info(f"After deduplication: {len(unique_examples)} unique examples")

    # Split train/val (85/15)
    random.shuffle(unique_examples)
    split_idx = int(len(unique_examples) * 0.85)

    train_data = unique_examples[:split_idx]
    val_data = unique_examples[split_idx:]

    logger.info(f"Split: {len(train_data)} training, {len(val_data)} validation")

    # Save processed data
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    train_file = processed_data_dir / "train_data.json"
    val_file = processed_data_dir / "val_data.json"

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved training data to {train_file}")
    logger.info(f"Saved validation data to {val_file}")

    # Create small subset for testing
    if len(train_data) >= 10:
        small_subset = random.sample(train_data, min(50, len(train_data)))
        small_file = processed_data_dir / "train_data_small.json"
        with open(small_file, 'w', encoding='utf-8') as f:
            json.dump(small_subset, f, ensure_ascii=False, indent=2)
        logger.info(f"Created small subset of {len(small_subset)} examples")

    # Print statistics
    print("\n" + "=" * 70)
    print("DATASET REGENERATION COMPLETE")
    print("=" * 70)
    print(f"\nRaw documents processed: {len(raw_documents)}")
    print(f"Spanish documents found: {spanish_doc_count}")
    print(f"Total QA pairs generated: {len(all_qa_pairs)}")
    print(f"Unique training examples: {len(unique_examples)}")
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print("=" * 70)


if __name__ == "__main__":
    clean_and_regenerate_dataset()

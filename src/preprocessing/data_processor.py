"""
Data preprocessing pipeline for banking regulation documents.
Cleans, structures, and prepares data for model finetuning.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import random
from datetime import datetime
import logging
from urllib.parse import unquote

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Preprocessor for banking regulation data."""

    def __init__(self, raw_data_dir: str = "data/raw",
                 processed_data_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self, filename: str = None) -> List[Dict]:
        """Load raw scraped data from JSON file."""
        if filename:
            file_path = self.raw_data_dir / filename
        else:
            # Get the most recent file
            json_files = list(self.raw_data_dir.glob("regulation_data_*.json"))
            if not json_files:
                raise FileNotFoundError("No regulation data files found in data/raw/")
            file_path = max(json_files, key=lambda p: p.stat().st_mtime)

        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} documents")
        return data

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep punctuation and accents
        text = re.sub(r'[^\w\s.,;:()¿?¡!áéíóúñÁÉÍÓÚÑ€%\-]', '', text)

        # Normalize line breaks
        text = text.replace('\n\n\n', '\n\n')

        # Remove very short lines (likely artifacts)
        lines = text.split('\n')
        lines = [line for line in lines if len(line.strip()) > 3]
        text = '\n'.join(lines)

        return text.strip()

    def chunk_document(self, text: str, chunk_size: int = 1000,
                       overlap: int = 200) -> List[str]:
        """Split long documents into overlapping chunks."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.split()) > 50:  # Minimum chunk size
                chunks.append(chunk)

        return chunks

    def _decode_title(self, title: str) -> str:
        """Decode URL-encoded titles and clean them."""
        if not title:
            return ""
        decoded = unquote(title)
        decoded = re.sub(r'\.(pdf|html|htm)$', '', decoded, flags=re.IGNORECASE)
        decoded = decoded.replace('_', ' ').replace('-', ' ')
        decoded = re.sub(r'\s+', ' ', decoded)
        return decoded.strip()

    def _is_spanish_text(self, text: str) -> bool:
        """Check if text is primarily in Spanish."""
        if not text or len(text) < 50:
            return False
        spanish_words = ['según', 'para', 'los', 'las', 'del', 'que', 'con',
                        'por', 'una', 'sus', 'son', 'como', 'más', 'será',
                        'artículo', 'normativa', 'regulación', 'entidades']
        english_words = ['the', 'and', 'for', 'that', 'with', 'this', 'which',
                        'shall', 'should', 'must', 'have', 'been', 'their']
        text_lower = text.lower()
        spanish_count = sum(1 for w in spanish_words if w in text_lower)
        english_count = sum(1 for w in english_words if w in text_lower)
        return spanish_count >= english_count

    def generate_qa_pairs(self, document: Dict) -> List[Dict]:
        """Generate question-answer pairs from a document."""
        qa_pairs = []
        text = document['text']
        source = document.get('source', 'Unknown')
        title = self._decode_title(document.get('title', 'Regulation Document'))
        url = document.get('url', '')
        keywords = document.get('keywords', [])

        # Skip non-Spanish documents
        if not self._is_spanish_text(text):
            logger.debug(f"Skipping non-Spanish document: {title}")
            return []

        # Create chunks for long documents
        chunks = self.chunk_document(text)

        # Spanish-only question templates (conceptual, not about filenames)
        question_templates = [
            "¿Cuáles son los requisitos regulatorios para {}?",
            "¿Cómo se calcula {} según la normativa?",
            "¿Qué establece la regulación sobre {}?",
            "¿Cuál es la metodología para estimar {}?",
            "Explica los requisitos de {} según la regulación bancaria",
            "¿Qué criterios aplican para {}?",
        ]

        # Keyword-specific conceptual questions
        keyword_questions = {
            'pd': ["¿Cómo se calcula la probabilidad de impago (PD)?",
                   "¿Cuáles son los requisitos para estimar la PD bajo el enfoque IRB?"],
            'lgd': ["¿Cómo se estima la pérdida en caso de impago (LGD)?",
                    "¿Cuáles son los requisitos regulatorios para el cálculo de LGD?"],
            'ead': ["¿Cómo se calcula la exposición en caso de impago (EAD)?"],
            'irb': ["¿Cuáles son los requisitos del enfoque basado en calificaciones internas?",
                    "¿Qué condiciones debe cumplir una entidad para usar el método IRB?"],
            'capital': ["¿Cuáles son los requisitos mínimos de capital según Basilea III?",
                       "¿Cómo se calculan los activos ponderados por riesgo?"],
            'pyme': ["¿Qué es el factor de apoyo a PYMES?"],
            'sme': ["¿Qué es el factor de apoyo a PYMES?"],
        }

        # Generate QA pairs based on keywords and document content
        used_questions = set()

        for i, chunk in enumerate(chunks[:3]):  # Limit to 3 chunks per doc
            # Generate keyword-specific questions
            for keyword in keywords[:4]:
                keyword_lower = keyword.lower().strip()

                # Get appropriate questions for this keyword
                if keyword_lower in keyword_questions:
                    questions = keyword_questions[keyword_lower]
                else:
                    questions = [random.choice(question_templates).format(keyword)]

                for question in questions:
                    if question in used_questions:
                        continue
                    used_questions.add(question)

                    answer = f"Según {source}:\n\n{chunk}\n\nFuente: {title} ({url})"

                    qa_pairs.append({
                        'instruction': question,
                        'response': answer,
                        'source': source,
                        'url': url,
                        'title': title
                    })

                    if len(qa_pairs) >= 5:  # Limit per document
                        break
                if len(qa_pairs) >= 5:
                    break
            if len(qa_pairs) >= 5:
                break

        return qa_pairs

    def create_training_examples(self, documents: List[Dict]) -> List[Dict]:
        """Create training examples from documents."""
        training_examples = []

        logger.info("Generating training examples...")
        for doc in documents:
            # Clean the document text
            doc['text'] = self.clean_text(doc['text'])

            # Skip documents that are too short
            if len(doc['text']) < 100:
                continue

            # Generate QA pairs
            qa_pairs = self.generate_qa_pairs(doc)
            training_examples.extend(qa_pairs)

        logger.info(f"Generated {len(training_examples)} training examples")
        return training_examples

    def format_for_training(self, example: Dict) -> Dict:
        """Format example in the instruction-response format for finetuning."""
        # Format with system prompt that enforces Spanish and proper citations
        system_prompt = """Eres un experto en regulación bancaria española y europea. Tu tarea es responder preguntas sobre normativa prudencial, especialmente sobre parámetros de riesgo de crédito (PD, LGD, EAD), Basilea III, CRR y directrices de la EBA.

REGLAS IMPORTANTES:
- Responde SIEMPRE en español
- Cita la fuente específica de tu información (artículo, directriz, documento)
- Si no tienes información suficiente, indícalo claramente
- Sé preciso y técnico en tus explicaciones
- Nunca inventes información regulatoria"""

        formatted = {
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': example['instruction']},
                {'role': 'assistant', 'content': example['response']}
            ],
            'metadata': {
                'source': example['source'],
                'url': example.get('url', ''),
                'title': example.get('title', '')
            }
        }

        return formatted

    def split_train_val(self, examples: List[Dict],
                       val_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict]]:
        """Split data into training and validation sets."""
        random.shuffle(examples)
        split_idx = int(len(examples) * (1 - val_ratio))

        train_data = examples[:split_idx]
        val_data = examples[split_idx:]

        logger.info(f"Split: {len(train_data)} training, {len(val_data)} validation")
        return train_data, val_data

    def save_processed_data(self, train_data: List[Dict], val_data: List[Dict]):
        """Save processed data to JSON files."""
        train_file = self.processed_data_dir / "train_data.json"
        val_file = self.processed_data_dir / "val_data.json"

        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved training data to {train_file}")
        logger.info(f"Saved validation data to {val_file}")

    def create_small_subset(self, train_data: List[Dict],
                           subset_size: int = 50) -> List[Dict]:
        """Create a small subset for overfitting test."""
        subset = random.sample(train_data, min(subset_size, len(train_data)))
        subset_file = self.processed_data_dir / "train_data_small.json"

        with open(subset_file, 'w', encoding='utf-8') as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)

        logger.info(f"Created small subset of {len(subset)} examples at {subset_file}")
        return subset

    def process(self, filename: str = None):
        """Run the complete preprocessing pipeline."""
        # Load raw data
        raw_documents = self.load_raw_data(filename)

        # Create training examples
        training_examples = self.create_training_examples(raw_documents)

        # Format for training
        formatted_examples = [
            self.format_for_training(ex) for ex in training_examples
        ]

        # Split train/val
        train_data, val_data = self.split_train_val(formatted_examples)

        # Save processed data
        self.save_processed_data(train_data, val_data)

        # Create small subset for overfitting test
        self.create_small_subset(train_data)

        logger.info("Preprocessing complete!")

        # Print statistics
        logger.info(f"\nDataset Statistics:")
        logger.info(f"Total examples: {len(formatted_examples)}")
        logger.info(f"Training examples: {len(train_data)}")
        logger.info(f"Validation examples: {len(val_data)}")

        return train_data, val_data


if __name__ == "__main__":
    processor = DataProcessor()
    processor.process()

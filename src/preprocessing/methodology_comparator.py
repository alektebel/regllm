#!/usr/bin/env python3
"""
Methodology Comparator

Generates comparison training data between banking regulation methodology documents.
Creates ChatML-formatted conversations for fine-tuning.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import random


@dataclass
class MethodologyDocument:
    """Represents a methodology document."""
    name: str
    short_name: str
    path: Path
    content: str
    sections: Dict[str, str]
    key_aspects: List[str]


class MethodologyComparator:
    """Generate comparison training data between methodology documents."""

    METHODOLOGY_DIR = Path("data/methodology")

    # Comparison question templates in Spanish
    COMPARISON_TEMPLATES = [
        "Compara {methodology_a} con {methodology_b} en términos de {aspect}.",
        "¿Cuáles son las principales diferencias entre {methodology_a} y {methodology_b}?",
        "¿Qué ventajas tiene {methodology_a} sobre {methodology_b}?",
        "¿Qué desventajas tiene {methodology_a} frente a {methodology_b}?",
        "¿En qué situaciones es preferible usar {methodology_a} en lugar de {methodology_b}?",
        "¿Cómo difiere el tratamiento de {aspect} entre {methodology_a} y {methodology_b}?",
        "Explica las diferencias en {aspect} según {methodology_a} versus {methodology_b}.",
        "¿Cuál metodología es más conservadora, {methodology_a} o {methodology_b}? ¿Por qué?",
        "¿Qué requisitos adicionales tiene {methodology_a} comparado con {methodology_b}?",
    ]

    # Specific aspects for comparison
    COMPARISON_ASPECTS = [
        "requisitos de capital",
        "parámetros de riesgo (PD, LGD, EAD)",
        "complejidad de implementación",
        "requisitos de datos",
        "ponderaciones de riesgo",
        "tratamiento de garantías",
        "sensibilidad al riesgo",
        "carga operativa",
        "requisitos de validación",
        "uso de modelos internos",
    ]

    # Methodology definitions
    METHODOLOGIES = {
        "irb_foundation": {
            "name": "IRB Fundación",
            "short_name": "F-IRB",
            "file": "methodology_irb_foundation.md",
            "key_aspects": [
                "PD estimación propia",
                "LGD valores supervisores (45%/75%)",
                "EAD valores supervisores",
                "5 años datos PD",
            ],
        },
        "irb_advanced": {
            "name": "IRB Avanzado",
            "short_name": "A-IRB",
            "file": "methodology_irb_advanced.md",
            "key_aspects": [
                "Todos los parámetros propios",
                "7 años datos LGD/EAD",
                "Downturn LGD",
                "Mayor beneficio de capital",
            ],
        },
        "standardized": {
            "name": "Método Estándar",
            "short_name": "SA",
            "file": "methodology_standardized.md",
            "key_aspects": [
                "Ponderaciones fijas regulatorias",
                "Sin modelos internos",
                "Dependencia de ratings externos",
                "Menor complejidad",
            ],
        },
        "basel_evolution": {
            "name": "Evolución de Basilea",
            "short_name": "Basel III/IV",
            "file": "methodology_basel_evolution.md",
            "key_aspects": [
                "Output floor 72.5%",
                "Restricciones a modelos internos",
                "Nuevas ponderaciones SA",
                "Floors de parámetros",
            ],
        },
    }

    def __init__(self, methodology_dir: Optional[Path] = None):
        self.methodology_dir = methodology_dir or self.METHODOLOGY_DIR
        self.documents: Dict[str, MethodologyDocument] = {}
        self._load_documents()

    def _load_documents(self):
        """Load all methodology documents."""
        for key, info in self.METHODOLOGIES.items():
            file_path = self.methodology_dir / info["file"]
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                sections = self._parse_sections(content)
                self.documents[key] = MethodologyDocument(
                    name=info["name"],
                    short_name=info["short_name"],
                    path=file_path,
                    content=content,
                    sections=sections,
                    key_aspects=info["key_aspects"],
                )

    def _parse_sections(self, content: str) -> Dict[str, str]:
        """Parse markdown content into sections."""
        sections = {}
        current_section = "introduction"
        current_content = []

        for line in content.split("\n"):
            if line.startswith("## "):
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = line[3:].strip().lower()
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def _extract_comparison_content(
        self, doc_a: MethodologyDocument, doc_b: MethodologyDocument, aspect: str
    ) -> Tuple[str, str]:
        """Extract relevant content for comparison from both documents."""
        # Find relevant sections based on aspect keywords
        aspect_keywords = aspect.lower().split()

        content_a = []
        content_b = []

        for section_name, section_content in doc_a.sections.items():
            if any(kw in section_name for kw in aspect_keywords):
                content_a.append(section_content[:1000])
            elif any(kw in section_content.lower()[:500] for kw in aspect_keywords):
                content_a.append(section_content[:500])

        for section_name, section_content in doc_b.sections.items():
            if any(kw in section_name for kw in aspect_keywords):
                content_b.append(section_content[:1000])
            elif any(kw in section_content.lower()[:500] for kw in aspect_keywords):
                content_b.append(section_content[:500])

        return "\n".join(content_a[:2]), "\n".join(content_b[:2])

    def _generate_comparison_answer(
        self,
        doc_a: MethodologyDocument,
        doc_b: MethodologyDocument,
        aspect: str,
        template_type: str,
    ) -> str:
        """Generate a comparison answer based on the documents."""
        content_a, content_b = self._extract_comparison_content(doc_a, doc_b, aspect)

        # Build structured comparison response
        response_parts = [
            f"Comparando {doc_a.name} ({doc_a.short_name}) con {doc_b.name} ({doc_b.short_name}):\n",
        ]

        # Add key differences
        response_parts.append(f"\n**{doc_a.name}:**")
        for aspect_point in doc_a.key_aspects[:3]:
            response_parts.append(f"- {aspect_point}")

        response_parts.append(f"\n**{doc_b.name}:**")
        for aspect_point in doc_b.key_aspects[:3]:
            response_parts.append(f"- {aspect_point}")

        # Add specific aspect content if available
        if content_a or content_b:
            response_parts.append(f"\n**En cuanto a {aspect}:**")
            if content_a:
                response_parts.append(
                    f"\nSegún {doc_a.short_name}: {content_a[:300]}..."
                )
            if content_b:
                response_parts.append(
                    f"\nSegún {doc_b.short_name}: {content_b[:300]}..."
                )

        # Add conclusion
        response_parts.append(
            f"\n\n**Conclusión:** La elección entre {doc_a.short_name} y {doc_b.short_name} "
            "depende de las capacidades de la entidad, disponibilidad de datos históricos, "
            "y el balance deseado entre sensibilidad al riesgo y complejidad operativa."
        )

        return "\n".join(response_parts)

    def generate_comparison_qa(
        self, doc_a_key: str, doc_b_key: str
    ) -> List[Dict]:
        """Generate comparison Q&A pairs between two methodology documents."""
        if doc_a_key not in self.documents or doc_b_key not in self.documents:
            return []

        doc_a = self.documents[doc_a_key]
        doc_b = self.documents[doc_b_key]

        qa_pairs = []

        # Generate questions for different aspects
        for aspect in self.COMPARISON_ASPECTS:
            # Select random template
            template = random.choice(self.COMPARISON_TEMPLATES)

            question = template.format(
                methodology_a=doc_a.name,
                methodology_b=doc_b.name,
                aspect=aspect,
            )

            answer = self._generate_comparison_answer(doc_a, doc_b, aspect, template)

            qa_pairs.append(
                {
                    "question": question,
                    "answer": answer,
                    "metadata": {
                        "type": "comparison",
                        "methodology_a": doc_a_key,
                        "methodology_b": doc_b_key,
                        "aspect": aspect,
                    },
                }
            )

        return qa_pairs

    def generate_all_comparisons(self) -> List[Dict]:
        """Generate comparison Q&A pairs for all methodology combinations."""
        all_qa_pairs = []
        doc_keys = list(self.documents.keys())

        for i, doc_a_key in enumerate(doc_keys):
            for doc_b_key in doc_keys[i + 1 :]:
                pairs = self.generate_comparison_qa(doc_a_key, doc_b_key)
                all_qa_pairs.extend(pairs)

        return all_qa_pairs

    def format_for_training(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Format Q&A pairs in ChatML format for training."""
        system_prompt = """Eres un experto en regulación bancaria española y europea, especializado en metodologías de riesgo de crédito.

Tu tarea es comparar diferentes enfoques regulatorios (IRB Fundación, IRB Avanzado, Método Estándar, Basilea III/IV) de manera precisa y objetiva.

REGLAS:
- Explica las diferencias clave de manera clara y estructurada
- Menciona ventajas y desventajas de cada enfoque
- Cita las referencias normativas relevantes (CRR, directrices EBA)
- Responde siempre en español"""

        formatted = []
        for qa in qa_pairs:
            formatted.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": qa["question"]},
                        {"role": "assistant", "content": qa["answer"]},
                    ],
                    "metadata": qa["metadata"],
                }
            )

        return formatted

    def save_training_data(
        self, output_path: Optional[Path] = None, append: bool = False
    ) -> Path:
        """Generate and save comparison training data."""
        qa_pairs = self.generate_all_comparisons()
        formatted = self.format_for_training(qa_pairs)

        output_path = output_path or Path("data/processed/methodology_comparisons.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if append and output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            formatted = existing + formatted

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(formatted, f, ensure_ascii=False, indent=2)

        print(f"Generated {len(qa_pairs)} comparison Q&A pairs")
        print(f"Saved to: {output_path}")

        return output_path


def main():
    """Main function to generate methodology comparison data."""
    comparator = MethodologyComparator()

    print("=" * 60)
    print("Methodology Comparator - Training Data Generator")
    print("=" * 60)
    print()

    # Show loaded documents
    print("Loaded methodology documents:")
    for key, doc in comparator.documents.items():
        print(f"  - {doc.name} ({doc.short_name}): {len(doc.sections)} sections")
    print()

    # Generate and save training data
    output_path = comparator.save_training_data()

    print()
    print("=" * 60)
    print("Done! Training data ready for fine-tuning.")
    print("=" * 60)


if __name__ == "__main__":
    main()

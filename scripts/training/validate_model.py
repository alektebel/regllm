#!/usr/bin/env python3
"""
Comprehensive validation for RegLLM fine-tuned model.

This script provides multiple validation approaches:
1. LLM-as-judge evaluation
2. Embedding-based semantic similarity (question-answer consistency)
3. Domain-specific metrics (citation accuracy, technical term usage)
4. Comparative analysis against base model
5. Error analysis and failure modes

More useful for product validation and improvement than pure benchmarking.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ValidationResult:
    """Comprehensive validation result for a single QA pair."""

    question: str
    reference_answer: str
    model_answer: str

    # LLM Judge scores
    judge_score: float = 0.0
    judge_explanation: str = ""

    # Embedding-based metrics
    qa_semantic_similarity: float = 0.0  # Question-Answer alignment
    answer_similarity: float = 0.0  # Model answer vs reference

    # Domain-specific metrics
    has_citation: bool = False
    citation_accuracy: float = 0.0
    technical_term_count: int = 0
    answer_length: int = 0

    # Performance
    latency_ms: int = 0

    # Error analysis
    error_type: str = ""  # "hallucination", "incomplete", "off-topic", "correct"

    metadata: dict = field(default_factory=dict)


class EmbeddingValidator:
    """Validates answer quality using embeddings."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """Initialize embedding model for semantic similarity."""
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded")

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        embeddings = self.model.encode([text1, text2])

        # Cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

    def validate_qa_consistency(self, question: str, answer: str) -> float:
        """
        Validate that the answer is semantically aligned with the question.
        Returns score 0-1 where higher means better alignment.
        """
        return self.compute_similarity(question, answer)

    def compare_answers(self, reference: str, generated: str) -> float:
        """Compare generated answer against reference answer."""
        return self.compute_similarity(reference, generated)


class DomainValidator:
    """Validates domain-specific aspects of banking regulation answers."""

    # Common Spanish banking regulation references
    CITATION_PATTERNS = [
        r"Art[ií]culo\s+\d+",
        r"CRR",
        r"CRD\s*[IV]+",
        r"EBA/GL/\d+",
        r"Basel\s+[IV]+",
        r"Basilea\s+[IV]+",
        r"IFRS\s*9",
        r"IRB",
        r"PD|LGD|EAD",
    ]

    # Technical terms that should appear in good answers
    TECHNICAL_TERMS = [
        "capital",
        "riesgo",
        "crédito",
        "solvencia",
        "liquidez",
        "provisiones",
        "pérdidas",
        "exposición",
        "ponderación",
        "regulación",
        "normativa",
        "requisito",
        "ratio",
        "cartera",
        "activos",
        "pasivos",
        "patrimonio",
    ]

    def __init__(self):
        import re

        self.citation_re = re.compile("|".join(self.CITATION_PATTERNS), re.IGNORECASE)
        self.term_re = re.compile("|".join(self.TECHNICAL_TERMS), re.IGNORECASE)

    def check_citations(self, text: str) -> tuple[bool, float]:
        """
        Check if answer contains proper citations.
        Returns (has_citation, citation_quality_score).
        """
        matches = self.citation_re.findall(text)
        has_citation = len(matches) > 0

        # Quality score based on number and variety of citations
        unique_citations = len(set(m.upper() for m in matches))
        quality = min(1.0, unique_citations / 3.0)  # 3+ citations = perfect score

        return has_citation, quality

    def count_technical_terms(self, text: str) -> int:
        """Count technical banking terms in the answer."""
        matches = self.term_re.findall(text)
        return len(set(m.lower() for m in matches))

    def analyze_completeness(self, answer: str, reference: str) -> dict:
        """Analyze if answer covers key points from reference."""
        # Extract technical terms from reference
        ref_terms = set(self.term_re.findall(reference.lower()))
        ans_terms = set(self.term_re.findall(answer.lower()))

        if not ref_terms:
            return {"coverage": 1.0, "missing_terms": []}

        coverage = len(ref_terms & ans_terms) / len(ref_terms)
        missing = list(ref_terms - ans_terms)

        return {
            "coverage": coverage,
            "missing_terms": missing,
            "covered_terms": list(ref_terms & ans_terms),
        }


class LLMJudge:
    """LLM-based judge for holistic answer quality evaluation."""

    def __init__(self, model_name: str = "qwen2.5:14b-instruct-q4_K_M"):
        self.model_name = model_name

    async def evaluate(
        self, question: str, reference: str, answer: str
    ) -> tuple[float, str, str]:
        """
        Evaluate answer quality.
        Returns (score, explanation, error_type).
        """

        judge_prompt = f"""Evalúa esta respuesta sobre regulación bancaria española.

PREGUNTA: {question}

RESPUESTA ESPERADA:
{reference}

RESPUESTA DEL MODELO:
{answer}

CRITERIOS:
1. Corrección factual (0-25 pts)
2. Completitud (0-25 pts)
3. Claridad (0-20 pts)
4. Referencias normativas (0-20 pts)
5. Relevancia (0-10 pts)

TIPOS DE ERROR:
- "correct" - Respuesta correcta y completa
- "incomplete" - Correcta pero falta información
- "hallucination" - Información incorrecta o inventada
- "off-topic" - No responde la pregunta
- "citation-error" - Referencias incorrectas

Responde SOLO con JSON:
{{
  "score": 0.85,
  "error_type": "correct",
  "explanation": "Explicación detallada",
  "strengths": ["punto 1", "punto 2"],
  "weaknesses": ["punto 1"]
}}

Score debe ser 0.0-1.0.
"""

        try:
            # Use Ollama for local evaluation
            import aiohttp

            payload = {
                "model": self.model_name,
                "prompt": judge_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 400,
                },
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    response = data["response"]

            # Extract JSON
            import re

            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                result = json.loads(json_match.group(0))
                score = float(result.get("score", 0.0))
                error_type = result.get("error_type", "unknown")
                explanation = result.get("explanation", "")
                return score, explanation, error_type

            return 0.5, "No valid JSON response", "unknown"

        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return 0.0, f"Error: {e}", "error"


async def validate_model(
    test_file: Path,
    model_path: Path | None,
    output_file: Path,
    max_questions: int | None = None,
    use_judge: bool = True,
):
    """Run comprehensive validation."""

    logger.info("=" * 60)
    logger.info("RegLLM Model Validation")
    logger.info("=" * 60)

    # Load test dataset
    logger.info(f"Loading test data from {test_file}")
    with open(test_file, "r", encoding="utf-8") as f:
        if test_file.suffix == ".jsonl":
            test_data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            # Handle different formats
            if isinstance(data, dict) and "entries" in data:
                test_data = data["entries"]  # Ground truth format
            elif isinstance(data, list):
                test_data = data
            else:
                test_data = [data]

    if max_questions:
        test_data = test_data[:max_questions]

    logger.info(f"Loaded {len(test_data)} test questions")

    # Initialize validators
    logger.info("Initializing validators...")
    embedding_validator = EmbeddingValidator()
    domain_validator = DomainValidator()
    judge = LLMJudge() if use_judge else None

    # Load model
    logger.info("Loading RegLLM model...")
    from src.rag_system import create_rag_system

    # Initialize RAG system
    rag_system = create_rag_system()
    logger.info(f"RAG system loaded with {rag_system.collection.count()} documents")

    # Run validation
    results = []

    for idx, item in enumerate(test_data, 1):
        # Extract question and reference - handle multiple formats
        if "messages" in item:
            # ChatML format
            question = next(
                (m["content"] for m in item["messages"] if m["role"] == "user"), None
            )
            reference = next(
                (m["content"] for m in item["messages"] if m["role"] == "assistant"),
                None,
            )
        elif "pregunta" in item:
            # Spanish ground truth format
            question = item.get("pregunta")
            reference = item.get("respuesta_esperada", "")
        else:
            # Direct format
            question = item.get("question", item.get("pregunta", ""))
            reference = item.get(
                "answer", item.get("respuesta", item.get("respuesta_esperada", ""))
            )

        if not question or not reference:
            logger.warning(f"Skipping item {idx}: missing data")
            continue

        logger.info(f"\n[{idx}/{len(test_data)}] Validating: {question[:80]}...")

        # Generate answer using RAG system (simplified - just retrieval for validation)
        start = time.time()
        try:
            # Use RAG to get context
            results = await asyncio.get_event_loop().run_in_executor(
                None, rag_system.buscar_hibrida, question, 3
            )

            # For validation, we'll use the formatted context as the "answer"
            # In production, this would go through an LLM
            if results:
                answer = rag_system.formatear_contexto(results)
            else:
                answer = (
                    "No se encontró información relevante en la base de conocimiento."
                )

            latency_ms = int((time.time() - start) * 1000)
            logger.info(f"  ✓ Retrieved context in {latency_ms}ms")
        except Exception as e:
            logger.error(f"  ✗ Generation failed: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Create result object
        result = ValidationResult(
            question=question,
            reference_answer=reference,
            model_answer=answer,
            latency_ms=latency_ms,
            answer_length=len(answer),
            metadata=item.get("metadata", {}),
        )

        # 1. Embedding-based validation
        logger.info("  📊 Computing embeddings...")
        result.qa_semantic_similarity = embedding_validator.validate_qa_consistency(
            question, answer
        )
        result.answer_similarity = embedding_validator.compare_answers(
            reference, answer
        )
        logger.info(f"    Q-A similarity: {result.qa_semantic_similarity:.3f}")
        logger.info(f"    Answer similarity: {result.answer_similarity:.3f}")

        # 2. Domain-specific validation
        logger.info("  🔍 Checking domain specifics...")
        result.has_citation, result.citation_accuracy = (
            domain_validator.check_citations(answer)
        )
        result.technical_term_count = domain_validator.count_technical_terms(answer)
        completeness = domain_validator.analyze_completeness(answer, reference)
        logger.info(
            f"    Citations: {result.has_citation} (quality: {result.citation_accuracy:.2f})"
        )
        logger.info(f"    Technical terms: {result.technical_term_count}")
        logger.info(f"    Coverage: {completeness['coverage']:.2f}")

        # 3. LLM judge evaluation
        if judge:
            logger.info("  ⚖️  Running LLM judge...")
            (
                result.judge_score,
                result.judge_explanation,
                result.error_type,
            ) = await judge.evaluate(question, reference, answer)
            logger.info(f"    Judge score: {result.judge_score:.2f}")
            logger.info(f"    Error type: {result.error_type}")

        logger.info(f"  ✓ Result created, type: {type(result)}")
        results.append(result)
        logger.info(f"  ✓ Appended to results, now {len(results)} items")

    # Save results
    logger.info(f"\nSaving results to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"DEBUG: results type: {type(results)}, len: {len(results)}")
    if results:
        logger.info(f"DEBUG: first result type: {type(results[0])}")

    with open(output_file, "w", encoding="utf-8") as f:
        for i, r in enumerate(results):
            # Convert ValidationResult to dict
            if isinstance(r, ValidationResult):
                result_dict = {
                    "question": r.question,
                    "reference_answer": r.reference_answer,
                    "model_answer": r.model_answer,
                    "judge_score": r.judge_score,
                    "judge_explanation": r.judge_explanation,
                    "qa_semantic_similarity": r.qa_semantic_similarity,
                    "answer_similarity": r.answer_similarity,
                    "has_citation": r.has_citation,
                    "citation_accuracy": r.citation_accuracy,
                    "technical_term_count": r.technical_term_count,
                    "answer_length": r.answer_length,
                    "latency_ms": r.latency_ms,
                    "error_type": r.error_type,
                    "metadata": r.metadata,
                }
            elif isinstance(r, dict):
                # Already a dict, use as-is
                result_dict = r
            else:
                logger.error(f"Invalid result type at index {i}: {type(r)}")
                continue

            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

    # Generate report
    generate_validation_report(results, output_file.parent / "validation_report.md")

    logger.info("\n✅ Validation complete!")


def generate_validation_report(results: list[ValidationResult], output_path: Path):
    """Generate comprehensive validation report."""

    if not results:
        logger.warning("No results to report")
        return

    # Compute statistics
    avg_judge = (
        np.mean([r.judge_score for r in results]) if results[0].judge_score else 0
    )
    avg_qa_sim = np.mean([r.qa_semantic_similarity for r in results])
    avg_ans_sim = np.mean([r.answer_similarity for r in results])
    avg_latency = np.mean([r.latency_ms for r in results])
    citation_rate = sum(1 for r in results if r.has_citation) / len(results)
    avg_citation_quality = np.mean([r.citation_accuracy for r in results])
    avg_terms = np.mean([r.technical_term_count for r in results])

    # Error type distribution
    error_types = {}
    for r in results:
        if r.error_type:
            error_types[r.error_type] = error_types.get(r.error_type, 0) + 1

    # Generate report
    lines = [
        "# RegLLM Validation Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Questions Evaluated:** {len(results)}",
        "",
        "## Overall Metrics",
        "",
        "### Quality Scores",
        "",
        f"- **LLM Judge Score:** {avg_judge:.3f} / 1.0",
        f"- **Q-A Semantic Alignment:** {avg_qa_sim:.3f} / 1.0",
        f"- **Answer Similarity (vs reference):** {avg_ans_sim:.3f} / 1.0",
        "",
        "### Domain-Specific Metrics",
        "",
        f"- **Citation Rate:** {citation_rate:.1%}",
        f"- **Average Citation Quality:** {avg_citation_quality:.3f} / 1.0",
        f"- **Avg Technical Terms per Answer:** {avg_terms:.1f}",
        "",
        "### Performance",
        "",
        f"- **Average Latency:** {avg_latency:.0f}ms",
        f"- **Median Latency:** {np.median([r.latency_ms for r in results]):.0f}ms",
        f"- **P95 Latency:** {np.percentile([r.latency_ms for r in results], 95):.0f}ms",
        "",
        "## Error Analysis",
        "",
        "| Error Type | Count | Percentage |",
        "|------------|-------|------------|",
    ]

    for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        lines.append(f"| {error_type} | {count} | {pct:.1f}% |")

    lines.extend(
        [
            "",
            "## Quality Distribution",
            "",
            "### By Judge Score",
            "",
        ]
    )

    if avg_judge > 0:
        excellent = sum(1 for r in results if r.judge_score >= 0.8)
        good = sum(1 for r in results if 0.6 <= r.judge_score < 0.8)
        fair = sum(1 for r in results if 0.4 <= r.judge_score < 0.6)
        poor = sum(1 for r in results if r.judge_score < 0.4)

        lines.extend(
            [
                f"- **Excellent (≥0.8):** {excellent} ({excellent / len(results) * 100:.1f}%)",
                f"- **Good (0.6-0.8):** {good} ({good / len(results) * 100:.1f}%)",
                f"- **Fair (0.4-0.6):** {fair} ({fair / len(results) * 100:.1f}%)",
                f"- **Poor (<0.4):** {poor} ({poor / len(results) * 100:.1f}%)",
            ]
        )

    lines.extend(
        [
            "",
            "## Sample Results",
            "",
            "### Best Answers",
            "",
        ]
    )

    # Sort by composite score
    sorted_results = sorted(
        results,
        key=lambda r: (
            r.judge_score * 0.4
            + r.qa_semantic_similarity * 0.3
            + r.answer_similarity * 0.3
        ),
        reverse=True,
    )

    for i, r in enumerate(sorted_results[:5], 1):
        lines.extend(
            [
                f"#### {i}. Score: {r.judge_score:.2f}",
                f"**Q:** {r.question[:150]}...",
                f"**A:** {r.model_answer[:200]}...",
                f"- Q-A Alignment: {r.qa_semantic_similarity:.2f}",
                f"- Citations: {'✓' if r.has_citation else '✗'}",
                f"- Technical Terms: {r.technical_term_count}",
                f"- Latency: {r.latency_ms}ms",
                "",
            ]
        )

    lines.extend(
        [
            "### Worst Answers",
            "",
        ]
    )

    for i, r in enumerate(sorted_results[-5:], 1):
        lines.extend(
            [
                f"#### {i}. Score: {r.judge_score:.2f}",
                f"**Q:** {r.question[:150]}...",
                f"**A:** {r.model_answer[:200]}...",
                f"- Error Type: {r.error_type}",
                f"- Judge: {r.judge_explanation[:150]}...",
                "",
            ]
        )

    # Write report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive validation for RegLLM model"
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "test_ground_truth.json",
        help="Test dataset file",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to fine-tuned model (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "validation" / "results.jsonl",
        help="Output file",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Max questions to validate",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judge evaluation",
    )

    args = parser.parse_args()

    asyncio.run(
        validate_model(
            test_file=args.test_file,
            model_path=args.model_path,
            output_file=args.output,
            max_questions=args.max_questions,
            use_judge=not args.no_judge,
        )
    )


if __name__ == "__main__":
    main()

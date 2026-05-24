#!/usr/bin/env python3
"""
Benchmark QA models: Compare local model against GPT-4, Sonnet-4.5, and Qwen2.5-7B.

This script:
1. Loads a test dataset of QA pairs
2. Generates answers using multiple models
3. Evaluates using LLM judge (multiple judges for fairness)
4. Computes metrics: accuracy, latency, cost
5. Generates comparison report
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BenchmarkResult:
    """Result for a single QA evaluation."""

    question: str
    reference_answer: str
    model_answer: str
    model_name: str
    judge_score: float = 0.0
    judge_explanation: str = ""
    latency_ms: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for a model to benchmark."""

    name: str
    provider: str  # "openai", "anthropic", "ollama", "local"
    model_id: str
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 512
    temperature: float = 0.7


class ModelBackend:
    """Abstract backend for different model providers."""

    @staticmethod
    async def generate(
        config: ModelConfig, prompt: str, system_prompt: str = ""
    ) -> tuple[str, int]:
        """Generate answer and return (answer, latency_ms)."""
        start = time.time()

        if config.provider == "openai":
            answer = await ModelBackend._generate_openai(config, prompt, system_prompt)
        elif config.provider == "anthropic":
            answer = await ModelBackend._generate_anthropic(
                config, prompt, system_prompt
            )
        elif config.provider == "ollama":
            answer = await ModelBackend._generate_ollama(config, prompt, system_prompt)
        elif config.provider == "local":
            answer = await ModelBackend._generate_local(config, prompt, system_prompt)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")

        latency_ms = int((time.time() - start) * 1000)
        return answer, latency_ms

    @staticmethod
    async def _generate_openai(
        config: ModelConfig, prompt: str, system_prompt: str
    ) -> str:
        """Generate using OpenAI API."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=config.api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=config.model_id,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

        return response.choices[0].message.content

    @staticmethod
    async def _generate_anthropic(
        config: ModelConfig, prompt: str, system_prompt: str
    ) -> str:
        """Generate using Anthropic API."""
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=config.api_key)

        response = await client.messages.create(
            model=config.model_id,
            system=system_prompt or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

        return response.content[0].text

    @staticmethod
    async def _generate_ollama(
        config: ModelConfig, prompt: str, system_prompt: str
    ) -> str:
        """Generate using Ollama."""
        import aiohttp

        base_url = config.base_url or "http://localhost:11434"

        payload = {
            "model": config.model_id,
            "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
            "stream": False,
            "options": {
                "num_predict": config.max_tokens,
                "temperature": config.temperature,
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["response"]

    @staticmethod
    async def _generate_local(
        config: ModelConfig, prompt: str, system_prompt: str
    ) -> str:
        """Generate using local model (via RAG system or direct inference)."""
        # For now, use the existing RAG system
        from src.rag_chat import RegulatoryRAGChat

        rag = RegulatoryRAGChat()
        result = rag.query(prompt)
        return result["respuesta"]


class LLMJudge:
    """LLM-based judge for evaluating answer quality."""

    def __init__(self, judge_config: ModelConfig):
        self.config = judge_config

    async def evaluate(
        self, question: str, reference: str, answer: str
    ) -> tuple[float, str]:
        """
        Evaluate answer quality.
        Returns (score, explanation) where score is 0.0-1.0.
        """

        judge_prompt = f"""Evalúa la siguiente respuesta a una pregunta sobre regulación bancaria.

PREGUNTA: {question}

RESPUESTA DE REFERENCIA (Gold Standard):
{reference}

RESPUESTA A EVALUAR:
{answer}

CRITERIOS DE EVALUACIÓN:
1. Corrección factual (¿es correcta la información?)
2. Completitud (¿cubre los puntos clave?)
3. Relevancia (¿responde directamente a la pregunta?)
4. Referencias (¿cita fuentes normativas cuando es apropiado?)
5. Claridad (¿es clara y comprensible?)

Responde SOLO con un objeto JSON:
{{
  "score": 0.85,
  "explanation": "Explicación breve de la evaluación",
  "strengths": ["punto fuerte 1", "punto fuerte 2"],
  "weaknesses": ["punto débil 1", "punto débil 2"]
}}

El score debe ser entre 0.0 (muy mala) y 1.0 (excelente).
"""

        try:
            response, _ = await ModelBackend.generate(
                self.config,
                judge_prompt,
                "Eres un experto evaluador de respuestas sobre regulación bancaria. Sé objetivo y preciso.",
            )

            # Extract JSON from response
            import re

            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group(0))
                score = float(data.get("score", 0.0))
                explanation = data.get("explanation", "")
                return score, explanation
            else:
                logger.warning("Judge did not return valid JSON, using default score")
                return 0.5, "No valid evaluation"

        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return 0.0, f"Evaluation error: {str(e)}"


async def run_benchmark(
    test_file: Path,
    models: list[ModelConfig],
    judge_config: ModelConfig,
    output_file: Path,
    max_questions: int = None,
):
    """Run benchmark across all models."""

    # Load test dataset
    logger.info(f"Loading test dataset from {test_file}")
    test_data = []

    if test_file.suffix == ".jsonl":
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
    else:
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

    if max_questions:
        test_data = test_data[:max_questions]

    logger.info(f"Loaded {len(test_data)} test questions")

    # Initialize judge
    judge = LLMJudge(judge_config)

    # Run evaluation for each model
    all_results = []

    for model_config in models:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Benchmarking: {model_config.name}")
        logger.info(f"{'=' * 60}")

        model_results = []

        for idx, item in enumerate(test_data, 1):
            # Extract question and reference answer
            if "messages" in item:
                # ChatML format
                question = next(
                    (m["content"] for m in item["messages"] if m["role"] == "user"),
                    None,
                )
                reference = next(
                    (
                        m["content"]
                        for m in item["messages"]
                        if m["role"] == "assistant"
                    ),
                    None,
                )
            else:
                # Direct format
                question = item.get("question", item.get("pregunta", ""))
                reference = item.get("answer", item.get("respuesta", ""))

            if not question or not reference:
                logger.warning(f"Skipping item {idx}: missing question or answer")
                continue

            logger.info(f"[{idx}/{len(test_data)}] Q: {question[:80]}...")

            # Generate answer
            try:
                answer, latency = await ModelBackend.generate(
                    model_config,
                    question,
                    "Eres un experto en regulación bancaria española. Responde de forma precisa y cita fuentes cuando sea posible.",
                )
                logger.info(f"  ✓ Generated in {latency}ms")
            except Exception as e:
                logger.error(f"  ✗ Generation failed: {e}")
                answer = ""
                latency = 0

            # Evaluate with judge
            try:
                score, explanation = await judge.evaluate(question, reference, answer)
                logger.info(f"  📊 Judge score: {score:.2f}")
            except Exception as e:
                logger.error(f"  ✗ Evaluation failed: {e}")
                score = 0.0
                explanation = f"Error: {e}"

            result = BenchmarkResult(
                question=question,
                reference_answer=reference,
                model_answer=answer,
                model_name=model_config.name,
                judge_score=score,
                judge_explanation=explanation,
                latency_ms=latency,
                metadata=item.get("metadata", {}),
            )

            model_results.append(result)

        all_results.extend(model_results)

        # Compute aggregate metrics for this model
        avg_score = (
            sum(r.judge_score for r in model_results) / len(model_results)
            if model_results
            else 0
        )
        avg_latency = (
            sum(r.latency_ms for r in model_results) / len(model_results)
            if model_results
            else 0
        )

        logger.info(f"\n{model_config.name} Summary:")
        logger.info(f"  Avg Score: {avg_score:.3f}")
        logger.info(f"  Avg Latency: {avg_latency:.0f}ms")

    # Save results
    logger.info(f"\nSaving results to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(
                json.dumps(
                    {
                        "question": result.question,
                        "reference_answer": result.reference_answer,
                        "model_answer": result.model_answer,
                        "model_name": result.model_name,
                        "judge_score": result.judge_score,
                        "judge_explanation": result.judge_explanation,
                        "latency_ms": result.latency_ms,
                        "metadata": result.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Generate summary report
    generate_report(all_results, output_file.parent / "benchmark_report.md")

    logger.info("\n✅ Benchmark complete!")


def generate_report(results: list[BenchmarkResult], output_path: Path):
    """Generate markdown report with benchmark results."""

    # Group by model
    by_model = {}
    for r in results:
        if r.model_name not in by_model:
            by_model[r.model_name] = []
        by_model[r.model_name].append(r)

    # Compute statistics
    stats = {}
    for model_name, model_results in by_model.items():
        scores = [r.judge_score for r in model_results]
        latencies = [r.latency_ms for r in model_results]

        stats[model_name] = {
            "count": len(model_results),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "min_latency": min(latencies) if latencies else 0,
            "max_latency": max(latencies) if latencies else 0,
        }

    # Generate markdown
    lines = [
        "# QA Model Benchmark Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Questions:** {len(results)}",
        "",
        "## Overall Results",
        "",
        "| Model | Avg Score | Min | Max | Avg Latency (ms) | Count |",
        "|-------|-----------|-----|-----|------------------|-------|",
    ]

    # Sort by avg score descending
    sorted_models = sorted(stats.items(), key=lambda x: x[1]["avg_score"], reverse=True)

    for model_name, s in sorted_models:
        lines.append(
            f"| {model_name} | {s['avg_score']:.3f} | {s['min_score']:.2f} | "
            f"{s['max_score']:.2f} | {s['avg_latency']:.0f} | {s['count']} |"
        )

    lines.extend(
        [
            "",
            "## Detailed Results",
            "",
        ]
    )

    # Add detailed results per model
    for model_name, model_results in by_model.items():
        lines.extend(
            [
                f"### {model_name}",
                "",
            ]
        )

        # Show top 5 and bottom 5 by score
        sorted_results = sorted(
            model_results, key=lambda r: r.judge_score, reverse=True
        )

        lines.extend(
            [
                "#### Best Answers",
                "",
            ]
        )

        for i, r in enumerate(sorted_results[:5], 1):
            lines.extend(
                [
                    f"**{i}. Score: {r.judge_score:.2f}** ({r.latency_ms}ms)",
                    f"- Q: {r.question[:120]}...",
                    f"- Judge: {r.judge_explanation[:150]}...",
                    "",
                ]
            )

        lines.extend(
            [
                "#### Worst Answers",
                "",
            ]
        )

        for i, r in enumerate(sorted_results[-5:], 1):
            lines.extend(
                [
                    f"**{i}. Score: {r.judge_score:.2f}** ({r.latency_ms}ms)",
                    f"- Q: {r.question[:120]}...",
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
        description="Benchmark QA models against each other using LLM judge"
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "test_ground_truth.json",
        help="Test dataset file (JSON or JSONL)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "results.jsonl",
        help="Output file for results",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4",
        help="Model to use as judge (default: gpt-4)",
    )
    parser.add_argument(
        "--judge-provider",
        default="openai",
        help="Provider for judge model",
    )

    args = parser.parse_args()

    # Configure models to benchmark
    models = [
        ModelConfig(
            name="GPT-4",
            provider="openai",
            model_id="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY", ""),
        ),
        ModelConfig(
            name="Claude Sonnet 4.5",
            provider="anthropic",
            model_id="claude-sonnet-4.5-20250514",
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        ),
        ModelConfig(
            name="Qwen2.5-7B (Local)",
            provider="ollama",
            model_id="qwen2.5:7b",
            base_url="http://localhost:11434",
        ),
        ModelConfig(
            name="Qwen2.5-14B (Local)",
            provider="ollama",
            model_id="qwen2.5:14b-instruct-q4_K_M",
            base_url="http://localhost:11434",
        ),
        ModelConfig(
            name="Fine-tuned Model",
            provider="local",
            model_id="regllm-finetuned",
        ),
    ]

    # Configure judge
    if args.judge_provider == "ollama":
        judge_config = ModelConfig(
            name=f"Judge-{args.judge_model}",
            provider="ollama",
            model_id=args.judge_model,
            base_url="http://localhost:11434",
        )
    else:
        judge_config = ModelConfig(
            name=f"Judge-{args.judge_model}",
            provider=args.judge_provider,
            model_id=args.judge_model,
            api_key=os.getenv(
                "OPENAI_API_KEY"
                if args.judge_provider == "openai"
                else "ANTHROPIC_API_KEY",
                "",
            ),
        )

    # Run benchmark
    asyncio.run(
        run_benchmark(
            test_file=args.test_file,
            models=models,
            judge_config=judge_config,
            output_file=args.output,
            max_questions=args.max_questions,
        )
    )


if __name__ == "__main__":
    main()

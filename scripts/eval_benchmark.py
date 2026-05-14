"""
RegLLM Evaluation Benchmark
============================

Runs a fixed set of gold Q&A pairs through the model and scores them.
Results are logged to MLflow so every training run can be compared.

TODO: Use Claude Opus as the judge instead of keyword F1.
  - Call Anthropic API with the gold answer + model answer.
  - Ask Claude to score on: accuracy (1-5), citation quality (1-5),
    hallucination (yes/no), regulatory correctness (1-5).
  - Log each dimension as a separate MLflow metric.
  - Example prompt:
      "You are a senior banking regulation expert. Given the reference answer
       and the model answer below, rate the model answer on:
       1. Accuracy (1-5): does it correctly describe the regulation?
       2. Citation quality (1-5): are article numbers/sources correct?
       3. Hallucination (0=none, 1=minor, 2=major): invented facts?
       4. Regulatory correctness (1-5): would a compliance officer trust it?
       Reference: {gold_answer}
       Model answer: {model_answer}"

TODO: Build a gold dataset of ~100 Q&A pairs.
  - Cover: CRR articles, EBA guidelines, Basel III/IV ratios, IFRS 9.
  - Store in data/benchmark/gold_qa.jsonl
  - Format: {"question": "...", "answer": "...", "articles": ["CRR art.92"]}
  - Sources: EBA Q&A tool, official FAQ documents, known compliance answers.
  - Validate with a domain expert before treating as ground truth.

TODO: Add as a GitHub Actions step after deploy.yml succeeds:
  - Run eval_benchmark.py against the deployed ECS endpoint.
  - Fail the deploy if avg accuracy drops below a threshold (e.g., 3.5/5).

Usage (current keyword-F1 implementation):
    python scripts/eval_benchmark.py [--adapter PATH] [--backend groq]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


def _keyword_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference (language-agnostic)."""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def load_gold_dataset() -> list[dict]:
    gold_path = PROJECT_ROOT / "data/benchmark/gold_qa.jsonl"
    if not gold_path.exists():
        logger.warning(
            f"Gold dataset not found at {gold_path}. "
            "Create data/benchmark/gold_qa.jsonl first.\n"
            "Format: {\"question\": \"...\", \"answer\": \"...\", \"articles\": [...]}"
        )
        return []
    items = []
    with open(gold_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info(f"Loaded {len(items)} gold Q&A pairs")
    return items


def run_benchmark(backend: str = "groq", adapter: str | None = None) -> dict:
    """Run evaluation and return summary metrics."""
    import time

    # Lazy import app to avoid loading heavy deps unless needed
    import app as regllm_app

    regllm_app.init(backend=backend, adapter=adapter)

    gold = load_gold_dataset()
    if not gold:
        return {}

    scores = []
    for i, item in enumerate(gold, 1):
        question = item["question"]
        reference = item["answer"]

        t0 = time.time()
        context, sources = regllm_app.engine.build_context(question)
        messages = regllm_app.engine.build_messages(question, context, [])

        if backend == "groq":
            raw = regllm_app._generate_groq(messages)
        else:
            raise NotImplementedError(f"benchmark --backend {backend} not yet implemented")

        latency_ms = int((time.time() - t0) * 1000)
        parsed = regllm_app.engine.parse_response(raw)
        answer = parsed.get("respuesta", raw)

        f1 = _keyword_f1(answer, reference)
        scores.append({"question": question[:60], "f1": f1, "latency_ms": latency_ms})
        logger.info(f"[{i}/{len(gold)}] F1={f1:.3f}  latency={latency_ms}ms  {question[:50]}")

    avg_f1 = sum(s["f1"] for s in scores) / len(scores)
    avg_latency = sum(s["latency_ms"] for s in scores) / len(scores)
    summary = {"avg_f1": round(avg_f1, 4), "avg_latency_ms": round(avg_latency)}
    logger.info(f"\nResults: avg_f1={avg_f1:.4f}  avg_latency={avg_latency:.0f}ms")

    # Log to MLflow if available
    try:
        import mlflow
        from config import MLFLOW as MLFLOW_CFG

        mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])
        mlflow.set_experiment("regllm-benchmark")
        with mlflow.start_run(run_name=f"benchmark_{backend}"):
            mlflow.log_metrics(summary)
            mlflow.log_param("backend", backend)
            mlflow.log_param("n_questions", len(gold))
    except Exception as e:
        logger.warning(f"MLflow logging skipped: {e}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="groq")
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args()
    results = run_benchmark(backend=args.backend, adapter=args.adapter)
    if not results:
        sys.exit(1)

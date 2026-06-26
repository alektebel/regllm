"""Tool-calling and Spanish case evaluator.

Evaluates the 4B model's ability to:
1. Select the correct tool for a given Spanish regulatory question
2. Follow the correct multi-step investigation flow (tool sequence)
3. Answer accurately based on tool results

Evaluation is done by comparing against golden datasets and can be run
against a live Ollama model (after training) or via a stub for CI.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(_PROJECT_ROOT))

from src.agent.agent import _LITE_SYSTEM_PROMPT
from src.agent.tools import TOOL_REGISTRY, dispatch_tool

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating a single question."""
    question: str
    expected_tool: str
    predicted_tool: str
    correct: bool
    latency_ms: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSummary:
    """Summary of an evaluation run."""
    total: int
    correct: int
    accuracy: float
    per_tool: dict[str, dict[str, float]]
    results: list[EvalResult]
    meta: dict[str, Any] = field(default_factory=dict)


class ToolCallingEvaluator:
    """Evaluates the 4B model's tool-calling accuracy.

    Uses a lightweight judge approach: for each question, we simulate
    calling the model and check if the predicted tool matches the expected
    tool. The "model" can be:
      - A callable (e.g. an LLM client) that returns tool calls
      - A stub that uses keyword matching (for CI / dry runs)
    """

    def __init__(
        self,
        model_fn: Callable[[str], tuple[str, dict[str, Any]]] | None = None,
    ) -> None:
        self._model_fn = model_fn or self._stub_predict

    def evaluate_manifest(
        self,
        manifest_path: str | Path,
        timeout_s: float = 600,
    ) -> EvalSummary:
        """Evaluate against a tool_selection_manifest.json.

        Each entry has: question, expected_tool (or expected_tools).
        """
        manifest = json.loads(Path(manifest_path).read_text())
        results: list[EvalResult] = []
        per_tool: dict[str, dict[str, float]] = {}
        deadline = time.time() + timeout_s

        for entry in manifest:
            if time.time() > deadline:
                logger.warning("Evaluation timed out after %ds", timeout_s)
                break

            question = entry["question"]
            expected = entry.get("expected_tool") or ""
            expected_tools: list[str] = entry.get("expected_tools", [])

            start = time.time()
            predicted_name, predicted_args = self._model_fn(question)
            latency = (time.time() - start) * 1000

            # For single-tool entries, check primary tool matches
            if expected:
                correct = predicted_name == expected
            else:
                correct = predicted_name in expected_tools if expected_tools else False

            results.append(EvalResult(
                question=question,
                expected_tool=expected or ",".join(expected_tools),
                predicted_tool=predicted_name,
                correct=correct,
                latency_ms=round(latency, 1),
                details={"predicted_args": predicted_args, "source_id": entry.get("source_id", "")},
            ))

            tool_key = expected or "multi"
            if tool_key not in per_tool:
                per_tool[tool_key] = {"correct": 0.0, "total": 0.0}
            per_tool[tool_key]["total"] += 1
            if correct:
                per_tool[tool_key]["correct"] += 1

        total = len(results)
        correct = sum(1 for r in results if r.correct)
        for v in per_tool.values():
            v["accuracy"] = v["correct"] / v["total"] if v["total"] > 0 else 0.0

        return EvalSummary(
            total=total, correct=correct,
            accuracy=correct / total if total > 0 else 0.0,
            per_tool=per_tool, results=results,
        )

    def evaluate_golden_dataset(
        self,
        dataset_path: str | Path,
        n: int | None = None,
        timeout_s: float = 600,
    ) -> EvalSummary:
        """Evaluate against the golden eval dataset (multi-tool traces)."""
        dataset = json.loads(Path(dataset_path).read_text())
        if n:
            dataset = dataset[:n]

        results: list[EvalResult] = []
        per_tool: dict[str, dict[str, float]] = {}
        deadline = time.time() + timeout_s

        for entry in dataset:
            if time.time() > deadline:
                break

            question = entry["question"]
            solution = entry.get("solution", {})
            tool_sequence = solution.get("tool_sequence", [])
            expected_tools = [
                self._parse_tool_name(t) for t in tool_sequence
                if self._parse_tool_name(t)
            ]
            if not expected_tools:
                expected_tools = entry.get("expected_tools", [])

            start = time.time()
            predicted_name, predicted_args = self._model_fn(question)
            latency = (time.time() - start) * 1000

            correct = predicted_name in expected_tools if expected_tools else False

            results.append(EvalResult(
                question=question,
                expected_tool=",".join(expected_tools[:3]),
                predicted_tool=predicted_name,
                correct=correct,
                latency_ms=round(latency, 1),
                details={"source_id": entry.get("id", ""), "category": entry.get("category", "")},
            ))

            for t in expected_tools[:1]:
                if t not in per_tool:
                    per_tool[t] = {"correct": 0.0, "total": 0.0}
                per_tool[t]["total"] += 1
                if correct:
                    per_tool[t]["correct"] += 1

        total = len(results)
        correct = sum(1 for r in results if r.correct)
        for v in per_tool.values():
            v["accuracy"] = v["correct"] / v["total"] if v["total"] > 0 else 0.0

        return EvalSummary(
            total=total, correct=correct,
            accuracy=correct / total if total > 0 else 0.0,
            per_tool=per_tool, results=results,
        )

    # ── Stub predictor (keyword matching for dry runs) ──────────────────

    _TOOL_PATTERNS: dict[str, list[str]] = {
        "trace_dependencies": ["depend", "alimenta", "intervienen", "componen", "dónde viene", "de dónde"],
        "inspect_lineage": ["dónde se calcula", "cómo se define", "qué línea", "data step", "asigna", "asignación"],
        "search_docs": ["documentación", "changelog", "información", "busca", "documental"],
        "search_regulation": ["regulación", "regula", "normativa", "artículo", "circular", "CRR", "EBA", "norma", "suelo"],
        "get_field_definition": ["qué es", "define", "definición", "explica", "significa"],
    }

    def _stub_predict(self, question: str) -> tuple[str, dict[str, Any]]:
        """Keyword-based stub that predicts the tool for CI/dry-run."""
        q_lower = question.lower()
        best_tool = "search_docs"
        best_score = 0

        for tool, patterns in self._TOOL_PATTERNS.items():
            score = sum(1 for p in patterns if p in q_lower)
            if score > best_score:
                best_score = score
                best_tool = tool

        # Extract field mentions
        field_pattern = re.compile(r"\b([A-Z_]{3,})\b")
        fields = field_pattern.findall(question.upper())
        field = fields[0] if fields else "ECL"

        args = self._make_args(best_tool, field)
        return best_tool, args

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_tool_name(raw: str) -> str | None:
        m = re.match(r"\s*([a-z_]+)\(", raw.strip(), re.IGNORECASE)
        return m.group(1).lower() if m else None

    @staticmethod
    def _make_args(tool: str, field: str) -> dict[str, Any]:
        if tool in ("trace_dependencies", "inspect_lineage"):
            return {"target": field, "session_id": "debug_lgd"}
        elif tool == "get_field_definition":
            return {"field": field}
        elif tool in ("search_docs", "search_regulation"):
            return {"query": field}
        return {}

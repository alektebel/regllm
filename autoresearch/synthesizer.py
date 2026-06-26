"""Synthetic data synthesizer for tool-calling and Spanish case SFT.

Generates chat-format training examples in two modes:
1. Tool-calling examples — teach the 4B model to select the right tool
   for a given Spanish regulatory question.
2. Spanish case analysis — full multi-turn investigation traces with
   tool calls, tool results, and final answers in Spanish.

Uses the same lite system prompt as the 4B agent, and dispatches real
tools to get realistic tool results.
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.agent.agent import _LITE_SYSTEM_PROMPT
from src.agent.tools import TOOL_REGISTRY, dispatch_tool

logger = logging.getLogger(__name__)

SESSION = "debug_lgd"

# ── Question templates by tool ─────────────────────────────────────────

_TOOL_QUESTIONS: dict[str, list[str]] = {
    "trace_dependencies": [
        "¿De qué campos depende {field} en el pipeline SAS?",
        "¿Qué campos alimentan el cálculo de {field}?",
        "¿Cuáles son las dependencias de {field}?",
        "¿Qué variables intervienen en {field}?",
        "¿De dónde obtiene {field} sus valores de entrada?",
    ],
    "inspect_lineage": [
        "¿Dónde se calcula {field} en el código SAS?",
        "¿En qué línea del pipeline se asigna {field}?",
        "¿Cómo se define {field} en el flujo de datos?",
        "¿Qué data steps crean o modifican {field}?",
        "¿Cuál es la expresión de asignación de {field}?",
    ],
    "search_docs": [
        "¿Qué documentación hay sobre {field}?",
        "¿Hay algún changelog que mencione {field}?",
        "¿Qué dice la documentación acerca de {field}?",
        "Busca información sobre {field} en la base documental.",
    ],
    "search_regulation": [
        "¿Qué regulación aplica a {field}?",
        "¿Qué artículos mencionan {field}?",
        "¿Qué normativa regula {field}?",
        "¿Hay circulares del BdE sobre {field}?",
        "¿Cómo regula la CRR el campo {field}?",
    ],
    "get_field_definition": [
        "¿Qué es el campo {field}?",
        "Define el campo {field}.",
        "¿Cuál es la definición de {field}?",
        "Explica qué significa {field} en el contexto IRB.",
    ],
}

# ── Multi-step case templates (Spanish investigation traces) ───────────

_CASE_TEMPLATES: list[dict[str, Any]] = [
    {
        "category": "lineage",
        "question": "Traza el linaje completo de {field} en el pipeline SAS.",
        "tools": ["trace_dependencies", "inspect_lineage"],
        "answer_template": (
            "El linaje de **{field}** se compone de:\n"
            "{deps}\n\n"
            "La expresión de asignación es:\n"
            "{expr}\n\n"
            "Campos involucrados: {fields}."
        ),
    },
    {
        "category": "regulation",
        "question": "¿Qué normativa regula {field} y cómo se implementa en el pipeline?",
        "tools": ["search_regulation", "trace_dependencies"],
        "answer_template": (
            "**Regulación aplicable a {field}:**\n{reg}\n\n"
            "**Implementación en pipeline:**\n{deps}"
        ),
    },
    {
        "category": "field_definition",
        "question": "¿Qué es {field} y cómo se calcula?",
        "tools": ["get_field_definition", "trace_dependencies"],
        "answer_template": (
            "**{field}:** {defn}\n\n"
            "**Dependencias en el pipeline:**\n{deps}"
        ),
    },
    {
        "category": "cross_reference",
        "question": "¿Qué dice la regulación sobre {field} y qué campos lo componen?",
        "tools": ["search_regulation", "trace_dependencies"],
        "answer_template": (
            "**Marco regulatorio de {field}:**\n{reg}\n\n"
            "**Composición:**\n{deps}"
        ),
    },
]

# ── Fields used in generation ──────────────────────────────────────────

_FIELDS = [
    "ECL", "PD_ESTIMADA", "LGD_ESTIMADA", "EAD", "RWA",
    "LGD_CON_MOC", "MoC", "LGD_FLOOR_APLICADO", "STAGE_IFRS9",
    "DPDS", "CURE_FLAG", "COLATERAL_TIPO", "SEGMENTO",
    "PROVISION_PERIOD_MONTHS", "VENTANA_OBSERVACION_YEARS",
    "OR_EAD_TIT", "OR_EAD", "SW_FUSION", "K_IRB", "RATING_GRADO",
]


class SFTDataSynthesizer:
    """Generates synthetic SFT training data for tool calling + Spanish cases."""

    def __init__(
        self,
        output_dir: str | Path = "autoresearch/data/sft",
        seed: int = 42,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._rng = random.Random(seed)

    def generate_tool_calling(
        self,
        n: int = 200,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate single-tool-call SFT examples.

        Each example: system prompt, user question (Spanish), tool call
        with real arguments, tool result, and assistant answer.
        """
        fields = fields or _FIELDS
        examples: list[dict[str, Any]] = []

        for _ in range(n):
            tool = self._rng.choice(list(_TOOL_QUESTIONS.keys()))
            field = self._rng.choice(fields)
            question_tmpl = self._rng.choice(_TOOL_QUESTIONS[tool])
            question = question_tmpl.format(field=field)

            args = self._make_args(tool, field)
            result = dispatch_tool(tool, args)
            answer = self._make_single_answer(tool, field, result)

            ex = self._format_trace(question, [(tool, args)], answer)
            ex["meta"] = {"tool": tool, "field": field, "kind": "tool_calling"}
            examples.append(ex)

        return examples

    def generate_spanish_cases(
        self,
        n: int = 100,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate multi-step investigation traces in Spanish.

        Each example chains 2-3 tool calls with real results and a
        synthesised answer in Spanish.
        """
        fields = fields or _FIELDS
        examples: list[dict[str, Any]] = []

        for _ in range(n):
            template = self._rng.choice(_CASE_TEMPLATES)
            field = self._rng.choice(fields)
            question = template["question"].format(field=field)

            steps: list[tuple[str, dict]] = []
            for tool_name in template["tools"]:
                args = self._make_args(tool_name, field)
                steps.append((tool_name, args))

            answer = self._make_case_answer(template, field, steps)
            ex = self._format_trace(question, steps, answer)
            ex["meta"] = {
                "tools": template["tools"],
                "field": field,
                "category": template["category"],
                "kind": "spanish_case",
            }
            examples.append(ex)

        return examples

    def generate_mixed(
        self,
        n_tool_calling: int = 200,
        n_spanish_cases: int = 100,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Generate both types and split train/eval (90/10)."""
        tool_ex = self.generate_tool_calling(n_tool_calling)
        case_ex = self.generate_spanish_cases(n_spanish_cases)
        all_ex = tool_ex + case_ex
        self._rng.shuffle(all_ex)

        split = max(1, int(len(all_ex) * 0.1))
        eval_ex = all_ex[:split]
        train_ex = all_ex[split:]

        return train_ex, eval_ex

    def write(self, train: list[dict], eval: list[dict]) -> dict[str, Path]:
        """Write train/eval JSONL files to output_dir. Returns paths."""
        paths = {}
        for name, data in [("train.jsonl", train), ("eval.jsonl", eval)]:
            path = self._output_dir / name
            with open(path, "w", encoding="utf-8") as f:
                for ex in data:
                    f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")
            paths[name] = path
            logger.info("Wrote %d examples → %s", len(data), path)
        return paths

    # ── Internals ──────────────────────────────────────────────────────

    def _make_args(self, tool: str, field: str) -> dict[str, Any]:
        if tool in ("trace_dependencies", "inspect_lineage"):
            return {"target": field, "session_id": SESSION}
        elif tool == "get_field_definition":
            return {"field": field}
        elif tool in ("search_docs", "search_regulation"):
            return {"query": field}
        return {}

    def _make_single_answer(self, tool: str, field: str, result: dict) -> str:
        if tool == "trace_dependencies" and result.get("found"):
            layers = result.get("layers", [])
            fields = set()
            for layer in layers:
                if isinstance(layer, dict):
                    fields.update(layer.get("fields", []))
            return f"**{field}** depende de: {', '.join(sorted(fields))}."
        elif tool == "inspect_lineage" and result.get("found"):
            expr = result.get("expression") or ""
            if not expr:
                layers = result.get("layers", [])
                if layers and isinstance(layers[0], dict):
                    fields = layers[0].get("fields", [])
                    expr = ", ".join(str(f) for f in fields)
            return f"**{field}** se calcula en el pipeline SAS. {expr}"[:400]
        elif tool == "get_field_definition":
            defn = result.get("result") or result.get("definition", "")
            if defn:
                return str(defn)[:400]
        elif tool == "search_regulation":
            hits = result.get("results") or result.get("hits") or []
            if hits:
                return f"Regulación relevante: {hits[0].get('title', hits[0].get('text', ''))[:300]}"
        elif tool == "search_docs":
            hits = result.get("hits", [])
            if hits:
                return f"Documentación encontrada: {hits[0].get('section', hits[0].get('text', ''))[:300]}"
        return f"Consulté las herramientas sobre **{field}**."

    def _make_case_answer(
        self, template: dict, field: str, steps: list[tuple[str, dict]]
    ) -> str:
        parts: dict[str, list[str]] = {"deps": [], "reg": [], "defn": [], "expr": [], "fields": []}
        for tool_name, args in steps:
            result = dispatch_tool(tool_name, args)
            if tool_name == "trace_dependencies" and result.get("found"):
                layers = result.get("layers", [])
                for layer in layers:
                    if isinstance(layer, dict):
                        parts["deps"].append(f"Nivel {layer.get('depth', '?')}: " + ", ".join(layer.get("fields", [])))
                edges = result.get("edges", [])
                for edge in edges[:4]:
                    if isinstance(edge, dict):
                        parts["fields"].append(str(edge.get("source", "")))
            elif tool_name == "inspect_lineage" and result.get("found"):
                expr = result.get("expression") or ""
                if expr:
                    parts["expr"].append(str(expr)[:200])
            elif tool_name == "get_field_definition":
                defn = result.get("result") or result.get("definition", "")
                if defn:
                    parts["defn"].append(str(defn)[:300])
            elif tool_name == "search_regulation":
                hits = result.get("results") or result.get("hits") or []
                for h in hits[:2]:
                    parts["reg"].append(str(h.get("title", h.get("text", "")))[:200])

        subs: dict[str, str] = {
            "field": field,
            "deps": "; ".join(parts["deps"]) or "No se encontraron dependencias.",
            "reg": "\n".join(parts["reg"]) or "No se encontró regulación específica.",
            "defn": " ".join(parts["defn"]) or "No se encontró definición.",
            "expr": "; ".join(parts["expr"]) or "Expresión disponible en el pipeline.",
            "fields": ", ".join(set(parts["fields"])) or "Varios.",
        }
        return template["answer_template"].format(**subs)[:600]

    def _format_trace(
        self,
        question: str,
        steps: list[tuple[str, dict]],
        final_answer: str,
    ) -> dict:
        """Build a chat-format conversation with tool calls."""
        messages: list[dict] = [
            {"role": "system", "content": _LITE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        for i, (tool_name, args) in enumerate(steps):
            call_id = f"call_{abs(hash((question, i, tool_name))) % 100000:05d}"
            result = dispatch_tool(tool_name, args)
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": call_id, "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }],
            })
            messages.append({
                "role": "tool", "tool_call_id": call_id, "name": tool_name,
                "content": json.dumps(result, ensure_ascii=False, default=str)[:4000],
            })
        messages.append({"role": "assistant", "content": final_answer})
        return {"messages": messages}

"""Tool-calling agent loop for the SAS-diff Q&A.

Given a natural-language question, drives a conversation with the local
LLM where the LLM may call tools from :mod:`src.agent.tools`. Yields
:class:`AgentEvent` records that the API streams to the UI as SSE.

Stops when the LLM returns a plain-text answer with no tool calls, or
when ``max_iters`` is reached, or when the same tool+args is repeated
twice in a row (loop guard).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from src.knowledge import LocalLLMClient, get_client

from .tools import TOOL_REGISTRY, dispatch_tool, tool_schemas
from .prompts import LITE_SYSTEM_PROMPT as _LITE_SYSTEM_PROMPT
from .formula_guard import check_formula_grounding

logger = logging.getLogger(__name__)

# Lite tool set for 4B SFT models (matches training/tool_utils.py)
LITE_TOOL_NAMES = [
    "trace_causal_chain", "query_cycle_data", "get_sas_formula",
    "trace_dependencies", "inspect_lineage",
    "search_docs", "search_regulation", "get_field_definition",
]

_TOOL_CALL_TAG = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL,
)
# Qwen2.5 via Ollama: _icall(\n{...},\n{}\n) — not matched by a single-line regex
_TOOL_PAYLOAD = re.compile(
    r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{(?:[^{}]|\[[^\]]*\])*\})\s*\}',
    re.DOTALL,
)
_ICALL_TAG = re.compile(r"_icall\s*\(", re.I)


def _parse_text_tool_calls(text: str) -> list[dict[str, Any]]:
    """Fallback when Ollama puts tool calls in message content instead of tool_calls."""
    calls: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    idx = 0

    def _add(name: str, args: dict[str, Any]) -> None:
        nonlocal idx
        key = (name, json.dumps(args, sort_keys=True, default=str))
        if not name or key in seen:
            return
        seen.add(key)
        calls.append({"id": f"text_call_{idx}", "name": name, "arguments": args})
        idx += 1

    for m in _TOOL_CALL_TAG.finditer(text or ""):
        try:
            payload = json.loads(m.group(1))
            _add(payload.get("name", ""), payload.get("arguments") or {})
        except json.JSONDecodeError:
            continue

    for m in _TOOL_PAYLOAD.finditer(text or ""):
        try:
            _add(m.group(1), json.loads(m.group(2)))
        except json.JSONDecodeError:
            continue

    return calls


def _text_is_unexecuted_icall(text: str) -> bool:
    """True when the model wrote _icall(...) in prose instead of using tool_calls."""
    if not text or not _ICALL_TAG.search(text):
        return False
    return bool(_TOOL_PAYLOAD.search(text))


_FAKE_TURN = re.compile(
    r"<\|im_start\|>|</tool_response>|<tool_response>|role\s*:\s*user",
    re.I,
)


def _looks_like_fake_turn(text: str) -> bool:
    """GRPO/SFT artefact: model writes tool_response blocks instead of real tools."""
    if not text:
        return False
    if _FAKE_TURN.search(text):
        return True
    if "<tool_response>" in text and not _TOOL_CALL_TAG.search(text):
        return True
    return False


def _strip_template_junk(text: str) -> str:
    t = text or ""
    t = re.sub(r"<\|im_start\|>.*?(?:<\|im_end\|>|$)", "", t, flags=re.DOTALL)
    t = re.sub(r"<tool_response>\s*\{.*?\}\s*</tool_response>", "", t, flags=re.DOTALL | re.I)
    t = re.sub(r"<tool_response>.*?</tool_response>", "", t, flags=re.DOTALL | re.I)
    t = re.sub(r"<tool_call>\s*\{.*?\}\s*</tool_call>", "", t, flags=re.DOTALL)
    return t.strip()


# ── Events streamed to the UI ────────────────────────────────────────────────


@dataclass
class AgentEvent:
    type: str                              # "thought" | "tool_call" | "tool_result" | "final" | "error" | "status"
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "ts": self.ts, **self.payload}


# ── Prompt ───────────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = """\
Eres un investigador de cumplimiento regulatorio para bases de datos de reporting
bancario (COREP/FINREP IRB). Tienes acceso a un grafo de conocimiento que enlaza
normativa EBA/BdE, columnas de base de datos, reglas de validación, lineage de
código SAS e insights de investigaciones previas. Tu trabajo es tomar un problema
descrito en lenguaje natural, investigarlo metódicamente y aprender del feedback.

IMPORTANTE: Responde SIEMPRE en español — pensamientos, planes, respuestas finales.

SAS code is uploaded per session. Always pass the current session_id
when calling SAS tools (inspect_lineage, trace_dependencies,
backtrace_sas_field, validate_sas, describe_egp_project, compute_shapley).

## Glosario de campos (nombres exactos en el pipeline)

- **MoC** = Margen de Conservadurismo (NO traduzcas como "Mooring Cost"). Campo: `MoC`.
- **ECL** = Expected Credit Loss. **LGD_ESTIMADA**, **SW_FUSION**, **PD_ESTIMADA**, **EAD**.
- Usa los nombres de campo EXACTOS del pipeline (MoC, no MO_C).

## Fase 1 — ENTENDER Y PLANIFICAR

Siempre empieza aquí. Antes de analizar:

1. Llama `search_experience(query)` solo si la pregunta es sobre conocimiento previo
   guardado — NO para investigar bugs, lineage o campos missing en el pipeline.
2. Llama `backtrace_sas_field(target, session_id)` para trazar dependencias SAS.
3. Llama `find_governing_rules(column)` para restricciones regulatorias.
4. Formula un plan y llama `create_investigation_plan(target_field, problem, steps)`.

## Fase 2 — REQUISITOS DE DATOS

Identifica qué datos necesita el usuario:

1. Llama `formulate_data_request(extractions, reason)`.
2. PARA y explica qué datos necesitas y por qué.

## Fase 3 — ANALIZAR

Cuando haya datos disponibles:

1. `analyze_uploaded_data(file_path)` si hay archivos subidos.
2. `compute_shapley(pk, target, session_id)` para atribución Shapley.
3. `trace_dependencies(target, session_id)` para la cadena completa con expresiones.
4. Cruza con normativa: `query_regulation`, `trace_regulation_chain`, `search_docs`.

## Fase 4 — CONCLUSIÓN Y APRENDIZAJE

1. Escribe el análisis final con citas regulatorias.
2. `save_insight` si descubres algo útil.
3. `save_feedback` si el usuario te corrige.

## Formato de respuesta

Responde en Markdown en español, con un bloque JSON final:

```json
{
  "lineage_highlight": ["CAMPO_A", "CAMPO_B", "TARGET"],
  "citations": [
    {"kind": "doc",        "path": "fields/X.md",   "heading": "Definición"},
    {"kind": "regulation", "article": "art15",       "heading": "Suelos LGD"},
    {"kind": "code",       "step": "work.foo"}
  ]
}
```

Restricciones: prefiere llamar más herramientas antes que alucinar valores.
Los resultados de herramientas son autoritativos — nunca los contradigas.
NUNCA inventes fórmulas (ECL, MoC, OR_EADIL, etc.) que no aparezcan en tool results.
Para "missing", "investiga" o "causa raíz": trace_dependencies e inspect_lineage son obligatorios.
"""


_PK_RE = re.compile(r"\b(CIC_\d{4,}|[A-Z]{2,}_\d{4,})\b")


def _seed_user_message(question: str) -> str:
    """Pre-parse what we can deterministically — saves the LLM a turn."""
    extras: list[str] = []
    pks = _PK_RE.findall(question.upper())
    if pks:
        extras.append(f"(detected primary key: {pks[0]})")
    return question.strip() + (" " + " ".join(extras) if extras else "")


# ── Agent ────────────────────────────────────────────────────────────────────


class SASDiffAgent:
    def __init__(
        self,
        client: LocalLLMClient | None = None,
        *,
        max_iters: int = 15,
        temperature: float = 0.1,
        session_id: str | None = None,
        sas_session_id: str | None = "debug_lgd",
        model: str | None = None,
        tool_names: list[str] | None = None,
    ) -> None:
        self.model_override = model
        if model and model.startswith("peft:"):
            import os
            peft_url = os.getenv("REGLLM_PEFT_URL", "http://127.0.0.1:8767/v1")
            self.client = LocalLLMClient(prefer="litert", litert_url=peft_url, litert_model="peft")
        elif model:
            self.client = LocalLLMClient(ollama_model=model)
        else:
            self.client = client or get_client()
        self.max_iters = max_iters
        self.temperature = temperature
        self.session_id = session_id
        self.sas_session_id = sas_session_id or "debug_lgd"
        self.tool_names = tool_names

    async def run(self, question: str) -> AsyncIterator[AgentEvent]:
        backend = self.client.detect_backend()
        active_tools = self.tool_names or list(TOOL_REGISTRY.keys())
        yield AgentEvent("status", {
            "stage": "started",
            "backend": backend,
            "model": (
                "peft" if (self.model_override and str(self.model_override).startswith("peft:"))
                else self.client.litert_model if backend == "litert"
                else self.client.ollama_model if backend == "ollama"
                else "stub"
            ),
            "tools": active_tools,
            "question": question,
        })

        system_prompt = _LITE_SYSTEM_PROMPT if self.tool_names else _SYSTEM_PROMPT
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _seed_user_message(question)},
        ]
        tools = tool_schemas(self.tool_names)
        last_call_signature: str | None = None
        repeat_count = 0
        formula_retries = 0

        for it in range(self.max_iters):
            try:
                resp = await asyncio.to_thread(
                    self.client.chat_tools,
                    messages, tools,
                    temperature=self.temperature,
                    max_tokens=4096,
                )
            except Exception as e:
                logger.exception("LLM call failed")
                yield AgentEvent("error", {"stage": "llm_call", "error": str(e)})
                return

            calls = resp.tool_calls or []
            if not calls and resp.text:
                calls = _parse_text_tool_calls(resp.text)

            if not calls and _looks_like_fake_turn(resp.text or ""):
                yield AgentEvent("thought", {
                    "iter": it,
                    "text": "(modelo simuló tool_response — reintentando con herramientas reales)",
                })
                messages.append({"role": "assistant", "content": resp.text or ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "ERROR: No simules bloques <tool_response> ni roles user/assistant. "
                        "Invoca herramientas reales (trace_dependencies, inspect_lineage) "
                        "o responde en español con evidencia de tool results."
                    ),
                })
                continue

            if calls:
                thinking_text = (resp.text or "").strip()
                if _text_is_unexecuted_icall(thinking_text):
                    thinking_text = ""
                elif _ICALL_TAG.search(thinking_text):
                    thinking_text = _TOOL_PAYLOAD.sub("", thinking_text)
                    thinking_text = re.sub(r"_icall\s*\([^)]*\)", "", thinking_text, flags=re.I)
                    thinking_text = thinking_text.strip()
                if thinking_text:
                    yield AgentEvent("thought", {"iter": it, "text": thinking_text})

                # Track the LLM's intent in the conversation. Ollama's
                # /api/chat parses `function.arguments` as an object (not a
                # JSON-encoded string), so we mirror that shape both ways.
                messages.append({
                    "role": "assistant",
                    "content": resp.text or "",
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": c["arguments"],
                            },
                        }
                        for c in calls
                    ],
                })
                # Loop guard
                sig = json.dumps([(c["name"], c["arguments"]) for c in calls], sort_keys=True, default=str)
                if sig == last_call_signature:
                    repeat_count += 1
                else:
                    repeat_count = 0
                last_call_signature = sig
                if repeat_count >= 1:
                    yield AgentEvent("status", {"stage": "loop_detected", "iter": it})
                    break

                for call in calls:
                    name = call["name"]
                    args = call["arguments"] or {}
                    # Force SAS session — ignore hallucinated session_id from the LLM
                    if self.sas_session_id:
                        spec = TOOL_REGISTRY.get(name)
                        if spec and "session_id" in (
                            spec.schema()["function"]["parameters"].get("properties", {})
                        ):
                            args["session_id"] = self.sas_session_id
                    yield AgentEvent("tool_call", {"iter": it, "tool": name, "args": args, "id": call["id"]})
                    result = await asyncio.to_thread(dispatch_tool, name, args)
                    yield AgentEvent("tool_result", {
                        "iter": it, "tool": name, "id": call["id"],
                        "result": result,
                    })
                    # Emit semantic events for plan/data_request tools
                    if name == "create_investigation_plan" and isinstance(result, dict) and result.get("saved"):
                        yield AgentEvent("plan", {"plan": result.get("plan", ""), "id": result.get("id", "")})
                    elif name == "formulate_data_request" and isinstance(result, dict) and result.get("data_request"):
                        yield AgentEvent("data_request", {
                            "extractions": result.get("extractions", []),
                            "reason": result.get("reason", ""),
                        })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": json.dumps(result, default=str)[:8000],
                    })
                continue

            # No tool calls → final answer
            text = _strip_template_junk(resp.text or "")
            if _text_is_unexecuted_icall(text):
                messages.append({"role": "assistant", "content": resp.text or ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "No escribas _icall(...) en texto. Usa function calling del API "
                        "o una sola herramienta por turno. Luego responde citando rows."
                    ),
                })
                continue
            if not text or len(text) < 20:
                messages.append({"role": "assistant", "content": resp.text or ""})
                messages.append({
                    "role": "user",
                    "content": "Responde en español con tu análisis. Usa herramientas si aún no lo hiciste.",
                })
                continue

            violations = check_formula_grounding(text, messages)
            if violations and formula_retries < 2:
                formula_retries += 1
                yield AgentEvent("thought", {
                    "iter": it,
                    "text": (
                        f"(verificador: fórmulas no ancladas — reintento {formula_retries}/2)"
                    ),
                })
                messages.append({"role": "assistant", "content": text})
                bullet = "\n".join(f"- {v}" for v in violations[:6])
                messages.append({
                    "role": "user",
                    "content": (
                        "ERROR: Tu respuesta cita fórmulas o archivos no verificados en los "
                        "resultados de herramientas:\n"
                        f"{bullet}\n\n"
                        "Llama get_sas_formula(field=...) o trace_dependencies(target=...) "
                        "y copia el campo expr literalmente. Si no hay expr, di que no consta "
                        "en el SAS de esta sesión. Reescribe la respuesta."
                    ),
                })
                continue

            messages.append({"role": "assistant", "content": text})
            parsed = _parse_final(text)
            yield AgentEvent("final", {
                "iter": it,
                "answer": parsed["prose"],
                "answer_raw": text,
                "lineage_highlight": parsed["lineage_highlight"],
                "citations": parsed["citations"],
            })
            # Fire-and-forget: grow the KG from this session
            asyncio.create_task(self._reflect(messages))
            return

        # max_iters exhausted
        yield AgentEvent("status", {"stage": "max_iters_reached", "iters": self.max_iters})
        # Ask for a final answer one more time, without tools
        try:
            messages.append({
                "role": "user",
                "content": "Escribe ahora la respuesta final en markdown en español, con el bloque JSON de cierre.",
            })
            final_resp = await asyncio.to_thread(
                self.client.chat,
                messages,
                temperature=self.temperature,
                max_tokens=4096,
            )
            text = (final_resp.text or "").strip()
            parsed = _parse_final(text)
            yield AgentEvent("final", {
                "iter": self.max_iters,
                "answer": parsed["prose"],
                "answer_raw": text,
                "lineage_highlight": parsed["lineage_highlight"],
                "citations": parsed["citations"],
            })
        except Exception as e:
            logger.exception("Final synthesis failed")
            yield AgentEvent("error", {"stage": "final_synthesis", "error": str(e)})

    async def _reflect(self, messages: list[dict[str, Any]]) -> None:
        """Background reflection: extract insights from this session into the KG."""
        try:
            from src.knowledge.graph_builder import get_builder
            builder = get_builder()

            # Build a compact session log from the conversation
            log_parts: list[str] = []
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                if role == "system":
                    continue
                if role == "tool":
                    log_parts.append(f"[tool:{m.get('name','')}] {content[:300]}")
                elif content:
                    log_parts.append(f"[{role}] {content[:500]}")
            session_log = "\n".join(log_parts)[:6000]

            await asyncio.to_thread(builder.reflect_on_session, session_log)
        except Exception:
            logger.debug("Session reflection failed (non-critical)", exc_info=True)


# ── Final-answer parsing ─────────────────────────────────────────────────────


_JSON_FENCE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BRACE_BLOCK = re.compile(r"(\{(?:[^{}]|(?:\{[^{}]*\}))*\})", re.DOTALL)


def _parse_final(text: str) -> dict[str, Any]:
    """Extract the JSON sidecar (lineage_highlight, citations) from the
    LLM's final response. Falls back to empty values if absent."""
    out: dict[str, Any] = {"prose": text, "lineage_highlight": [], "citations": []}
    if not text:
        return out
    m = _JSON_FENCE.search(text)
    raw_json: str | None = None
    if m:
        raw_json = m.group(1)
    else:
        # last brace-balanced block at the end
        candidates = _BRACE_BLOCK.findall(text[-2000:])
        if candidates:
            raw_json = candidates[-1]
    if not raw_json:
        return out
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return out
    if isinstance(data, dict):
        lh = data.get("lineage_highlight") or []
        if isinstance(lh, list):
            out["lineage_highlight"] = [str(x).upper() for x in lh if isinstance(x, (str, int))]
        cit = data.get("citations") or []
        if isinstance(cit, list):
            out["citations"] = [c for c in cit if isinstance(c, dict)]
    # Strip the JSON tail from the prose for nicer display
    if m:
        out["prose"] = (text[:m.start()] + text[m.end():]).strip()
    return out


# ── Convenience coroutine ───────────────────────────────────────────────────


async def run_agent(question: str, **kwargs: Any) -> AsyncIterator[AgentEvent]:
    agent = SASDiffAgent(**kwargs)
    async for ev in agent.run(question):
        yield ev

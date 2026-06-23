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

logger = logging.getLogger(__name__)


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
You are a regulatory compliance investigator for banking reporting
databases (COREP/FINREP IRB). You have access to a knowledge graph
linking EBA/BdE regulations, database columns, validation rules, SAS
code lineage, and insights from past investigations. Your job is to
take a problem described in natural language, investigate it
methodically, and learn from the user's feedback.

## Phase 1 — UNDERSTAND & PLAN

Always start here. Before doing any analysis:

1. Call `search_experience(query)` to check for prior investigations
   on this topic. Feedback insights (priority=1.0) are most important.
2. Call `backtrace_sas_field(target)` to trace the target field's
   dependency tree through the SAS code.
3. Call `find_governing_rules(column)` for regulatory constraints on
   the target field.
4. Based on what you learn, formulate an investigation plan and call
   `create_investigation_plan(target_field, problem, steps)` to
   persist it. Stream your plan to the user as you go.

## Phase 2 — DATA REQUIREMENTS

From the dependency trace, identify what input data the user needs
to provide for you to analyze the case:

1. Call `formulate_data_request(extractions, reason)` specifying the
   exact tables, fields, filters, and cycles needed.
2. STOP and explain to the user what data you need and why. Wait for
   them to upload it.

If the user has already uploaded data or sample CSVs are available,
skip to Phase 3.

## Phase 3 — ANALYZE

Once data is available:

1. If the user uploaded files, call `analyze_uploaded_data(file_path)`
   to load and inspect the data.
2. Call `compute_attribution(pk, target)` for gradient + Shapley
   attribution on specific cycles.
3. Call `trace_dependencies(target)` for the full calculation chain
   with expressions.
4. Call `compare_sas_versions(target)` to see V2 vs V3 code changes.
5. Cross-reference findings against regulation using
   `query_regulation`, `trace_regulation_chain`, and `search_docs`.

## Phase 4 — CONCLUDE & LEARN

1. Write your final analysis with regulatory citations.
2. If you discovered something useful, call `save_insight` to store
   it for future sessions.
3. If the user corrects you or confirms a non-obvious finding, call
   `save_feedback(feedback_type, content, original_claim,
   corrected_understanding)` immediately.

## Handling User Feedback

When the user says things like "no, that's wrong", "actually the issue
is...", or "yes, exactly":
- Call `save_feedback` with the appropriate type (correction,
  confirmation, or clarification).
- Adjust your analysis accordingly.
- These feedback insights have priority=1.0 and will be recalled
  first in future investigations.

## Answer Format

Reply with a Markdown answer that includes a final JSON block:

```json
{
  "lineage_highlight": ["FIELD_A", "FIELD_B", "TARGET"],
  "citations": [
    {"kind": "doc",        "path": "fields/X.md",   "heading": "Definition"},
    {"kind": "regulation", "article": "art15",       "heading": "LGD floors"},
    {"kind": "code",       "step": "work.foo",       "version": "v3"},
    {"kind": "changelog",  "path": "2025-q1-...",    "heading": "..."}
  ]
}
```

The JSON is parsed by the UI to highlight the graph and render
citation chips, so emit it even if some lists are empty.

Constraints: prefer calling more tools over hallucinating values.
Tool results are authoritative — never override them.
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
    ) -> None:
        self.client = client or get_client()
        self.max_iters = max_iters
        self.temperature = temperature
        self.session_id = session_id

    async def run(self, question: str) -> AsyncIterator[AgentEvent]:
        backend = self.client.detect_backend()
        yield AgentEvent("status", {
            "stage": "started",
            "backend": backend,
            "model": (
                self.client.litert_model if backend == "litert"
                else self.client.ollama_model if backend == "ollama"
                else "stub"
            ),
            "tools": list(TOOL_REGISTRY.keys()),
            "question": question,
        })

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _seed_user_message(question)},
        ]
        tools = tool_schemas()
        last_call_signature: str | None = None
        repeat_count = 0

        for it in range(self.max_iters):
            try:
                resp = await asyncio.to_thread(
                    self.client.chat_tools,
                    messages, tools,
                    temperature=self.temperature,
                    max_tokens=2048,
                )
            except Exception as e:
                logger.exception("LLM call failed")
                yield AgentEvent("error", {"stage": "llm_call", "error": str(e)})
                return

            calls = resp.tool_calls or []
            if calls:
                # Emit the LLM's reasoning text as a "thought" event
                thinking_text = (resp.text or "").strip()
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
            text = (resp.text or "").strip()
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
                "content": "Please now write the final markdown answer with the closing JSON block.",
            })
            final_resp = await asyncio.to_thread(
                self.client.chat,
                messages,
                temperature=self.temperature,
                max_tokens=2048,
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

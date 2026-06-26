"""Integration test for the SASDiffAgent tool-calling loop.

Uses a deterministic fake LLM that scripts the tool-call sequence — so we
exercise the dispatcher, message accumulation, loop guard, final-answer
parsing, and event stream without needing a live model.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.agent import SASDiffAgent
from src.knowledge.llm_client import ChatResponse, LocalLLMClient


class _ScriptedLLM(LocalLLMClient):
    """An LLM whose responses are a hand-written script keyed by call count."""

    def __init__(self, script: list[ChatResponse]) -> None:
        super().__init__(prefer="stub")
        self.script = list(script)
        self.calls = 0

    def detect_backend(self) -> str:  # type: ignore[override]
        return "ollama"  # claim a real backend so the agent uses chat_tools

    def chat_tools(  # type: ignore[override]
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        if self.calls >= len(self.script):
            # Default: end with a plain answer
            return ChatResponse(
                text="```json\n{\"lineage_highlight\": [], \"citations\": []}\n```",
                backend="ollama", model="scripted", tool_calls=None,
            )
        resp = self.script[self.calls]
        self.calls += 1
        return resp


def _collect(agent: SASDiffAgent, question: str) -> list[dict[str, Any]]:
    async def _go() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        async for ev in agent.run(question):
            out.append(ev.to_dict())
        return out
    return asyncio.run(_go())


def test_loop_terminates_on_plain_answer() -> None:
    script = [
        ChatResponse(
            text="The answer is 42.",
            backend="ollama", model="scripted", tool_calls=None,
        ),
    ]
    agent = SASDiffAgent(client=_ScriptedLLM(script), max_iters=4)
    events = _collect(agent, "What's the meaning?")
    assert events[0]["type"] == "status"
    final = next(e for e in events if e["type"] == "final")
    assert final["answer"] == "The answer is 42."


def test_dispatches_tool_then_finalises() -> None:
    script = [
        ChatResponse(
            text="",
            backend="ollama", model="scripted",
            tool_calls=[{
                "id": "c0", "name": "inspect_lineage",
                "arguments": {"target": "ECL", "session_id": "default"},
            }],
        ),
        ChatResponse(
            text=(
                "ECL depends on PD, LGD, EAD.\n"
                "```json\n"
                "{\"lineage_highlight\": [\"PD_ESTIMADA\", \"LGD_ESTIMADA\", \"EAD\"], "
                "\"citations\": [{\"kind\": \"code\", \"step\": \"work.ecl_calculo\"}]}"
                "\n```"
            ),
            backend="ollama", model="scripted", tool_calls=None,
        ),
    ]
    agent = SASDiffAgent(client=_ScriptedLLM(script), max_iters=4)
    events = _collect(agent, "Where does ECL come from?")
    types = [e["type"] for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    assert types[-1] == "final"

    tool_call = next(e for e in events if e["type"] == "tool_call")
    assert tool_call["tool"] == "inspect_lineage"

    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert "ancestors" in tool_result["result"]

    final = next(e for e in events if e["type"] == "final")
    assert "PD_ESTIMADA" in final["lineage_highlight"]
    assert final["citations"] and final["citations"][0]["kind"] == "code"


def test_loop_guard_breaks_on_repeated_call() -> None:
    same_call = ChatResponse(
        text="",
        backend="ollama", model="scripted",
        tool_calls=[{
            "id": "c", "name": "inspect_lineage",
            "arguments": {"target": "ECL", "session_id": "default"},
        }],
    )
    # Repeat the same tool call indefinitely — agent should detect the loop
    agent = SASDiffAgent(client=_ScriptedLLM([same_call, same_call, same_call]), max_iters=8)
    events = _collect(agent, "loop me")
    statuses = [e for e in events if e["type"] == "status"]
    assert any(s.get("stage") == "loop_detected" for s in statuses)


def test_max_iters_triggers_synthesis() -> None:
    # 5 distinct tool calls, never terminates → max_iters reached after 3
    targets = ["ECL", "LGD_ESTIMADA", "PD_ESTIMADA", "EAD", "RWA"]
    distinct = [
        ChatResponse(
            text="",
            backend="ollama", model="scripted",
            tool_calls=[{
                "id": f"c{i}", "name": "inspect_lineage",
                "arguments": {"target": t, "session_id": "default"},
            }],
        )
        for i, t in enumerate(targets)
    ]
    agent = SASDiffAgent(client=_ScriptedLLM(distinct), max_iters=3)
    events = _collect(agent, "exhaust me")
    statuses = [e for e in events if e["type"] == "status"]
    # We at least see the cap message
    assert any(
        s.get("stage") in ("max_iters_reached", "loop_detected")
        for s in statuses
    )


@pytest.mark.parametrize("question,expected_pk", [
    ("Why does CIC_00031 differ?", "CIC_00031"),
    ("Investigate cycle CIC_99 please", None),  # too short, won't match
])
def test_seed_user_message_extracts_pk(question: str, expected_pk: str | None) -> None:
    from src.agent.agent import _seed_user_message
    seeded = _seed_user_message(question)
    if expected_pk:
        assert expected_pk in seeded
        assert "detected primary key" in seeded
    else:
        assert "detected primary key" not in seeded

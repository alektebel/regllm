"""Tests for the Amazon Bedrock backend (Converse API) in
``src.knowledge.llm_client``.

No real AWS credentials or network access needed — ``LocalLLMClient``'s
lazily-created boto3 bedrock-runtime client (``self._bedrock_client``) is
replaced with a fake that mimics the ``converse``/``converse_stream``
surface the client actually calls. This exercises the real dispatch/parsing
code paths (``detect_backend``, ``chat``, ``chat_json``, ``chat_json_stream``,
``chat_tools``) without needing network access or IAM credentials in CI.
"""

from __future__ import annotations

import pytest

from src.knowledge import llm_client as llm_client_module
from src.knowledge.llm_client import ChatResponse, LocalLLMClient


class _FakeBedrockClient:
    """Stands in for a boto3 ``bedrock-runtime`` client."""

    def __init__(self):
        self.converse_calls: list[dict] = []
        self.converse_stream_calls: list[dict] = []

    def converse(self, **kwargs):
        self.converse_calls.append(kwargs)
        if kwargs.get("toolConfig"):
            return {
                "output": {
                    "message": {
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "call_0",
                                    "name": "inspect_lineage",
                                    "input": {"target": "ECL"},
                                }
                            },
                        ],
                    },
                },
            }
        return {"output": {"message": {"content": [{"text": "hola desde bedrock"}]}}}

    def converse_stream(self, **kwargs):
        self.converse_stream_calls.append(kwargs)
        events = [
            {"contentBlockDelta": {"delta": {"text": '{"d'}}},
            {"contentBlockDelta": {"delta": {"text": 'qcs": '}}},
            {"contentBlockDelta": {"delta": {"text": "[]}"}}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
        return {"stream": iter(events)}


def _make_client(fake_client: "_FakeBedrockClient | None" = None) -> LocalLLMClient:
    c = LocalLLMClient(prefer="bedrock")
    c._bedrock_client = fake_client or _FakeBedrockClient()
    return c


# ── detection ────────────────────────────────────────────────────────────

def test_detect_backend_bedrock_when_boto3_present():
    c = LocalLLMClient(prefer="bedrock")
    assert c.detect_backend() == "bedrock"


def test_detect_backend_falls_back_to_stub_when_boto3_missing(monkeypatch):
    monkeypatch.setattr(llm_client_module, "_boto3", None)
    c = LocalLLMClient(prefer="bedrock")
    assert c.detect_backend() == "stub"


# ── chat / chat_json / streaming / tools ────────────────────────────────────

def test_chat_returns_bedrock_backend():
    c = _make_client()
    resp = c.chat([{"role": "user", "content": "hola"}])
    assert isinstance(resp, ChatResponse)
    assert resp.backend == "bedrock"
    assert resp.text == "hola desde bedrock"


def test_chat_json_mode_appends_json_instruction_to_system():
    fake = _FakeBedrockClient()
    c = _make_client(fake)
    c.chat(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        json_mode=True,
    )
    kwargs = fake.converse_calls[0]
    assert "Respond ONLY with valid JSON." in kwargs["system"][0]["text"]


def test_chat_json_stream_yields_tokens_then_parsed_json():
    c = _make_client()
    events = list(c.chat_json_stream(system="sys", user="user"))
    *tokens, final = events
    assert tokens == ['{"d', 'qcs": ', "[]}"]
    assert final == {"dqcs": []}


def test_chat_tools_parses_tool_use_blocks():
    c = _make_client()
    resp = c.chat_tools(
        messages=[{"role": "user", "content": "why?"}],
        tools=[{"type": "function", "function": {"name": "inspect_lineage", "parameters": {}}}],
    )
    assert resp.backend == "bedrock"
    assert resp.tool_calls == [
        {"id": "call_0", "name": "inspect_lineage", "arguments": {"target": "ECL"}},
    ]


def test_chat_tools_converts_tool_call_history_to_bedrock_shape():
    fake = _FakeBedrockClient()
    c = _make_client(fake)
    c.chat_tools(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "why?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "inspect_lineage", "arguments": {"target": "ECL"}},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "name": "inspect_lineage",
                "content": '{"ok": true}',
            },
        ],
        tools=[{"type": "function", "function": {"name": "inspect_lineage", "parameters": {}}}],
    )
    kwargs = fake.converse_calls[0]
    assert kwargs["system"] == [{"text": "sys"}]
    msgs = kwargs["messages"]
    assert msgs[0] == {"role": "user", "content": [{"text": "why?"}]}
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"][0]["toolUse"] == {
        "toolUseId": "call_0", "name": "inspect_lineage", "input": {"target": "ECL"},
    }
    assert msgs[2] == {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "call_0", "content": [{"text": '{"ok": true}'}]}}],
    }

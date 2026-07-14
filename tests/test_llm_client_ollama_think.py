"""Tests for thinking-model handling on the Ollama backend.

Thinking models (qwen3*, deepseek-r1) must be sent Ollama's native
``think: false`` field so no hidden reasoning is generated — critically in
JSON mode, where the old let-it-think-and-strip approach burned minutes per
call and could exhaust the token budget (empty output). Older Ollama
servers (< 0.9) reject the field with a 400, so the client falls back to
the previous behaviour.

No Ollama server is needed — ``httpx.post`` is monkeypatched.
"""

from __future__ import annotations

import httpx
import pytest

from src.knowledge import llm_client as llm_client_module
from src.knowledge.llm_client import LocalLLMClient


class _FakeResponse:
    def __init__(self, status_code=200, content="{}"):
        self.status_code = status_code
        self._content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=None, response=self)

    def json(self):
        return {"message": {"content": self._content}}


def _client(model: str) -> LocalLLMClient:
    client = LocalLLMClient(ollama_model=model, prefer="ollama")
    client._backend = "ollama"
    client._probed = True
    return client


def _capture_posts(monkeypatch, responses):
    """Monkeypatch httpx.post to pop canned responses, recording payloads."""
    calls: list[dict] = []

    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        return responses.pop(0)

    monkeypatch.setattr(llm_client_module.httpx, "post", fake_post)
    return calls


def test_thinking_model_json_mode_sends_think_false(monkeypatch):
    calls = _capture_posts(monkeypatch, [_FakeResponse(content='{"ok": 1}')])
    result = _client("qwen3:8b").chat_json(system="s", user="u", max_tokens=512)
    assert result == {"ok": 1}
    payload = calls[0]
    assert payload["think"] is False
    assert payload["format"] == "json"
    # no +4096 thinking budget when thinking is disabled natively
    assert payload["options"]["num_predict"] == 512


def test_thinking_model_plain_mode_sends_think_false(monkeypatch):
    calls = _capture_posts(monkeypatch, [_FakeResponse(content="hola")])
    resp = _client("qwen3:8b").chat([{"role": "user", "content": "hola?"}])
    assert resp.text == "hola"
    payload = calls[0]
    assert payload["think"] is False
    # native field replaces the /no_think prompt hack
    assert all("/no_think" not in m["content"] for m in payload["messages"])


def test_non_thinking_model_omits_think_field(monkeypatch):
    calls = _capture_posts(monkeypatch, [_FakeResponse(content='{"ok": 1}')])
    _client("qwen2.5:7b-instruct").chat_json(system="s", user="u")
    assert "think" not in calls[0]


def test_old_ollama_400_falls_back_with_think_budget(monkeypatch):
    calls = _capture_posts(monkeypatch, [
        _FakeResponse(status_code=400),
        _FakeResponse(content='<think>razonando…</think>{"ok": 1}'),
    ])
    result = _client("qwen3:8b").chat_json(system="s", user="u", max_tokens=512)
    assert result == {"ok": 1}
    assert len(calls) == 2
    retry = calls[1]
    assert "think" not in retry
    # old behaviour restored: extra budget for <think> tokens in JSON mode
    assert retry["options"]["num_predict"] == 512 + 4096


def test_old_ollama_400_falls_back_to_no_think_prompt(monkeypatch):
    calls = _capture_posts(monkeypatch, [
        _FakeResponse(status_code=400),
        _FakeResponse(content="hola"),
    ])
    resp = _client("qwen3:8b").chat([{"role": "user", "content": "hola?"}])
    assert resp.text == "hola"
    retry = calls[1]
    assert "think" not in retry
    assert any("/no_think" in m["content"] for m in retry["messages"])


def test_server_error_is_not_swallowed_by_fallback(monkeypatch):
    _capture_posts(monkeypatch, [_FakeResponse(status_code=500)])
    with pytest.raises(httpx.HTTPStatusError):
        _client("qwen3:8b").chat_json(system="s", user="u")

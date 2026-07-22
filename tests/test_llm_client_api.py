"""Tests for the generic HTTP inference-gateway backend (REGLLM_API).

No network — httpx.post is monkeypatched. Covers URL resolution, auth
header selection, OpenAI-shape parsing, json_mode, and backend detection.
"""

from __future__ import annotations

import json

from src.knowledge import llm_client as llm_client_module
from src.knowledge.llm_client import ChatResponse, LocalLLMClient


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _client(**kw) -> LocalLLMClient:
    c = LocalLLMClient(prefer="api")
    c.api_url = kw.get("url", "https://gw.example.com/prod")
    c.api_key = kw.get("key", "")
    c.api_key_header = kw.get("header", "Authorization")
    c.api_model = kw.get("model", "amazon.nova-micro-v1:0")
    return c


def _capture(monkeypatch, payload=None):
    payload = payload or {"choices": [{"message": {"content": "ok"}}]}
    calls: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "payload": json})
        return _FakeResponse(payload)

    monkeypatch.setattr(llm_client_module.httpx, "post", fake_post)
    return calls


def test_detect_backend_api_requires_url():
    assert _client(url="https://gw/prod").detect_backend() == "api"
    c = LocalLLMClient(prefer="api")
    c.api_url = ""
    assert c.detect_backend() == "stub"


def test_bare_base_url_gets_openai_path(monkeypatch):
    calls = _capture(monkeypatch)
    _client(url="https://gw.example.com").chat([{"role": "user", "content": "hi"}])
    assert calls[0]["url"] == "https://gw.example.com/v1/chat/completions"


def test_full_path_url_used_as_is(monkeypatch):
    calls = _capture(monkeypatch)
    _client(url="https://gw.example.com/prod/invoke").chat(
        [{"role": "user", "content": "hi"}])
    assert calls[0]["url"] == "https://gw.example.com/prod/invoke"


def test_bearer_auth_by_default(monkeypatch):
    calls = _capture(monkeypatch)
    _client(key="sekret").chat([{"role": "user", "content": "hi"}])
    assert calls[0]["headers"]["Authorization"] == "Bearer sekret"


def test_api_gateway_key_header(monkeypatch):
    calls = _capture(monkeypatch)
    _client(key="sekret", header="x-api-key").chat(
        [{"role": "user", "content": "hi"}])
    assert calls[0]["headers"]["x-api-key"] == "sekret"
    assert "Authorization" not in calls[0]["headers"]


def test_no_auth_header_when_no_key(monkeypatch):
    calls = _capture(monkeypatch)
    _client().chat([{"role": "user", "content": "hi"}])
    assert "Authorization" not in calls[0]["headers"]
    assert "x-api-key" not in calls[0]["headers"]


def test_chat_json_sets_response_format_and_parses(monkeypatch):
    calls = _capture(monkeypatch,
                     {"choices": [{"message": {"content": json.dumps({"a": 1})}}]})
    out = _client().chat_json(system="s", user="u", max_tokens=128)
    assert out == {"a": 1}
    p = calls[0]["payload"]
    assert p["response_format"] == {"type": "json_object"}
    assert p["model"] == "amazon.nova-micro-v1:0"
    assert p["max_tokens"] == 128


def test_tolerates_non_openai_response_shapes(monkeypatch):
    _capture(monkeypatch, {"output_text": "from a custom gateway"})
    resp = _client().chat([{"role": "user", "content": "hi"}])
    assert isinstance(resp, ChatResponse)
    assert resp.backend == "api"
    assert resp.text == "from a custom gateway"

"""Tests for the Azure OpenAI backend in ``src.knowledge.llm_client``.

No Azure resource needed — ``httpx.post`` is monkeypatched. Exercises the
real dispatch (detect_backend/chat/chat_json) and the request shape Azure
expects (deployment-scoped URL, api-version param, api-key header).
"""

from __future__ import annotations

import json

from src.knowledge import llm_client as llm_client_module
from src.knowledge.llm_client import ChatResponse, LocalLLMClient


class _FakeResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _client(**kwargs) -> LocalLLMClient:
    client = LocalLLMClient(prefer="azure")
    client.azure_endpoint = kwargs.get("endpoint", "https://demo.openai.azure.com")
    client.azure_api_key = kwargs.get("key", "sekret")
    client.azure_deployment = kwargs.get("deployment", "gpt-4o-mini")
    return client


def _capture(monkeypatch, content='{"ok": 1}'):
    calls: list[dict] = []

    def fake_post(url, params=None, headers=None, json=None, timeout=None):
        calls.append({"url": url, "params": params, "headers": headers,
                      "payload": json})
        return _FakeResponse(content)

    monkeypatch.setattr(llm_client_module.httpx, "post", fake_post)
    return calls


def test_detect_backend_azure_with_credentials():
    assert _client().detect_backend() == "azure"


def test_detect_backend_falls_to_stub_without_credentials():
    client = LocalLLMClient(prefer="azure")
    client.azure_endpoint = ""
    client.azure_api_key = ""
    assert client.detect_backend() == "stub"


def test_chat_hits_deployment_url_with_auth(monkeypatch):
    calls = _capture(monkeypatch, content="hola")
    resp = _client().chat([{"role": "user", "content": "hola?"}])
    assert isinstance(resp, ChatResponse)
    assert resp.backend == "azure" and resp.text == "hola"
    call = calls[0]
    assert call["url"] == ("https://demo.openai.azure.com/openai/deployments/"
                           "gpt-4o-mini/chat/completions")
    assert call["params"] == {"api-version": "2024-06-01"}
    assert call["headers"]["api-key"] == "sekret"
    assert "response_format" not in call["payload"]


def test_chat_json_sets_response_format_and_parses(monkeypatch):
    calls = _capture(monkeypatch, content=json.dumps({"plan": []}))
    result = _client().chat_json(system="s", user="u", max_tokens=256)
    assert result == {"plan": []}
    payload = calls[0]["payload"]
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 256
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]

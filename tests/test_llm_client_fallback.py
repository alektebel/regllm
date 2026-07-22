"""GGUF fallback: when the preferred backend is unavailable or a call
fails, fall back to a local GGUF model instead of the stub.

The GGUF backend itself is faked (no llama-cpp / weights needed) by
monkeypatching the probe + the chat method.
"""

from __future__ import annotations

import pytest

from src.knowledge.llm_client import ChatResponse, LocalLLMClient


def _gguf_ready(client: LocalLLMClient, monkeypatch):
    """Make the GGUF backend look available and answer deterministically."""
    monkeypatch.setattr(client, "_probe_gguf", lambda: True)
    monkeypatch.setattr(
        client, "_chat_gguf",
        lambda msgs, t, m, j: ChatResponse(text="from gguf", backend="gguf",
                                           model="local.gguf"))


# ── detection-time fallback (preferred backend unavailable) ──────────────────

def test_api_without_url_falls_back_to_gguf(monkeypatch):
    c = LocalLLMClient(prefer="api")
    c.api_url = ""
    c.fallback = "gguf"
    _gguf_ready(c, monkeypatch)
    assert c.detect_backend() == "gguf"
    assert c.chat([{"role": "user", "content": "hi"}]).text == "from gguf"


def test_bedrock_without_boto3_falls_back_to_gguf(monkeypatch):
    import src.knowledge.llm_client as m
    monkeypatch.setattr(m, "_boto3", None)
    c = LocalLLMClient(prefer="bedrock")
    c.fallback = "gguf"
    _gguf_ready(c, monkeypatch)
    assert c.detect_backend() == "gguf"


def test_default_fallback_is_still_stub(monkeypatch):
    c = LocalLLMClient(prefer="api")
    c.api_url = ""
    # fallback defaults to "stub"
    monkeypatch.setattr(c, "_probe_gguf", lambda: True)
    assert c.detect_backend() == "stub"


def test_fallback_gguf_but_weights_missing_is_stub(monkeypatch):
    c = LocalLLMClient(prefer="api")
    c.api_url = ""
    c.fallback = "gguf"
    monkeypatch.setattr(c, "_probe_gguf", lambda: False)   # no weights
    assert c.detect_backend() == "stub"


# ── runtime fallback (backend resolves, but the live call fails) ─────────────

def test_api_call_failure_retries_on_gguf(monkeypatch):
    c = LocalLLMClient(prefer="api")
    c.api_url = "https://gw.example.com/prod"
    c.fallback = "gguf"
    _gguf_ready(c, monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("gateway 503")

    monkeypatch.setattr(c, "_chat_api", _boom)
    resp = c.chat([{"role": "user", "content": "hi"}])
    assert resp.backend == "gguf" and resp.text == "from gguf"


def test_runtime_failure_without_gguf_fallback_raises(monkeypatch):
    c = LocalLLMClient(prefer="api")
    c.api_url = "https://gw.example.com/prod"
    # fallback stays "stub" → error surfaces, not silently swallowed
    monkeypatch.setattr(c, "_chat_api",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("503")))
    with pytest.raises(RuntimeError):
        c.chat([{"role": "user", "content": "hi"}])


def test_chat_json_inherits_the_fallback(monkeypatch):
    c = LocalLLMClient(prefer="api")
    c.api_url = "https://gw.example.com/prod"
    c.fallback = "gguf"
    monkeypatch.setattr(c, "_probe_gguf", lambda: True)
    monkeypatch.setattr(
        c, "_chat_gguf",
        lambda msgs, t, m, j: ChatResponse(text='{"ok": 1}', backend="gguf",
                                           model="local.gguf"))
    monkeypatch.setattr(c, "_chat_api",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    assert c.chat_json(system="s", user="u") == {"ok": 1}

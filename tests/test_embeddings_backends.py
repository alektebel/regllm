"""Tests for the multi-backend src.knowledge.embeddings.EmbeddingService."""

from __future__ import annotations

import pytest

from src.knowledge import embeddings as embeddings_module
from src.knowledge.embeddings import EmbeddingService


class _FakeLlama:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create_embedding(self, texts):
        return {"data": [{"embedding": [float(len(t)), 0.5]} for t in texts]}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Never let these tests hit a real network Ollama server."""
    monkeypatch.setattr(EmbeddingService, "_probe_ollama", lambda self: False)
    yield


def test_stub_backend_returns_zero_vectors():
    svc = EmbeddingService(backend="stub", dim=4)
    assert svc.available is False
    assert svc.embed(["hola", "adios"]) == [[0.0] * 4, [0.0] * 4]
    assert svc.embed_one("x") == [0.0] * 4


def test_auto_falls_back_to_stub_when_nothing_configured(monkeypatch):
    monkeypatch.setattr(EmbeddingService, "_probe_gguf", lambda self: False)
    monkeypatch.setattr(EmbeddingService, "_probe_bedrock", lambda self: False)
    svc = EmbeddingService(backend="auto", dim=3)
    assert svc.available is False
    assert svc.embed(["x"]) == [[0.0, 0.0, 0.0]]


def test_empty_input_returns_empty_list():
    svc = EmbeddingService(backend="stub")
    assert svc.embed([]) == []


# ── GGUF backend ─────────────────────────────────────────────────────────────

def test_gguf_backend_used_when_file_and_dep_present(monkeypatch, tmp_path):
    gguf_path = tmp_path / "embed.gguf"
    gguf_path.write_bytes(b"fake")
    monkeypatch.setattr(embeddings_module, "_Llama", _FakeLlama)
    svc = EmbeddingService(backend="gguf")
    svc.gguf_embed_model_path = str(gguf_path)
    assert svc.available is True
    vecs = svc.embed(["ab", "abcd"])
    assert vecs == [[2.0, 0.5], [4.0, 0.5]]


def test_gguf_backend_falls_back_to_stub_when_dep_missing(monkeypatch, tmp_path):
    gguf_path = tmp_path / "embed.gguf"
    gguf_path.write_bytes(b"fake")
    monkeypatch.setattr(embeddings_module, "_Llama", None)
    svc = EmbeddingService(backend="gguf", dim=2)
    svc.gguf_embed_model_path = str(gguf_path)
    assert svc.available is False
    assert svc.embed(["x"]) == [[0.0, 0.0]]


def test_gguf_embed_model_path_falls_back_to_chat_model(monkeypatch):
    monkeypatch.setenv("GGUF_MODEL_PATH", "/some/chat-model.gguf")
    monkeypatch.delenv("GGUF_EMBED_MODEL_PATH", raising=False)
    svc = EmbeddingService()
    assert svc.gguf_embed_model_path == "/some/chat-model.gguf"


# ── Bedrock backend ──────────────────────────────────────────────────────────

class _FakeBody:
    def __init__(self, payload):
        self._payload = payload
    def read(self):
        return self._payload


class _FakeBedrockClient:
    def invoke_model(self, modelId, body):
        import json
        _FakeBedrockClient.calls.append((modelId, json.loads(body)))
        return {"body": _FakeBody(json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode())}


def _fake_boto3(has_credentials: bool):
    _FakeBedrockClient.calls = []

    class _FakeSession:
        def get_credentials(self):
            return object() if has_credentials else None

    return type("FakeBoto3", (), {
        "client": staticmethod(lambda *a, **k: _FakeBedrockClient()),
        "Session": _FakeSession,
    })


def test_bedrock_backend_embeds_via_titan(monkeypatch):
    monkeypatch.setattr(embeddings_module, "_boto3", _fake_boto3(has_credentials=True))
    svc = EmbeddingService(backend="bedrock")
    assert svc.available is True
    vecs = svc.embed(["hola", "mundo"])
    assert vecs == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert len(_FakeBedrockClient.calls) == 2
    assert _FakeBedrockClient.calls[0][0] == "amazon.titan-embed-text-v2:0"
    assert _FakeBedrockClient.calls[0][1] == {"inputText": "hola"}


def test_bedrock_unavailable_when_boto3_missing(monkeypatch):
    monkeypatch.setattr(embeddings_module, "_boto3", None)
    svc = EmbeddingService(backend="bedrock", dim=2)
    assert svc.available is False
    assert svc.embed(["x"]) == [[0.0, 0.0]]


def test_bedrock_unavailable_when_credentials_missing(monkeypatch):
    monkeypatch.setattr(embeddings_module, "_boto3", _fake_boto3(has_credentials=False))
    svc = EmbeddingService(backend="bedrock", dim=2)
    assert svc.available is False
    assert svc.embed(["x"]) == [[0.0, 0.0]]


def test_auto_skips_bedrock_when_credentials_missing(monkeypatch):
    """The bug this guards: boto3 being merely *importable* (always true in
    the DQC image) must not make 'auto' prefer a misconfigured Bedrock over
    falling back to stub."""
    monkeypatch.setattr(embeddings_module, "_boto3", _fake_boto3(has_credentials=False))
    monkeypatch.setattr(EmbeddingService, "_probe_gguf", lambda self: False)
    svc = EmbeddingService(backend="auto", dim=2)
    assert svc.resolved_backend == "stub"


# ── explicit backend forcing / failure degrades to zero vectors ────────────

def test_backend_exception_degrades_to_zero_vectors(monkeypatch, tmp_path):
    class _BrokenLlama:
        def __init__(self, **kwargs):
            pass
        def create_embedding(self, texts):
            raise RuntimeError("boom")

    gguf_path = tmp_path / "embed.gguf"
    gguf_path.write_bytes(b"fake")
    monkeypatch.setattr(embeddings_module, "_Llama", _BrokenLlama)
    svc = EmbeddingService(backend="gguf", dim=5)
    svc.gguf_embed_model_path = str(gguf_path)
    assert svc.embed(["x"]) == [[0.0] * 5]

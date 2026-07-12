"""Tests for the standalone GGUF backend (`llama-cpp-python`) in
``src.knowledge.llm_client``.

No real ``.gguf`` weight file or ``llama-cpp-python`` install is needed —
``src.knowledge.llm_client._Llama`` (the lazily-imported class) is
monkeypatched with a fake that mimics the ``create_chat_completion`` surface
the client actually calls. This exercises the real dispatch/parsing code
paths (``detect_backend``, ``chat``, ``chat_json``, ``chat_json_stream``,
``chat_tools``) without needing GBs of model weights in CI.
"""

from __future__ import annotations

import json

import pytest

from src.knowledge import llm_client as llm_client_module
from src.knowledge.llm_client import ChatResponse, LocalLLMClient


class _FakeLlama:
    """Stands in for ``llama_cpp.Llama``."""

    instances: list["_FakeLlama"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeLlama.instances.append(self)

    def create_chat_completion(self, **kwargs):
        if kwargs.get("stream"):
            return self._stream()
        if kwargs.get("tools"):
            return {
                "choices": [{
                    "message": {
                        "content": "",
                        "tool_calls": [{
                            "id": "call_0",
                            "function": {
                                "name": "inspect_lineage",
                                "arguments": json.dumps({"target": "ECL"}),
                            },
                        }],
                    },
                }],
            }
        if kwargs.get("response_format"):
            content = json.dumps({"dqcs": [{"dqc_id": "DQC_TEST"}]})
        else:
            content = "hola desde gguf"
        return {"choices": [{"message": {"content": content}}]}

    def _stream(self):
        for chunk in ('{"d', 'qcs": ', "[]}"):
            yield {"choices": [{"delta": {"content": chunk}}]}


@pytest.fixture()
def gguf_file(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"not a real gguf, just needs to exist")
    return path


@pytest.fixture(autouse=True)
def _reset_fake_llama(monkeypatch):
    _FakeLlama.instances = []
    monkeypatch.setattr(llm_client_module, "_Llama", _FakeLlama)
    yield
    _FakeLlama.instances = []


def _make_client(gguf_file, **overrides) -> LocalLLMClient:
    c = LocalLLMClient(prefer="gguf")
    c.gguf_model_path = str(gguf_file)
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


# ── detection & lazy loading ────────────────────────────────────────────────

def test_detect_backend_gguf_when_file_and_dep_present(gguf_file):
    c = _make_client(gguf_file)
    assert c.detect_backend() == "gguf"


def test_detect_backend_falls_back_to_stub_when_file_missing(tmp_path):
    c = LocalLLMClient(prefer="gguf")
    c.gguf_model_path = str(tmp_path / "does_not_exist.gguf")
    assert c.detect_backend() == "stub"


def test_detect_backend_falls_back_to_stub_when_dep_missing(gguf_file, monkeypatch):
    monkeypatch.setattr(llm_client_module, "_Llama", None)
    c = _make_client(gguf_file)
    assert c.detect_backend() == "stub"


def test_probing_does_not_load_the_model(gguf_file):
    """The whole point of the cheap probe: detect_backend() must not
    instantiate Llama (which would load GBs of weights)."""
    c = _make_client(gguf_file)
    c.detect_backend()
    assert _FakeLlama.instances == []


def test_llama_instance_is_lazily_created_once(gguf_file):
    c = _make_client(gguf_file)
    c.chat([{"role": "user", "content": "hola"}])
    assert len(_FakeLlama.instances) == 1
    c.chat([{"role": "user", "content": "de nuevo"}])
    assert len(_FakeLlama.instances) == 1, "second call must reuse the cached instance"


def test_gguf_constructor_kwargs_from_config(gguf_file):
    c = _make_client(
        gguf_file, gguf_n_ctx=4096, gguf_n_gpu_layers=20,
        gguf_n_threads=8, gguf_chat_format="chatml",
    )
    c.chat([{"role": "user", "content": "hola"}])
    kwargs = _FakeLlama.instances[0].kwargs
    assert kwargs["n_ctx"] == 4096
    assert kwargs["n_gpu_layers"] == 20
    assert kwargs["n_threads"] == 8
    assert kwargs["chat_format"] == "chatml"
    assert kwargs["model_path"] == str(gguf_file)


# ── chat / chat_json / streaming / tools ────────────────────────────────────

def test_chat_returns_gguf_backend(gguf_file):
    c = _make_client(gguf_file)
    resp = c.chat([{"role": "user", "content": "hola"}])
    assert isinstance(resp, ChatResponse)
    assert resp.backend == "gguf"
    assert resp.model == "model.gguf"
    assert resp.text == "hola desde gguf"


def test_chat_json_parses_gguf_response(gguf_file):
    c = _make_client(gguf_file)
    result = c.chat_json(system="sys", user="user")
    assert result == {"dqcs": [{"dqc_id": "DQC_TEST"}]}


def test_chat_json_stream_yields_tokens_then_parsed_json(gguf_file):
    c = _make_client(gguf_file)
    events = list(c.chat_json_stream(system="sys", user="user"))
    *tokens, final = events
    assert tokens == ['{"d', 'qcs": ', "[]}"]
    assert final == {"dqcs": []}


def test_chat_tools_parses_tool_calls(gguf_file):
    c = _make_client(gguf_file)
    resp = c.chat_tools(
        messages=[{"role": "user", "content": "why?"}],
        tools=[{"type": "function", "function": {"name": "inspect_lineage"}}],
    )
    assert resp.backend == "gguf"
    assert resp.tool_calls == [{
        "id": "call_0", "name": "inspect_lineage", "arguments": {"target": "ECL"},
    }]

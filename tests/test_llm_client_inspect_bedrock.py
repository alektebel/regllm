"""get_inspect_client model selection on an all-Bedrock deployment.

Only exercises client construction / attribute wiring (no chat), so it runs
without boto3 installed — detect_backend is never called here.
"""

from __future__ import annotations

import pytest

from src.knowledge import llm_client as m


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # neutralise config.yaml so only env vars decide, and start fresh
    monkeypatch.setattr(m, "_LLM_CFG", {}, raising=False)
    m.reset_client()
    yield
    m.reset_client()


def test_inspect_uses_cheap_bedrock_model_while_generation_uses_strong(monkeypatch):
    monkeypatch.setenv("REGLLM_LLM", "bedrock")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "eu.anthropic.claude-3-5-sonnet-v1:0")
    monkeypatch.setenv("INSPECT_BEDROCK_MODEL_ID", "eu.amazon.nova-micro-v1:0")

    gen = m.get_client()
    insp = m.get_inspect_client()

    assert gen.prefer == "bedrock" and insp.prefer == "bedrock"
    assert gen.bedrock_model_id == "eu.anthropic.claude-3-5-sonnet-v1:0"
    assert insp.bedrock_model_id == "eu.amazon.nova-micro-v1:0"
    assert insp is not gen


def test_inspect_defaults_to_generation_model_when_override_unset(monkeypatch):
    monkeypatch.setenv("REGLLM_LLM", "bedrock")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "eu.amazon.nova-micro-v1:0")
    monkeypatch.delenv("INSPECT_BEDROCK_MODEL_ID", raising=False)
    monkeypatch.delenv("INSPECT_MODEL", raising=False)

    # no inspect-specific model → falls through to the main client
    assert m.get_inspect_client() is m.get_client()

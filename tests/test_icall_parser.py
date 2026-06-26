"""Tests for Qwen2.5 _icall(...) text tool parsing."""

from __future__ import annotations

from src.agent.agent import _parse_text_tool_calls, _text_is_unexecuted_icall

MULTI_ICALL = """
_icall(

{"name": "query_cycle_data", "arguments": {"ciclo_id": "CIC_00042", "fields": ["LGD_ESTIMADA","MoC","ECL","OR_EAD"]}},

{}

)

_icall(

{"name": "query_cycle_data", "arguments": {"ciclo_id": "CIC_00043", "fields": ["LGD_ESTIMADA","MoC","ECL","OR_EAD"]}},

{}

)

_icall(

{"name": "query_cycle_data", "arguments": {"ciclo_id": "CIC_00044", "fields": ["LGD_ESTIMADA","MoC","ECL","OR_EAD"]}},

{}

)
"""


def test_parse_multiline_icall_blocks() -> None:
    calls = _parse_text_tool_calls(MULTI_ICALL)
    assert len(calls) == 3
    ids = [c["arguments"]["ciclo_id"] for c in calls]
    assert ids == ["CIC_00042", "CIC_00043", "CIC_00044"]


def test_text_is_unexecuted_icall() -> None:
    assert _text_is_unexecuted_icall(MULTI_ICALL) is True
    assert _text_is_unexecuted_icall("MoC = 0.05 * LGD") is False

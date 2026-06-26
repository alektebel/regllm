"""Agentic SAS-diff Q&A — lazy imports so training scripts can load tools without kuzu."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import AgentEvent, SASDiffAgent, run_agent
    from .tools import TOOL_REGISTRY, ToolSpec, dispatch_tool, tool_schemas

__all__ = [
    "AgentEvent",
    "SASDiffAgent",
    "run_agent",
    "TOOL_REGISTRY",
    "ToolSpec",
    "dispatch_tool",
    "tool_schemas",
]

_LAZY = {
    "AgentEvent": ".agent",
    "SASDiffAgent": ".agent",
    "run_agent": ".agent",
    "TOOL_REGISTRY": ".tools",
    "ToolSpec": ".tools",
    "dispatch_tool": ".tools",
    "tool_schemas": ".tools",
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(_LAZY[name], __name__)
    return getattr(mod, name)

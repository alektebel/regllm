"""Agentic SAS-diff Q&A.

Public API:

- :class:`SASDiffAgent` — natural-language Q&A loop over the SAS pipeline
  versions, attribution methods, change-log GraphRAG and a free-form
  markdown corpus (``data/docs/**/*.md``).
- :func:`run_agent` — convenience coroutine that yields :class:`AgentEvent`
  records suitable for SSE streaming.
- :data:`TOOL_REGISTRY` — the tool inventory exposed to the LLM.
"""

from __future__ import annotations

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

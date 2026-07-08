"""Agent package: tools and the two engine builders (LangGraph / Deep Agents).

The tools are bound to a `LeadStore` at construction time via `build_tools`, so
the same agent core runs unchanged against any storage backend.
"""

from __future__ import annotations

from app.agents.deep_agent import build_deep_agent
from app.agents.langgraph_agent import build_langgraph_agent
from app.agents.runner import astream, run
from app.agents.tools import build_tools

__all__ = [
    "astream",
    "build_deep_agent",
    "build_langgraph_agent",
    "build_tools",
    "run",
]

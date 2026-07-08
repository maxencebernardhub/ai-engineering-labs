"""Agent package: tools and the two engine builders (LangGraph / Deep Agents).

The tools are bound to a `LeadStore` at construction time via `build_tools`, so
the same agent core runs unchanged against any storage backend.
"""

from __future__ import annotations

from app.agents.tools import build_tools

__all__ = ["build_tools"]

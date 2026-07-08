"""Deep Agents commercial assistant — stateless, autonomous.

Adapted from lab 06 (`deep_agents/agent/agent.py`) for a stateless HTTP service:

* No checkpointer — the service is stateless (client-carried history).
* `interrupt_on={}` — the lab-06 declarative HITL gates are removed; the agent
  runs autonomously and its actions are surfaced in the structured response.
* The model is injected: `create_deep_agent` accepts either a `BaseChatModel`
  instance (used with a BYOK model, and with a fake model in tests) or a
  `"provider:model"` string, so `build_deep_agent` takes `model_or_string`.

The same six store-bound tools and system prompt as the LangGraph agent are used,
preserving lab 06's side-by-side comparison of the two engines.
"""

from __future__ import annotations

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from app.agents.langgraph_agent import SYSTEM_PROMPT


def build_deep_agent(
    model_or_string: BaseChatModel | str, tools: list[BaseTool]
) -> CompiledStateGraph:
    """Build the stateless Deep Agents agent over `model_or_string` and `tools`.

    `interrupt_on={}` makes it fully autonomous; no checkpointer is attached.
    """
    return create_deep_agent(
        model=model_or_string,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        interrupt_on={},
    )

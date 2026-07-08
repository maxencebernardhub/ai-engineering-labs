"""LangGraph commercial assistant — stateless, autonomous StateGraph.

Adapted from lab 06 (`langgraph/agent/agent.py`) for a stateless HTTP service:

* No checkpointer — the service is stateless; the client carries the full
  conversation history and replays it on every request.
* No human-in-the-loop node — sensitive actions (email drafts, status changes)
  run autonomously and are surfaced in the structured response; the client
  re-invokes with new instructions if it wants changes.
* The LLM and tools are injected (`build_langgraph_agent(llm, tools)`) instead of
  being constructed from a provider name, so the API can inject a BYOK model and
  store-bound tools per request.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

SYSTEM_PROMPT = """You are a commercial assistant for a SME (small/medium enterprise).
Your role is to help manage the sales leads pipeline.

You can help with:
- Listing and searching leads
- Adding new leads or notes
- Updating lead statuses
- Generating email drafts for leads
- Providing pipeline statistics

You ONLY answer questions related to leads, sales, and the commercial pipeline.
For any other topic, politely decline and explain your specialization."""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_langgraph_agent(
    llm: BaseChatModel, tools: list[BaseTool]
) -> CompiledStateGraph:
    """Compile the stateless LangGraph agent over `llm` and `tools`.

    Two nodes — `agent` (calls the tool-bound LLM) and `tools` (executes any
    requested tool calls) — with a conditional edge that loops back through the
    agent until it stops requesting tools. No checkpointer, no HITL node.
    """
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def should_continue(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        return "tools" if getattr(last_msg, "tool_calls", None) else "end"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    builder.add_edge("tools", "agent")

    return builder.compile()

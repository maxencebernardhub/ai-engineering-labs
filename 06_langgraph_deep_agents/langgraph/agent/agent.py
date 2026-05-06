"""LangGraph commercial assistant agent — low-level StateGraph implementation."""

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from langgraph.agent.state import AgentState
from shared.config import get_llm
from shared.tools import (
    add_lead_tool,
    add_note_tool,
    generate_email_draft_tool,
    get_pipeline_stats_tool,
    list_leads_tool,
    update_lead_status_tool,
)

TOOLS = [
    list_leads_tool,
    add_lead_tool,
    add_note_tool,
    update_lead_status_tool,
    generate_email_draft_tool,
    get_pipeline_stats_tool,
]

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


def _find_hitl_call(tool_calls: list) -> dict | None:
    """Return the first tool call that requires human review, or None."""
    for tc in tool_calls:
        if tc["name"] == "generate_email_draft_tool":
            return tc
        if (
            tc["name"] == "update_lead_status_tool"
            and tc["args"].get("new_status") == "lost"
        ):
            return tc
    return None


def create_agent(provider: str = "anthropic", model: str | None = None):
    """Build and compile the LangGraph commercial assistant.

    Returns a compiled StateGraph with MemorySaver checkpointing.
    """
    llm = get_llm(provider, model)
    llm_with_tools = llm.bind_tools(TOOLS)

    # ── Nodes ────────────────────────────────────────────────────────────────

    def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def human_review_node(state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        hitl_call = _find_hitl_call(last_msg.tool_calls)
        if not hitl_call:
            return {}

        human_response = interrupt(
            {
                "tool_name": hitl_call["name"],
                "tool_args": hitl_call["args"],
                "message": (
                    f"Action requested: **{hitl_call['name']}**"
                    f" with args `{hitl_call['args']}`.\n"
                    "Reply with 'approve', 'cancel', or adjustment instructions."
                ),
            }
        )

        if isinstance(human_response, dict):
            decision = human_response.get("decision", "cancel").lower()
            feedback = human_response.get("feedback", "Action cancelled by user.")
        else:
            decision = str(human_response).strip().lower()
            feedback = str(human_response)

        if decision == "approve":
            # No new messages — ToolNode will execute the pending tool_calls.
            return {}

        # Cancelled or adjustment requested — surface feedback to the agent.
        return {
            "messages": [
                ToolMessage(
                    content=f"Action not executed. User response: {feedback}",
                    tool_call_id=hitl_call["id"],
                )
            ]
        }

    # ── Routing ──────────────────────────────────────────────────────────────

    def should_continue(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if not getattr(last_msg, "tool_calls", None):
            return "end"
        if _find_hitl_call(last_msg.tool_calls):
            return "hitl"
        return "tools"

    def after_hitl(state: AgentState) -> str:
        """Route to tools (approved) or back to agent (cancelled/adjusted)."""
        if isinstance(state["messages"][-1], ToolMessage):
            return "agent"
        return "tools"

    # ── Graph ────────────────────────────────────────────────────────────────

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("hitl", human_review_node)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"hitl": "hitl", "tools": "tools", "end": END},
    )
    builder.add_conditional_edges(
        "hitl",
        after_hitl,
        {"agent": "agent", "tools": "tools"},
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())

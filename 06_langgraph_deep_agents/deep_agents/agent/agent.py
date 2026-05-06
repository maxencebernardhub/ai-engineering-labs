"""Deep Agents commercial assistant — high-level declarative implementation.

Contrast with langgraph/agent/agent.py:
- No manual StateGraph, nodes, or edges
- HITL configured declaratively via interrupt_on (not via interrupt() calls)
- ~3x less boilerplate for equivalent functionality
"""

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

from shared.config import DEFAULT_MODELS
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

# Deep Agents uses "provider:model" strings; map our provider names accordingly.
_PROVIDER_MAP: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google_genai",
}

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

# Tools that require human review before execution.
# Note: unlike the LangGraph agent, Deep Agents interrupts on ALL calls to
# update_lead_status_tool (not only when new_status="lost"). This is a
# trade-off of the declarative approach: less granular, but zero boilerplate.
HITL_TOOLS: dict[str, bool] = {
    "generate_email_draft_tool": True,
    "update_lead_status_tool": True,
}


def create_agent(provider: str = "anthropic", model: str | None = None):
    """Build a Deep Agents commercial assistant.

    Returns a compiled agent with MemorySaver checkpointing and HITL
    configured declaratively via interrupt_on.

    Usage:
        agent = create_agent()
        config = {"configurable": {"thread_id": "my-thread"}}

        # Standard invoke (LangGraph-compatible)
        result = agent.invoke({"messages": [...]}, config)

        # Deep Agents v2 invoke (exposes .interrupts property)
        result = agent.invoke({"messages": [...]}, config, version="v2")
        if result.interrupts:
            decisions = [{"type": "approve"}]
            agent.invoke(Command(resume={"decisions": decisions}), config, version="v2")
    """
    model_name = model or DEFAULT_MODELS.get(provider, DEFAULT_MODELS["anthropic"])
    da_provider = _PROVIDER_MAP.get(provider, provider)
    model_string = f"{da_provider}:{model_name}"

    return create_deep_agent(
        model=model_string,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        interrupt_on=HITL_TOOLS,
        checkpointer=MemorySaver(),
    )

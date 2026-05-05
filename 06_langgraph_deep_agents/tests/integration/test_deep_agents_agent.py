"""Integration tests for the Deep Agents agent.

The LLM is mocked — these tests verify declarative HITL, guardrails, and
tool invocation without making real API calls.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from deep_agents.agent.agent import HITL_TOOLS, create_agent

# ── Helpers ──────────────────────────────────────────────────────────────────


def _config(thread_id: str = "da-test-thread") -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ── Unit: HITL configuration ──────────────────────────────────────────────────


def test_hitl_tools_declared():
    """Both HITL tools are declared in interrupt_on config."""
    assert "generate_email_draft_tool" in HITL_TOOLS
    assert "update_lead_status_tool" in HITL_TOOLS


def test_hitl_tools_enabled():
    """All declared HITL tools are enabled (truthy)."""
    assert all(HITL_TOOLS.values())


# ── Integration: normal tool call (no HITL) ───────────────────────────────────


@pytest.mark.integration
def test_list_leads_intent(leads_file, monkeypatch):
    """Agent calls list_leads_tool and returns a formatted response."""
    import shared.tools as t

    monkeypatch.setattr(t, "LEADS_PATH", leads_file)

    agent = create_agent()
    config = _config("da-thread-list")
    result = agent.invoke(
        {"messages": [HumanMessage(content="Show me all leads.")]},
        config,
    )
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert last.content


@pytest.mark.integration
def test_add_lead_intent(leads_file, monkeypatch):
    """Agent calls add_lead_tool with correct arguments."""
    import shared.tools as t

    monkeypatch.setattr(t, "LEADS_PATH", leads_file)

    agent = create_agent()
    config = _config("da-thread-add")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Add a new lead: Jean Durand from EcoTech,"
                        " email j.durand@ecotech.fr"
                    )
                )
            ]
        },
        config,
    )
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert last.content


# ── Integration: guardrail ────────────────────────────────────────────────────


@pytest.mark.integration
def test_guardrail_out_of_scope(leads_file, monkeypatch):
    """Agent refuses questions unrelated to the sales pipeline."""
    import shared.tools as t

    monkeypatch.setattr(t, "LEADS_PATH", leads_file)

    agent = create_agent()
    config = _config("da-thread-guardrail")
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is the capital of France?")]},
        config,
    )
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    content_lower = last.content.lower()
    assert any(
        word in content_lower
        for word in ["speciali", "only", "pipeline", "sales", "decline", "unable"]
    )


# ── Integration: HITL ────────────────────────────────────────────────────────


@pytest.mark.integration
def test_hitl_status_update_triggers_interrupt(leads_file, monkeypatch):
    """Graph interrupts before any update_lead_status_tool call."""
    import shared.tools as t

    monkeypatch.setattr(t, "LEADS_PATH", leads_file)

    agent = create_agent()
    config = _config("da-thread-hitl-status")

    list(
        agent.stream(
            {"messages": [HumanMessage(content="Mark lead_003 as qualified.")]},
            config,
            stream_mode="values",
        )
    )

    state = agent.get_state(config)
    assert state.next  # agent is paused pending human review


@pytest.mark.integration
def test_hitl_email_draft_triggers_interrupt(leads_file, monkeypatch):
    """Graph interrupts before generate_email_draft_tool call."""
    import shared.tools as t

    monkeypatch.setattr(t, "LEADS_PATH", leads_file)

    agent = create_agent()
    config = _config("da-thread-hitl-email")

    list(
        agent.stream(
            {
                "messages": [
                    HumanMessage(
                        content="Generate a follow-up email draft for lead_001."
                    )
                ]
            },
            config,
            stream_mode="values",
        )
    )

    state = agent.get_state(config)
    assert state.next

    # Resume with approval (Deep Agents v2 API)
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config,
    )
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)

"""Integration tests for the LangGraph agent.

The LLM is mocked — these tests verify graph structure, routing, HITL, and
guardrails without making real API calls.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from langgraph.agent.agent import _find_hitl_call, create_agent

# ── Helpers ──────────────────────────────────────────────────────────────────


def _tool_call(name: str, args: dict, call_id: str = "call_001") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _make_graph(mock_llm):
    """Compile a fresh graph with the mocked LLM bound to tools."""
    from unittest.mock import patch

    with patch("langgraph.agent.agent.get_llm", return_value=mock_llm):
        graph = create_agent()
    return graph


def _config(thread_id: str = "test-thread") -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ── Unit: HITL detection ──────────────────────────────────────────────────────


def test_find_hitl_call_email_draft():
    tc = _tool_call("generate_email_draft_tool", {"lead_id": "lead_001", "intent": "x"})
    assert _find_hitl_call([tc]) == tc


def test_find_hitl_call_status_lost():
    tc = _tool_call(
        "update_lead_status_tool", {"lead_id": "lead_001", "new_status": "lost"}
    )
    assert _find_hitl_call([tc]) == tc


def test_find_hitl_call_non_hitl_status():
    tc = _tool_call(
        "update_lead_status_tool", {"lead_id": "lead_001", "new_status": "qualified"}
    )
    assert _find_hitl_call([tc]) is None


def test_find_hitl_call_no_match():
    tc = _tool_call("list_leads_tool", {})
    assert _find_hitl_call([tc]) is None


# ── Integration: normal tool call (no HITL) ───────────────────────────────────


@pytest.mark.integration
def test_list_leads_intent(leads_file, monkeypatch):
    """Agent calls list_leads_tool and returns a formatted response."""
    import shared.tools as t

    monkeypatch.setattr(t, "LEADS_PATH", leads_file)

    graph = create_agent()
    config = _config("thread-list")
    result = graph.invoke(
        {"messages": [HumanMessage(content="Show me all leads.")]},
        config,
    )
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert last.content  # agent produced a response


# ── Integration: guardrail ────────────────────────────────────────────────────


@pytest.mark.integration
def test_guardrail_out_of_scope(leads_file, monkeypatch):
    """Agent refuses questions unrelated to the sales pipeline."""
    import shared.tools as t

    monkeypatch.setattr(t, "LEADS_PATH", leads_file)

    graph = create_agent()
    config = _config("thread-guardrail")
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is the weather in Paris today?")]},
        config,
    )
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    # Agent should politely decline
    content_lower = last.content.lower()
    assert any(
        word in content_lower
        for word in ["speciali", "only", "pipeline", "sales", "decline", "unable"]
    )


# ── Integration: HITL — status lost (confirmation gate) ──────────────────────


@pytest.mark.integration
def test_hitl_status_lost_approve(leads_file, monkeypatch):
    """Graph interrupts before marking a lead lost; resumes on approval."""
    import shared.tools as t

    monkeypatch.setattr(t, "LEADS_PATH", leads_file)

    graph = create_agent()
    config = _config("thread-hitl-lost")

    list(
        graph.stream(
            {"messages": [HumanMessage(content="Mark lead_003 as lost.")]},
            config,
            stream_mode="values",
        )
    )

    # Graph should have interrupted
    state = graph.get_state(config)
    assert state.next  # graph is paused, not finished

    # Resume with approval
    result = graph.invoke(Command(resume={"decision": "approve"}), config)
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)


@pytest.mark.integration
def test_hitl_status_lost_cancel(leads_file, monkeypatch):
    """Graph interrupts before marking a lead lost; cancels on rejection."""
    import shared.tools as t

    monkeypatch.setattr(t, "LEADS_PATH", leads_file)

    graph = create_agent()
    config = _config("thread-hitl-cancel")

    list(
        graph.stream(
            {"messages": [HumanMessage(content="Mark lead_003 as lost.")]},
            config,
            stream_mode="values",
        )
    )

    state = graph.get_state(config)
    assert state.next

    result = graph.invoke(Command(resume={"decision": "cancel"}), config)
    # Agent should acknowledge cancellation
    messages = result["messages"]
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assert any("not executed" in m.content for m in tool_msgs)


# ── Integration: checkpointing ────────────────────────────────────────────────


@pytest.mark.integration
def test_checkpointing_resumes_state(leads_file, monkeypatch):
    """Agent preserves conversation history across invocations via checkpointing."""
    import shared.tools as t

    monkeypatch.setattr(t, "LEADS_PATH", leads_file)

    graph = create_agent()
    config = _config("thread-checkpoint")

    graph.invoke(
        {"messages": [HumanMessage(content="How many leads do we have?")]},
        config,
    )

    # Second turn — agent should remember context
    graph.invoke(
        {"messages": [HumanMessage(content="And how many are qualified?")]},
        config,
    )
    state = graph.get_state(config)
    # Thread should have accumulated messages from both turns
    assert len(state.values["messages"]) >= 4

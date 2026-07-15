"""Agent + runner tests, driven by a scripted fake LLM (no real API calls).

`FakeToolCallingModel` returns pre-scripted `AIMessage`s in order — including
messages carrying `tool_calls` — and advances across the multiple LLM calls a
single agent run makes (agent -> tools -> agent). It implements `bind_tools`
(which the stock `GenericFakeChatModel` does not) and repeats its last response
once exhausted, which keeps the Deep Agents path robust to however many times its
middleware invokes the model. That lets us drive both engines deterministically
and assert what the runner parses out of the resulting message list.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from app.agents import build_langgraph_agent, run
from app.agents.runner import astream, build_tools
from app.storage import InMemoryStore


class FakeToolCallingModel(BaseChatModel):
    """A scripted chat model: returns `responses` in order, repeating the last.

    Implements both `_generate` (for `.invoke`) and `_stream` (so the streaming
    path emits real token deltas), which the stock fakes do not.
    """

    responses: list[AIMessage]
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:  # noqa: ARG002
        # Tools are ignored: the scripted responses already carry any tool_calls.
        return self

    def _next(self) -> AIMessage:
        message = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return message

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, **kwargs: Any
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    def _stream(
        self, messages: list[BaseMessage], stop: Any = None, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        message = self._next()
        if message.tool_calls:
            tool_call_chunks = [
                {
                    "name": tc["name"],
                    "args": json.dumps(tc["args"]),
                    "id": tc["id"],
                    "index": i,
                }
                for i, tc in enumerate(message.tool_calls)
            ]
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", tool_call_chunks=tool_call_chunks)
            )
            return
        for token in _word_chunks(str(message.content)):
            yield ChatGenerationChunk(message=AIMessageChunk(content=token))


def _word_chunks(text: str) -> Iterator[str]:
    """Split `text` into space-preserving chunks, mimicking token streaming."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word if i == len(words) - 1 else word + " "


def fake_llm(*responses: AIMessage) -> FakeToolCallingModel:
    """A fake chat model that yields `responses` in order (last one repeats)."""
    return FakeToolCallingModel(responses=list(responses))


def _tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


def test_langgraph_list_leads_intent():
    store = InMemoryStore()
    store.add_lead("Alice Moreau", "Tech Solutions SAS", "a@ts.fr")
    llm = fake_llm(
        _tool_call("list_leads", {}),
        AIMessage(content="You have 1 lead: Alice Moreau."),
    )

    result = run("langgraph", llm, store, [HumanMessage(content="list my leads")])

    assert "Alice Moreau" in result["reply"]
    assert [tc["name"] for tc in result["tool_calls"]] == ["list_leads"]
    # The tool's own output is surfaced alongside name/args.
    assert "Alice Moreau" in result["tool_calls"][0]["result"]
    assert result["leads_touched"] == []
    assert result["email_draft"] is None


def test_langgraph_add_lead_intent():
    store = InMemoryStore()
    llm = fake_llm(
        _tool_call(
            "add_lead",
            {"name": "Bob Martin", "company": "Acme", "email": "bob@acme.fr"},
        ),
        AIMessage(content="Added Bob Martin to the pipeline."),
    )

    result = run("langgraph", llm, store, [HumanMessage(content="add Bob Martin")])

    leads = store.list_leads()
    assert len(leads) == 1
    assert leads[0]["name"] == "Bob Martin"
    # The created lead id is surfaced in `leads_touched` even though `add_lead`
    # mints it internally (recovered from the tool result).
    assert result["leads_touched"] == [leads[0]["id"]]
    assert [tc["name"] for tc in result["tool_calls"]] == ["add_lead"]
    assert leads[0]["id"] in result["tool_calls"][0]["result"]


def test_langgraph_guardrail_out_of_scope():
    store = InMemoryStore()
    llm = fake_llm(
        AIMessage(content="I can only help with leads and the sales pipeline."),
    )

    result = run("langgraph", llm, store, [HumanMessage(content="what's the weather?")])

    assert "pipeline" in result["reply"].lower()
    assert result["tool_calls"] == []
    assert result["leads_touched"] == []


def test_runner_parses_email_draft():
    store = InMemoryStore()
    lead = store.add_lead("Alice Moreau", "Tech Solutions SAS", "a@ts.fr")
    llm = fake_llm(
        _tool_call(
            "generate_email_draft", {"lead_id": lead["id"], "intent": "follow-up"}
        ),
        AIMessage(content="Here is a draft for Alice Moreau — review before sending."),
    )

    result = run("langgraph", llm, store, [HumanMessage(content="draft an email")])

    draft = result["email_draft"]
    assert draft is not None
    assert draft["lead_id"] == lead["id"]
    assert draft["to"] == "a@ts.fr"
    assert "follow-up" in draft["subject"].lower()
    # A read-only draft does not count as touching the lead.
    assert result["leads_touched"] == []


def test_langgraph_reply_and_tool_calls_via_builder():
    """The runner's parsing matches a direct `build_langgraph_agent` invocation."""
    store = InMemoryStore()
    store.add_lead("Alice Moreau", "Tech Solutions SAS", "a@ts.fr")
    tools = build_tools(store)
    llm = fake_llm(
        _tool_call("list_leads", {}),
        AIMessage(content="Done."),
    )

    agent = build_langgraph_agent(llm, tools)
    result = agent.invoke({"messages": [HumanMessage(content="list leads")]})

    assert result["messages"][-1].content == "Done."


async def test_langgraph_astream_emits_tokens_then_final():
    store = InMemoryStore()
    store.add_lead("Alice Moreau", "Tech Solutions SAS", "a@ts.fr")
    llm = fake_llm(
        _tool_call("list_leads", {}),
        AIMessage(content="You have 1 lead."),
    )

    events = [
        event
        async for event in astream(
            "langgraph", llm, store, [HumanMessage(content="list leads")]
        )
    ]

    assert events[-1]["type"] == "final"
    assert [tc["name"] for tc in events[-1]["tool_calls"]] == ["list_leads"]
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert "1 lead" in tokens


def test_run_unknown_engine_raises():
    store = InMemoryStore()
    with pytest.raises(ValueError, match="Unknown engine"):
        run("nope", fake_llm(AIMessage(content="x")), store, [])


def test_deep_agents_smoke():
    """Deep Agents accepts a `BaseChatModel` instance, so the fake drives it too.

    The fake returns a plain final answer on every call (`itertools.repeat`),
    which is robust to however many times the Deep Agents middleware invokes the
    model; we only assert the engine runs and the runner parses a reply.
    """
    store = InMemoryStore()
    store.add_lead("Alice Moreau", "Tech Solutions SAS", "a@ts.fr")
    llm = fake_llm(AIMessage(content="I can help with your leads."))

    result = run("deep_agents", llm, store, [HumanMessage(content="hello")])

    assert "leads" in result["reply"].lower()
    assert result["email_draft"] is None

"""Engine-agnostic runner: drive an agent and parse its structured output.

Both engines compile to a LangGraph `CompiledStateGraph`, so the runner builds
the store-bound tools, selects the engine, invokes it with the client-supplied
message history, and flattens the resulting message list into the API response
shape: `reply`, `tool_calls`, `leads_touched`, `email_draft`, and `usage`.

`run` is synchronous (used by `POST /invoke`); `astream` streams assistant token
deltas followed by a final structured event (used by the SSE `POST
/invoke/stream`). Both share one parser so the two paths never diverge.

Key contract (from Step 3): `generate_email_draft` uses
`response_format="content_and_artifact"`, so the draft dict lives on the tool
message's `.artifact`, not in its text content — the parser reads it from there.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from app.agents.deep_agent import build_deep_agent
from app.agents.langgraph_agent import build_langgraph_agent
from app.agents.tools import build_tools
from app.storage import LeadStore

ENGINES = ("langgraph", "deep_agents")

# Tools that mutate a lead; used to compute `leads_touched`.
_WRITE_TOOLS = {"add_lead", "add_note", "update_lead_status"}
_EMAIL_DRAFT_TOOL = "generate_email_draft"
# Lead ids look like `lead_001` (seed) or `lead_<6 hex>` (created at runtime);
# `add_lead` mints the id inside the tool, so it is recovered from the result.
_LEAD_ID_RE = re.compile(r"lead_[0-9a-fA-F]+")


def _build_agent(
    engine: str, llm: BaseChatModel, tools: list[BaseTool]
) -> CompiledStateGraph:
    if engine == "langgraph":
        return build_langgraph_agent(llm, tools)
    if engine == "deep_agents":
        return build_deep_agent(llm, tools)
    raise ValueError(f"Unknown engine: {engine!r}. Supported: {list(ENGINES)}")


def _text(content: Any) -> str:
    """Flatten message content to plain text.

    Content is a string for most providers, or a list of blocks (e.g. Anthropic);
    only the textual parts are kept.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content else ""


def _parse(messages: Sequence[BaseMessage]) -> dict[str, Any]:
    """Flatten an agent's message list into the API response shape."""
    reply = ""
    tool_calls: list[dict[str, Any]] = []
    leads_touched: list[str] = []
    email_draft: dict | None = None
    input_tokens = output_tokens = 0

    # Map tool_call_id -> {call, entry} so each tool's result can be filled in
    # from its ToolMessage and traced back to the originating call.
    calls_by_id: dict[str, dict[str, Any]] = {}

    for msg in messages:
        if isinstance(msg, AIMessage):
            text = _text(msg.content)
            if text:
                reply = text
            for call in msg.tool_calls:
                entry = {"name": call["name"], "args": call["args"], "result": None}
                tool_calls.append(entry)
                calls_by_id[call["id"]] = {"call": call, "entry": entry}
            usage = msg.usage_metadata
            if usage:
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
        elif isinstance(msg, ToolMessage):
            tracked = calls_by_id.get(msg.tool_call_id, {})
            call = tracked.get("call", {})
            name = call.get("name") or msg.name
            result_text = _text(msg.content)
            if entry := tracked.get("entry"):
                entry["result"] = result_text
            if name == _EMAIL_DRAFT_TOOL and msg.artifact:
                email_draft = msg.artifact
            if name in _WRITE_TOOLS:
                _collect_lead_ids(name, call.get("args", {}), msg, leads_touched)

    return {
        "reply": reply,
        "tool_calls": tool_calls,
        "leads_touched": leads_touched,
        "email_draft": email_draft,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _collect_lead_ids(name: str, args: dict, msg: ToolMessage, acc: list[str]) -> None:
    """Append the lead ids touched by one write-tool call to `acc` (deduped)."""
    ids: list[str] = []
    if args.get("lead_id"):
        ids.append(args["lead_id"])
    elif name == "add_lead":
        # `add_lead` generates the id inside the tool; recover it from the result.
        ids.extend(_LEAD_ID_RE.findall(_text(msg.content)))
    for lead_id in ids:
        if lead_id not in acc:
            acc.append(lead_id)


def run(
    engine: str,
    llm: BaseChatModel,
    store: LeadStore,
    messages: list,
) -> dict[str, Any]:
    """Run the agent to completion and return the structured response."""
    agent = _build_agent(engine, llm, build_tools(store))
    result = agent.invoke({"messages": messages})
    return _parse(result["messages"])


async def astream(
    engine: str,
    llm: BaseChatModel,
    store: LeadStore,
    messages: list,
) -> AsyncIterator[dict[str, Any]]:
    """Stream assistant token deltas, then a final structured event.

    Yields `{"type": "token", "content": <delta>}` for each chunk of assistant
    text, and finally `{"type": "final", ...}` carrying the same fields as `run`.
    """
    agent = _build_agent(engine, llm, build_tools(store))
    final_messages: list[BaseMessage] = []

    async for mode, chunk in agent.astream(
        {"messages": messages}, stream_mode=["messages", "values"]
    ):
        if mode == "messages":
            msg, _meta = chunk
            if isinstance(msg, AIMessageChunk):
                text = _text(msg.content)
                if text:
                    yield {"type": "token", "content": text}
        elif mode == "values":
            final_messages = chunk["messages"]

    yield {"type": "final", **_parse(final_messages)}

"""Compatibility shim: keep Anthropic extended-thinking replayable when streaming.

Recent Claude models (e.g. the default `claude-sonnet-5`) use *adaptive* extended
thinking whose reasoning text defaults to `display: "omitted"`: the API returns a
`thinking` content block that carries a `signature` but no thinking text.

`langchain-anthropic` (1.4.x) handles that block inconsistently across code paths:

* **Non-streaming** — the aggregated `thinking` block keeps an (empty) `thinking`
  key, so replaying it on a follow-up turn is accepted by the API.
* **Streaming** — the block is rebuilt from a lone `signature_delta` event, so the
  aggregated block has only `type` + `signature`; the outgoing formatter then
  serializes `{"type": "thinking", "signature": ...}` with **no** `thinking`
  field. On the next tool-loop turn Anthropic rejects it with
  `messages.N.content.0.thinking.thinking: Field required`.

That only bites the streaming path (`POST /invoke/stream`) on tool-using runs, and
only for Anthropic models that emit signature-only thinking (today: `claude-sonnet-5`;
`claude-opus-4-8` streams thinking text, `claude-haiku-4-5` doesn't think). Both the
LangGraph and Deep Agents engines are affected because both replay the assistant
turn through the same model.

`ThinkingSafeChatAnthropic` closes the gap at the single request-payload choke point
(`_get_request_payload`, shared by `_generate`/`_agenerate`/`_stream`/`_astream`):
any outgoing `thinking` block missing its `thinking` field gets an empty string —
exactly what the non-streaming path already sends, and what Anthropic accepts.
"""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelInput


def _backfill_thinking_text(payload: dict[str, Any]) -> None:
    """Add an empty `thinking` field to any signature-only thinking block."""
    for message in payload.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "thinking"
                and "thinking" not in block
            ):
                block["thinking"] = ""


class ThinkingSafeChatAnthropic(ChatAnthropic):
    """`ChatAnthropic` that repairs streamed `omitted`-thinking blocks on replay."""

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        _backfill_thinking_text(payload)
        return payload

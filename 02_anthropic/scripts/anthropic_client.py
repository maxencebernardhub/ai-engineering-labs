"""Reusable helpers for the Anthropic labs in this directory.

This module centralizes a few common patterns used across the labs:
- client creation
- plain text generation
- structured output generation with Pydantic
- text streaming
- embedding generation (via VoyageAI)

The goal is not to hide the Anthropic SDK completely. The goal is to provide a
small, readable wrapper that reduces repeated boilerplate while staying close
to the official API.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TypeVar

import voyageai
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY is not set")

DEFAULT_TEXT_MODEL = "claude-haiku-4-5"
DEFAULT_EMBEDDING_MODEL = "voyage-3"

StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)


def require_api_key() -> None:
    """Raise a clear error if the Anthropic API key is missing."""

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")


def get_client() -> Anthropic:
    """Create and return an Anthropic client for the current environment."""

    require_api_key()
    return Anthropic()


def generate_text(
    prompt: str,
    *,
    model: str = DEFAULT_TEXT_MODEL,
    system: str | None = None,
    max_tokens: int = 1024,
    client: Anthropic | None = None,
) -> str:
    """Generate a plain text response from one text prompt.

    Args:
        prompt: Main user prompt sent to the model.
        model: Text model to use.
        system: Optional system prompt for the conversation.
        max_tokens: Maximum number of tokens to generate.
        client: Optional pre-existing Anthropic client. If omitted, one is created.

    Returns:
        The final text returned by the model.
    """

    anthropic_client = client or get_client()
    params: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        params["system"] = system

    response = anthropic_client.messages.create(**params)
    return response.content[0].text


def generate_structured[StructuredOutputT: BaseModel](
    prompt: str,
    *,
    output_model: type[StructuredOutputT],
    model: str = DEFAULT_TEXT_MODEL,
    system: str | None = None,
    max_tokens: int = 1024,
    client: Anthropic | None = None,
) -> StructuredOutputT:
    """Generate a structured response validated with a Pydantic model.

    Uses tool use to enforce the JSON schema derived from the Pydantic model,
    mirroring the behaviour of OpenAI's responses.parse().

    Args:
        prompt: Main user prompt sent to the model.
        output_model: Pydantic model used as the structured output schema.
        model: Text model to use.
        system: Optional system prompt for the conversation.
        max_tokens: Maximum number of tokens to generate.
        client: Optional pre-existing Anthropic client. If omitted, one is created.

    Returns:
        A validated Pydantic object parsed from the model output.
    """

    anthropic_client = client or get_client()

    tool = {
        "name": "structured_output",
        "description": "Return the result in the required structured format.",
        "input_schema": output_model.model_json_schema(),
    }

    params: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": "structured_output"},
    }
    if system:
        params["system"] = system

    response = anthropic_client.messages.create(**params)

    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    return output_model.model_validate(tool_use_block.input)


def stream_response(
    prompt: str,
    *,
    model: str = DEFAULT_TEXT_MODEL,
    system: str | None = None,
    max_tokens: int = 1024,
    client: Anthropic | None = None,
) -> Iterator[str]:
    """Yield text chunks as they arrive from a streaming response.

    Args:
        prompt: Main user prompt sent to the model.
        model: Text model to use.
        system: Optional system prompt for the conversation.
        max_tokens: Maximum number of tokens to generate.
        client: Optional pre-existing Anthropic client. If omitted, one is created.

    Yields:
        Incremental text deltas from the streaming response.
    """

    anthropic_client = client or get_client()
    params: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        params["system"] = system

    with anthropic_client.messages.stream(**params) as stream:
        yield from stream.text_stream


def generate_text_from_messages(
    messages: list[dict],
    *,
    model: str = DEFAULT_TEXT_MODEL,
    system: str | None = None,
    max_tokens: int = 1024,
    client: Anthropic | None = None,
) -> str:
    """Generate text from a richer multi-turn message list.

    This helper is useful when the input is not a simple string, for example a
    list of alternating user/assistant turns or multimodal content blocks.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        model: Text model to use.
        system: Optional system prompt for the conversation.
        max_tokens: Maximum number of tokens to generate.
        client: Optional pre-existing Anthropic client. If omitted, one is created.

    Returns:
        The final text returned by the model.
    """

    anthropic_client = client or get_client()
    params: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        params["system"] = system

    response = anthropic_client.messages.create(**params)
    return response.content[0].text


def generate_embedding(
    text: str,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    client: voyageai.Client | None = None,
) -> list[float]:
    """Generate one embedding vector for one text input via VoyageAI.

    Note: Anthropic does not expose a native embedding API. VoyageAI is the
    recommended partner provider for embeddings used alongside Claude.
    """

    voyage_client = client or voyageai.Client()
    result = voyage_client.embed([text], model=model, input_type="document")
    return result.embeddings[0]


def generate_embeddings(
    texts: Sequence[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    client: voyageai.Client | None = None,
) -> list[list[float]]:
    """Generate embedding vectors for multiple texts in one request via VoyageAI."""

    voyage_client = client or voyageai.Client()
    result = voyage_client.embed(list(texts), model=model, input_type="document")
    return result.embeddings

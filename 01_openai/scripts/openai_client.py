"""Reusable helpers for the OpenAI labs in this directory.

This module centralizes a few common patterns used across the labs:
- client creation
- plain text generation
- structured output generation with Pydantic
- text streaming
- embedding generation

The goal is not to hide the OpenAI SDK completely. The goal is to provide a
small, readable wrapper that reduces repeated boilerplate while staying close
to the official API.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses.response_input_param import ResponseInputParam
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

DEFAULT_TEXT_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)


def require_api_key() -> None:
    """Raise a clear error if the OpenAI API key is missing."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")


def get_client() -> OpenAI:
    """Create and return an OpenAI client for the current environment."""

    require_api_key()
    return OpenAI()


def generate_text(
    prompt: str,
    *,
    model: str = DEFAULT_TEXT_MODEL,
    instructions: str | None = None,
    client: OpenAI | None = None,
) -> str:
    """Generate a plain text response from one text prompt.

    Args:
        prompt: Main user prompt sent to the model.
        model: Text model to use.
        instructions: Optional system-like instructions for the response.
        client: Optional pre-existing OpenAI client. If omitted, one is created.

    Returns:
        The final text returned by the model.
    """

    openai_client = client or get_client()
    response = openai_client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
    )
    return response.output_text


def generate_structured(
    prompt: str,
    *,
    output_model: type[StructuredOutputT],
    model: str = DEFAULT_TEXT_MODEL,
    instructions: str | None = None,
    client: OpenAI | None = None,
) -> StructuredOutputT:
    """Generate a structured response validated with a Pydantic model.

    Args:
        prompt: Main user prompt sent to the model.
        output_model: Pydantic model used as the structured output schema.
        model: Text model to use.
        instructions: Optional system-like instructions for the response.
        client: Optional pre-existing OpenAI client. If omitted, one is created.

    Returns:
        A validated Pydantic object parsed from the model output.
    """

    openai_client = client or get_client()
    response = openai_client.responses.parse(
        model=model,
        instructions=instructions,
        input=prompt,
        text_format=output_model,
    )
    return response.output_parsed


def stream_response(
    prompt: str,
    *,
    model: str = DEFAULT_TEXT_MODEL,
    instructions: str | None = None,
    client: OpenAI | None = None,
) -> Iterator[str]:
    """Yield text chunks as they arrive from a streaming response.

    Args:
        prompt: Main user prompt sent to the model.
        model: Text model to use.
        instructions: Optional system-like instructions for the response.
        client: Optional pre-existing OpenAI client. If omitted, one is created.

    Yields:
        Incremental text deltas from the streaming response.
    """

    openai_client = client or get_client()
    stream = openai_client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
        stream=True,
    )

    for event in stream:
        if event.type == "response.output_text.delta":
            yield getattr(event, "delta", "")


def generate_text_from_input(
    input_items: ResponseInputParam,
    *,
    model: str = DEFAULT_TEXT_MODEL,
    instructions: str | None = None,
    client: OpenAI | None = None,
) -> str:
    """Generate text from a richer Responses API input payload.

    This helper is useful when the input is not a simple string, for example a
    list of messages or multimodal content blocks.
    """

    openai_client = client or get_client()
    response = openai_client.responses.create(
        model=model,
        instructions=instructions,
        input=input_items,
    )
    return response.output_text


def generate_embedding(
    text: str,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int | None = None,
    client: OpenAI | None = None,
) -> list[float]:
    """Generate one embedding vector for one text input."""

    openai_client = client or get_client()
    response = openai_client.embeddings.create(
        model=model,
        input=text,
        dimensions=dimensions,
    )
    return response.data[0].embedding


def generate_embeddings(
    texts: Sequence[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int | None = None,
    client: OpenAI | None = None,
) -> list[list[float]]:
    """Generate embedding vectors for multiple texts in one request."""

    openai_client = client or get_client()
    response = openai_client.embeddings.create(
        model=model,
        input=list(texts),
        dimensions=dimensions,
    )
    return [item.embedding for item in response.data]

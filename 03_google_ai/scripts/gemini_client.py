"""Reusable helpers for the Google AI (Gemini) labs in this directory.

This module centralizes a few common patterns used across the labs:
- client creation
- plain text generation
- structured output generation with Pydantic
- text streaming
- embedding generation

The goal is not to hide the Google GenAI SDK completely. The goal is to provide a
small, readable wrapper that reduces repeated boilerplate while staying close
to the official API.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

# Load environment variables from the project root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

DEFAULT_TEXT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)


def require_api_key() -> None:
    """Raise a clear error if the Gemini API key is missing."""
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set")


def get_client() -> genai.Client:
    """Create and return a Google GenAI client for the current environment."""
    require_api_key()
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_text(
    prompt: str,
    *,
    model: str = DEFAULT_TEXT_MODEL,
    system_instruction: str | None = None,
    client: genai.Client | None = None,
) -> str:
    """Generate a plain text response from one text prompt.

    Args:
        prompt: Main user prompt sent to the model.
        model: Text model to use.
        system_instruction: Optional system-like instructions for the response.
        client: Optional pre-existing GenAI client. If omitted, one is created.

    Returns:
        The final text returned by the model.
    """
    genai_client = client or get_client()

    config = None
    if system_instruction:
        config = types.GenerateContentConfig(system_instruction=system_instruction)

    response = genai_client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text


def generate_structured[StructuredOutputT: BaseModel](
    prompt: str,
    *,
    output_model: type[StructuredOutputT],
    model: str = DEFAULT_TEXT_MODEL,
    system_instruction: str | None = None,
    client: genai.Client | None = None,
) -> StructuredOutputT:
    """Generate a structured response validated with a Pydantic model.

    Args:
        prompt: Main user prompt sent to the model.
        output_model: Pydantic model used as the structured output schema.
        model: Text model to use.
        system_instruction: Optional system-like instructions for the response.
        client: Optional pre-existing GenAI client. If omitted, one is created.

    Returns:
        A validated Pydantic object parsed from the model output.
    """
    genai_client = client or get_client()

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=output_model,
    )

    response = genai_client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    return output_model.model_validate_json(response.text)


def stream_response(
    prompt: str,
    *,
    model: str = DEFAULT_TEXT_MODEL,
    system_instruction: str | None = None,
    client: genai.Client | None = None,
) -> Iterator[str]:
    """Yield text chunks as they arrive from a streaming response.

    Args:
        prompt: Main user prompt sent to the model.
        model: Text model to use.
        system_instruction: Optional system-like instructions for the response.
        client: Optional pre-existing GenAI client. If omitted, one is created.

    Yields:
        Incremental text deltas from the streaming response.
    """
    genai_client = client or get_client()

    config = None
    if system_instruction:
        config = types.GenerateContentConfig(system_instruction=system_instruction)

    stream = genai_client.models.generate_content_stream(
        model=model,
        contents=prompt,
        config=config,
    )

    for chunk in stream:
        if chunk.text:
            yield chunk.text


def generate_embedding(
    text: str,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    client: genai.Client | None = None,
) -> list[float]:
    """Generate one embedding vector for one text input."""
    genai_client = client or get_client()
    response = genai_client.models.embed_content(
        model=model,
        contents=text,
    )
    # The SDK returns a list of embeddings even for a single input if passed as a string
    return response.embeddings[0].values


def generate_embeddings(
    texts: Sequence[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    client: genai.Client | None = None,
) -> list[list[float]]:
    """Generate embedding vectors for multiple texts in one request."""
    genai_client = client or get_client()
    response = genai_client.models.embed_content(
        model=model,
        contents=list(texts),
    )
    return [e.values for e in response.embeddings]

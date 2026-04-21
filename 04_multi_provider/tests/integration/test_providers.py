"""Integration tests for all three LLM providers.

These tests make REAL API calls and require valid keys set in the repo-root
``.env`` file::

    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    GEMINI_API_KEY=AI...

Run with::

    uv run pytest tests/integration/ -v -m integration

DO NOT run these in CI without providing secrets via GitHub Actions Secrets.
"""

import pytest
from pydantic import BaseModel

from llm_client import (
    AnthropicProvider,
    GeminiProvider,
    LLMResponse,
    OpenAIProvider,
    StreamingResponse,
)

_PROMPT = "Explain what a neural network is in 2 sentences."


# ---------------------------------------------------------------------------
# Pydantic schema used for structured output tests
# ---------------------------------------------------------------------------


class Summary(BaseModel):
    """A short structured summary."""

    title: str
    body: str


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_openai_generate_returns_llm_response(openai_provider: OpenAIProvider):
    """OpenAI non-streaming call must return a valid LLMResponse."""
    result = await openai_provider.generate(prompt=_PROMPT, model="gpt-5.4")
    assert isinstance(result, LLMResponse)
    assert result.text
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.provider == "openai"
    assert result.cost_usd >= 0


@pytest.mark.integration
async def test_openai_structured_output_returns_parsed_model(
    openai_provider: OpenAIProvider,
):
    """OpenAI structured output must populate result.parsed with a Summary object."""
    result = await openai_provider.generate(
        prompt=_PROMPT, model="gpt-5.4", schema=Summary
    )
    assert isinstance(result, LLMResponse)
    assert isinstance(result.parsed, Summary)
    assert result.parsed.title
    assert result.parsed.body


@pytest.mark.integration
async def test_openai_stream_yields_chunks(openai_provider: OpenAIProvider):
    """OpenAI streaming must yield at least one text chunk."""
    result = await openai_provider.generate(
        prompt=_PROMPT, model="gpt-5.4", stream=True
    )
    assert isinstance(result, StreamingResponse)
    chunks = [chunk async for chunk in result]
    assert len(chunks) > 0
    assert any(chunk for chunk in chunks)


@pytest.mark.integration
async def test_openai_stream_with_schema_returns_parsed_final_response(
    openai_provider: OpenAIProvider,
):
    """OpenAI streaming with schema must expose a parsed object on final_response."""
    result = await openai_provider.generate(
        prompt=_PROMPT, model="gpt-5.4", stream=True, schema=Summary
    )
    assert isinstance(result, StreamingResponse)
    async for _ in result:
        pass  # exhaust the iterator
    final = result.final_response
    assert isinstance(final, LLMResponse)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_anthropic_generate_returns_llm_response(
    anthropic_provider: AnthropicProvider,
):
    """Anthropic non-streaming call must return a valid LLMResponse."""
    result = await anthropic_provider.generate(
        prompt=_PROMPT, model="claude-sonnet-4-6"
    )
    assert isinstance(result, LLMResponse)
    assert result.text
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.provider == "anthropic"
    assert result.cost_usd >= 0


@pytest.mark.integration
async def test_anthropic_structured_output_returns_parsed_model(
    anthropic_provider: AnthropicProvider,
):
    """Anthropic structured output must populate result.parsed with a Summary object."""
    result = await anthropic_provider.generate(
        prompt=_PROMPT, model="claude-sonnet-4-6", schema=Summary
    )
    assert isinstance(result, LLMResponse)
    assert isinstance(result.parsed, Summary)
    assert result.parsed.title
    assert result.parsed.body


@pytest.mark.integration
async def test_anthropic_stream_yields_chunks(anthropic_provider: AnthropicProvider):
    """Anthropic streaming must yield at least one text chunk."""
    result = await anthropic_provider.generate(
        prompt=_PROMPT, model="claude-sonnet-4-6", stream=True
    )
    assert isinstance(result, StreamingResponse)
    chunks = [chunk async for chunk in result]
    assert len(chunks) > 0


@pytest.mark.integration
async def test_anthropic_stream_with_schema_returns_parsed_final_response(
    anthropic_provider: AnthropicProvider,
):
    """Anthropic streaming with schema must expose a valid final_response."""
    result = await anthropic_provider.generate(
        prompt=_PROMPT, model="claude-sonnet-4-6", stream=True, schema=Summary
    )
    assert isinstance(result, StreamingResponse)
    async for _ in result:
        pass
    final = result.final_response
    assert isinstance(final, LLMResponse)


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_gemini_generate_returns_llm_response(gemini_provider: GeminiProvider):
    """Gemini non-streaming call must return a valid LLMResponse."""
    result = await gemini_provider.generate(prompt=_PROMPT, model="gemini-3.1-pro")
    assert isinstance(result, LLMResponse)
    assert result.text
    assert result.provider == "gemini"
    assert result.cost_usd >= 0


@pytest.mark.integration
async def test_gemini_structured_output_returns_parsed_model(
    gemini_provider: GeminiProvider,
):
    """Gemini structured output must populate result.parsed with a Summary object."""
    result = await gemini_provider.generate(
        prompt=_PROMPT, model="gemini-3.1-pro", schema=Summary
    )
    assert isinstance(result, LLMResponse)
    assert isinstance(result.parsed, Summary)
    assert result.parsed.title
    assert result.parsed.body


@pytest.mark.integration
async def test_gemini_stream_yields_chunks(gemini_provider: GeminiProvider):
    """Gemini streaming must yield at least one text chunk."""
    result = await gemini_provider.generate(
        prompt=_PROMPT, model="gemini-3.1-pro", stream=True
    )
    assert isinstance(result, StreamingResponse)
    chunks = [chunk async for chunk in result]
    assert len(chunks) > 0


@pytest.mark.integration
async def test_gemini_stream_with_schema_returns_parsed_final_response(
    gemini_provider: GeminiProvider,
):
    """Gemini streaming with schema must expose a valid final_response."""
    result = await gemini_provider.generate(
        prompt=_PROMPT, model="gemini-3.1-pro", stream=True, schema=Summary
    )
    assert isinstance(result, StreamingResponse)
    async for _ in result:
        pass
    final = result.final_response
    assert isinstance(final, LLMResponse)

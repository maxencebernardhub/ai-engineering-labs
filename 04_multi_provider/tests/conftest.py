"""Shared pytest configuration and fixtures.

The ``integration`` mark is registered here so that tests decorated with
``@pytest.mark.integration`` are properly recognised by pytest.

Run only unit tests (no API keys required)::

    uv run pytest tests/unit/ -v

Run integration tests (requires real API keys in .env)::

    uv run pytest tests/integration/ -v -m integration
"""

from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env from repo root so that API keys are available during integration tests.
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")


# ---------------------------------------------------------------------------
# Async fixtures for integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
async def openai_provider():
    """Return a live OpenAIProvider (requires OPENAI_API_KEY)."""
    from llm_client import OpenAIProvider

    return OpenAIProvider()


@pytest.fixture()
async def anthropic_provider():
    """Return a live AnthropicProvider (requires ANTHROPIC_API_KEY)."""
    from llm_client import AnthropicProvider

    return AnthropicProvider()


@pytest.fixture()
async def gemini_provider():
    """Return a live GeminiProvider (requires GEMINI_API_KEY)."""
    from llm_client import GeminiProvider

    return GeminiProvider()

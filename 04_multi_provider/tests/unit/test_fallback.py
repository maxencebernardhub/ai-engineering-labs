"""Unit tests for generate_with_fallback — written before implementation (TDD)."""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_client.base import (
    AllProvidersFailedError,
    LLMResponse,
    NoProviderAvailableError,
    ProviderTimeoutError,
    RateLimitError,
    StreamingResponse,
)
from llm_client.router import generate_with_fallback


def make_response(provider: str = "openai", model: str = "gpt-5.4") -> LLMResponse:
    """Return a minimal LLMResponse for testing."""
    return LLMResponse(
        text="Hello",
        parsed=None,
        input_tokens=10,
        output_tokens=5,
        model=model,
        provider=provider,
        cost_usd=0.0001,
        latency_ms=200,
    )


def make_provider(name: str, response: LLMResponse | Exception) -> MagicMock:
    """Return a mock BaseProvider whose generate() either returns or raises."""
    provider = MagicMock()
    provider.PROVIDER_NAME = name
    if isinstance(response, Exception):
        provider.generate = AsyncMock(side_effect=response)
    else:
        provider.generate = AsyncMock(return_value=response)
    return provider


# Models available for fallback tests — two providers so fallback is possible.
_AVAILABLE_MODELS = ["gpt-5.4", "claude-sonnet-4-6"]


class TestImmediateFallback:
    async def test_rate_limit_error_triggers_immediate_fallback(self):
        """A RateLimitError on the first provider must trigger fallback to the next."""
        openai_provider = make_provider("openai", RateLimitError("429"))
        anthropic_response = make_response("anthropic", "claude-sonnet-4-6")
        anthropic_provider = make_provider("anthropic", anthropic_response)

        providers = {"openai": openai_provider, "anthropic": anthropic_provider}

        result = await generate_with_fallback(
            prompt="hello",
            providers=providers,
            available_models=_AVAILABLE_MODELS,
            strategy="cheapest",
        )
        assert isinstance(result, LLMResponse)
        assert result.provider == "anthropic"

    async def test_generic_exception_triggers_immediate_fallback(self):
        """Any unexpected exception on the first provider must trigger fallback."""
        openai_provider = make_provider("openai", RuntimeError("boom"))
        anthropic_response = make_response("anthropic", "claude-sonnet-4-6")
        anthropic_provider = make_provider("anthropic", anthropic_response)

        providers = {"openai": openai_provider, "anthropic": anthropic_provider}

        result = await generate_with_fallback(
            prompt="hello",
            providers=providers,
            available_models=_AVAILABLE_MODELS,
            strategy="cheapest",
        )
        assert isinstance(result, LLMResponse)
        assert result.provider == "anthropic"


class TestTimeoutRetry:
    async def test_timeout_retries_once_before_fallback(self):
        """A ProviderTimeoutError must be retried once, then fall back."""
        # First two calls raise timeout (original + 1 retry), third succeeds.
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ProviderTimeoutError("timeout")
            return make_response("anthropic", "claude-sonnet-4-6")

        openai_provider = MagicMock()
        openai_provider.PROVIDER_NAME = "openai"
        openai_provider.generate = AsyncMock(side_effect=ProviderTimeoutError("to"))

        anthropic_response = make_response("anthropic", "claude-sonnet-4-6")
        anthropic_provider = make_provider("anthropic", anthropic_response)

        providers = {"openai": openai_provider, "anthropic": anthropic_provider}

        result = await generate_with_fallback(
            prompt="hello",
            providers=providers,
            available_models=_AVAILABLE_MODELS,
            strategy="cheapest",
        )
        # After 1 retry (2 attempts) on openai, falls back to anthropic
        assert openai_provider.generate.call_count == 2
        assert result.provider == "anthropic"


class TestAllProvidersFail:
    async def test_all_providers_fail_raises_all_providers_failed(self):
        """When every provider fails, AllProvidersFailedError must be raised."""
        openai_provider = make_provider("openai", RateLimitError("429"))
        anthropic_provider = make_provider("anthropic", RuntimeError("boom"))

        providers = {"openai": openai_provider, "anthropic": anthropic_provider}

        with pytest.raises(AllProvidersFailedError):
            await generate_with_fallback(
                prompt="hello",
                providers=providers,
                available_models=_AVAILABLE_MODELS,
                strategy="cheapest",
            )

    async def test_successful_fallback_returns_llm_response(self):
        """A successful fallback must return the response from the second provider."""
        openai_provider = make_provider("openai", RateLimitError("429"))
        anthropic_response = make_response("anthropic", "claude-sonnet-4-6")
        anthropic_provider = make_provider("anthropic", anthropic_response)

        providers = {"openai": openai_provider, "anthropic": anthropic_provider}

        result = await generate_with_fallback(
            prompt="hello",
            providers=providers,
            available_models=_AVAILABLE_MODELS,
            strategy="cheapest",
        )
        assert result.text == "Hello"
        assert result.provider == "anthropic"


class TestStreamingFallback:
    async def _make_streaming_response(
        self, provider: str, model: str
    ) -> StreamingResponse:
        """Build a StreamingResponse that immediately exhausts."""

        async def _gen() -> AsyncIterator[str]:
            yield "hello"

        return StreamingResponse(
            iterator=_gen(),
            model=model,
            provider=provider,
            schema=None,
            cost_tracker_callback=lambda r: None,
            start_time=0.0,
        )

    async def test_streaming_fallback_on_rate_limit(self):
        """A RateLimitError during streaming must fall back to the next provider."""
        openai_provider = make_provider("openai", RateLimitError("429"))

        anthropic_streaming = await self._make_streaming_response(
            "anthropic", "claude-sonnet-4-6"
        )
        anthropic_provider = make_provider("anthropic", anthropic_streaming)

        providers = {"openai": openai_provider, "anthropic": anthropic_provider}

        result = await generate_with_fallback(
            prompt="hello",
            providers=providers,
            available_models=_AVAILABLE_MODELS,
            strategy="cheapest",
            stream=True,
        )
        assert isinstance(result, StreamingResponse)

    async def test_streaming_fallback_on_timeout(self):
        """A ProviderTimeoutError during streaming must retry once then fall back."""
        openai_provider = make_provider("openai", ProviderTimeoutError("timeout"))

        anthropic_streaming = await self._make_streaming_response(
            "anthropic", "claude-sonnet-4-6"
        )
        anthropic_provider = make_provider("anthropic", anthropic_streaming)

        providers = {"openai": openai_provider, "anthropic": anthropic_provider}

        result = await generate_with_fallback(
            prompt="hello",
            providers=providers,
            available_models=_AVAILABLE_MODELS,
            strategy="cheapest",
            stream=True,
        )
        assert isinstance(result, StreamingResponse)
        # openai was tried twice (original + 1 retry)
        assert openai_provider.generate.call_count == 2


class TestNoProviderAvailable:
    async def test_empty_available_models_raises_no_provider_available(self):
        """generate_with_fallback with no models raises NoProviderAvailableError."""
        with pytest.raises(NoProviderAvailableError):
            await generate_with_fallback(
                prompt="hello",
                providers={"openai": MagicMock()},
                available_models=[],
                strategy="cheapest",
            )

    async def test_prompt_exceeds_all_context_windows_raises_no_provider_available(
        self,
    ):
        """A prompt longer than every model's context window raises NoProviderAvailableError."""
        # 4 chars × 4 = 1 estimated token — use a prompt that exceeds all windows.
        # gpt-5.4-nano has the smallest context among registry models (400k tokens),
        # so 400_001 * 4 chars guarantees the hard filter eliminates everything.
        huge_prompt = "x" * (400_001 * 4)
        with pytest.raises(NoProviderAvailableError):
            await generate_with_fallback(
                prompt=huge_prompt,
                providers={"openai": MagicMock()},
                available_models=["gpt-5.4-nano"],
                strategy="cheapest",
            )

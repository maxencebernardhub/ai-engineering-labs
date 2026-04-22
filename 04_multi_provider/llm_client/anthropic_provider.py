"""Anthropic provider using the Messages API (anthropic >= 0.95.0).

Pricing (as of 2026-04, source: platform.claude.com/docs/en/about-claude/models):
  - claude-opus-4-7:    $5.00  / 1M input,  $25.00 / 1M output
  - claude-opus-4-6:    $5.00  / 1M input,  $25.00 / 1M output
  - claude-sonnet-4-6:  $3.00  / 1M input,  $15.00 / 1M output
  - claude-sonnet-4-5:  $3.00  / 1M input,  $15.00 / 1M output
  - claude-haiku-4-5:   $1.00  / 1M input,   $5.00 / 1M output
"""

import datetime
import time
from pathlib import Path

from anthropic import APIStatusError, APITimeoutError, AsyncAnthropic
from anthropic import RateLimitError as AnthropicRateLimitError
from dotenv import load_dotenv
from pydantic import BaseModel

from llm_client.base import (
    BaseProvider,
    LLMResponse,
    ProviderTimeoutError,
    RateLimitError,
    StreamingResponse,
    StructuredOutputParseError,
)
from llm_client.cost_tracker import CostEntry, CostTracker

# Load .env from repo root (../../.env relative to this file's location)
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

# ---------------------------------------------------------------------------
# Pricing table (USD per 1M tokens)
# ---------------------------------------------------------------------------
_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7": {"input": 5.00, "output": 25.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}
_DEFAULT_PRICING = {"input": 3.00, "output": 15.00}


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute approximate cost in USD for a given model and token counts.

    Args:
        model: Anthropic model name.
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.

    Returns:
        Estimated cost in USD.
    """
    pricing = _PRICING.get(model, _DEFAULT_PRICING)
    return (
        input_tokens * pricing["input"] / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )


class AnthropicProvider(BaseProvider):
    """LLM provider backed by the Anthropic Messages API.

    Supports non-streaming, streaming, and structured output (via
    ``messages.parse`` with ``output_format=MyModel``).
    """

    PROVIDER_NAME = "anthropic"

    def __init__(
        self,
        client: AsyncAnthropic | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """Initialise the provider.

        Args:
            client: An existing :class:`AsyncAnthropic` instance.  When None,
                a new client is created (requires ``ANTHROPIC_API_KEY`` in the
                environment or ``.env``).
            cost_tracker: Optional :class:`CostTracker` to record call costs.
        """
        self._client = client or AsyncAnthropic()
        self._cost_tracker = cost_tracker

    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        schema: type[BaseModel] | None = None,
    ) -> LLMResponse | StreamingResponse:
        """Generate a response from an Anthropic model.

        Args:
            prompt: The user prompt.
            model: Anthropic model identifier (e.g. ``"claude-sonnet-4-6"``).
            temperature: Sampling temperature (0–1).
            max_tokens: Maximum output tokens.
            stream: When True, return a :class:`StreamingResponse`.
            schema: Optional Pydantic model enabling structured output.

        Returns:
            :class:`LLMResponse` or :class:`StreamingResponse`.

        Raises:
            RateLimitError: On HTTP 429 responses.
            ProviderTimeoutError: On request timeout.
        """
        if stream:
            return await self._generate_streaming(
                prompt, model, temperature, max_tokens, schema
            )
        return await self._generate_blocking(
            prompt, model, temperature, max_tokens, schema
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _generate_blocking(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        schema: type[BaseModel] | None,
    ) -> LLMResponse:
        """Non-streaming generation (with or without structured output)."""
        start = time.monotonic()
        try:
            if schema is not None:
                raw = await self._client.messages.parse(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                    output_format=schema,
                )
                # Extract text from the first content block
                text_blocks = [b.text for b in raw.content if b.type == "text"]
                text = "".join(text_blocks)
                parsed = raw.parsed_output
                if parsed is None:
                    raise StructuredOutputParseError(
                        f"Anthropic returned parsed=None for schema {schema}"
                    )
            else:
                raw = await self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                text_blocks = [b.text for b in raw.content if b.type == "text"]
                text = "".join(text_blocks)
                parsed = None

        except AnthropicRateLimitError as exc:
            # AnthropicRateLimitError is a subclass of APIStatusError, so this
            # clause is always matched first for 429 responses.
            raise RateLimitError(str(exc)) from exc
        except APIStatusError:
            raise
        except APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc)) from exc

        latency_ms = int((time.monotonic() - start) * 1000)
        input_tokens = raw.usage.input_tokens if raw.usage else 0
        output_tokens = raw.usage.output_tokens if raw.usage else 0
        cost = _cost(model, input_tokens, output_tokens)

        response = LLMResponse(
            text=text,
            parsed=parsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            provider=self.PROVIDER_NAME,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
        self._record(response)
        return response

    async def _generate_streaming(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        schema: type[BaseModel] | None,
    ) -> StreamingResponse:
        """Streaming generation — wraps Anthropic's stream context manager."""
        start = time.monotonic()
        meta: dict = {}

        async def _chunk_generator():
            """Async generator yielding text deltas from the Anthropic stream."""
            kwargs: dict = dict(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if schema is not None:
                # Pass output_format directly so messages.stream() populates
                # parsed_output on each ParsedTextBlock after the stream ends.
                kwargs["output_format"] = schema

            try:
                async with self._client.messages.stream(**kwargs) as stream:
                    async for text_chunk in stream.text_stream:
                        yield text_chunk
                    # Still inside the context manager — fetch final metadata.
                    final = await stream.get_final_message()
                    if final.usage:
                        inp = final.usage.input_tokens
                        out = final.usage.output_tokens
                        meta["input_tokens"] = inp
                        meta["output_tokens"] = out
                        meta["cost_usd"] = _cost(model, inp, out)
            except AnthropicRateLimitError as exc:
                # AnthropicRateLimitError is a subclass of APIStatusError, so this
                # clause is always matched first for 429 responses.
                raise RateLimitError(str(exc)) from exc
            except APIStatusError:
                raise
            except APITimeoutError as exc:
                raise ProviderTimeoutError(str(exc)) from exc

        return StreamingResponse(
            iterator=_chunk_generator(),
            model=model,
            provider=self.PROVIDER_NAME,
            schema=schema,
            cost_tracker_callback=self._record,
            start_time=start,
            meta=meta,
        )

    def _record(self, response: LLMResponse) -> None:
        """Log a completed response to the cost tracker if one is set."""
        if self._cost_tracker is None:
            return

        self._cost_tracker.log(
            CostEntry(
                ts=datetime.datetime.now(datetime.UTC).isoformat(),
                provider=response.provider,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
            )
        )

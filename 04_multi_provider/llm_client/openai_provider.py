"""OpenAI provider using the Responses API (openai >= 2.31.0).

Pricing (as of 2026-04, source: developers.openai.com/api/docs/models):
  - gpt-5.4:      $2.50 / 1M input,  $15.00 / 1M output
  - gpt-5.4-mini: $0.75 / 1M input,   $4.50 / 1M output
  - gpt-5.4-nano: $0.20 / 1M input,   $1.25 / 1M output
"""

import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APIStatusError, APITimeoutError, AsyncOpenAI
from openai.lib.streaming.responses import ResponseTextDeltaEvent
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
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
}
_DEFAULT_PRICING = {"input": 2.50, "output": 15.00}


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute approximate cost in USD for a given model and token counts.

    Args:
        model: OpenAI model name.
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


class OpenAIProvider(BaseProvider):
    """LLM provider backed by the OpenAI Responses API.

    Supports non-streaming, streaming, and structured output (via
    ``responses.parse`` with a Pydantic ``text_format``).
    """

    PROVIDER_NAME = "openai"

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """Initialise the provider.

        Args:
            client: An existing :class:`AsyncOpenAI` instance.  When None, a
                new client is created (requires ``OPENAI_API_KEY`` in the
                environment or ``.env``).
            cost_tracker: Optional :class:`CostTracker` to record call costs.
        """
        self._client = client or AsyncOpenAI()
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
        """Generate a response from an OpenAI model.

        Args:
            prompt: The user prompt.
            model: OpenAI model identifier (e.g. ``"gpt-5.4"``).
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
                raw = await self._client.responses.parse(
                    model=model,
                    input=prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    text_format=schema,
                )
                text = raw.output_text
                # .output_parsed holds the validated Pydantic object
                parsed = raw.output_parsed
                if parsed is None:
                    raise StructuredOutputParseError(
                        f"OpenAI returned output_parsed=None for schema {schema}"
                    )
            else:
                raw = await self._client.responses.create(
                    model=model,
                    input=prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                text = raw.output_text
                parsed = None

        except APIStatusError as exc:
            if exc.status_code == 429:
                raise RateLimitError(str(exc)) from exc
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
        """Streaming generation — wraps OpenAI's stream context manager."""
        start = time.monotonic()
        meta: dict = {}

        async def _chunk_generator():
            kwargs: dict = dict(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            if schema is not None:
                kwargs["text_format"] = schema

            try:
                async with self._client.responses.stream(**kwargs) as stream:
                    async for event in stream:
                        if isinstance(event, ResponseTextDeltaEvent):
                            yield event.delta
                    # Still inside the context manager — fetch final metadata.
                    final = await stream.get_final_response()
                    if final.usage:
                        inp = final.usage.input_tokens
                        out = final.usage.output_tokens
                        meta["input_tokens"] = inp
                        meta["output_tokens"] = out
                        meta["cost_usd"] = _cost(model, inp, out)
            except APIStatusError as exc:
                if exc.status_code == 429:
                    raise RateLimitError(str(exc)) from exc
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
        import datetime

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

"""Gemini provider using the google-genai SDK (google-genai >= 1.73.0).

Pricing (as of 2026-04, source: ai.google.dev/gemini-api/docs/pricing).
Two models have tiered pricing based on prompt length (threshold: 200k tokens):

  Model                       | Input ≤200k | Input >200k | Output ≤200k | Output >200k
  ----------------------------|-------------|-------------|--------------|-------------
  gemini-3.1-pro-preview      |       $2.00 |       $4.00 |       $12.00 |       $18.00
  gemini-2.5-pro              |       $1.25 |       $2.50 |       $10.00 |       $15.00
  gemini-3-flash-preview      |       $0.50 |       $0.50 |        $3.00 |        $3.00
  gemini-2.5-flash            |       $0.30 |       $0.30 |        $2.50 |        $2.50
  gemini-3.1-flash-lite-preview|      $0.25 |       $0.25 |        $1.50 |        $1.50

Structured output uses ``response_mime_type="application/json"`` combined
with ``response_json_schema=MyModel.model_json_schema()`` in the config.
"""

import time
from pathlib import Path

from dotenv import load_dotenv
from google.genai import Client
from google.genai.errors import APIError, ClientError
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel, ValidationError

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
# Pricing uses tiered keys: input_low/input_high/output_low/output_high
# where "low" = prompt ≤200k tokens, "high" = prompt >200k tokens.
_PRICING: dict[str, dict[str, float]] = {
    "gemini-3.1-pro-preview": {
        "input_low": 2.00,
        "input_high": 4.00,
        "output_low": 12.00,
        "output_high": 18.00,
    },
    "gemini-2.5-pro": {
        "input_low": 1.25,
        "input_high": 2.50,
        "output_low": 10.00,
        "output_high": 15.00,
    },
    "gemini-3-flash-preview": {
        "input_low": 0.50,
        "input_high": 0.50,
        "output_low": 3.00,
        "output_high": 3.00,
    },
    "gemini-2.5-flash": {
        "input_low": 0.30,
        "input_high": 0.30,
        "output_low": 2.50,
        "output_high": 2.50,
    },
    "gemini-3.1-flash-lite-preview": {
        "input_low": 0.25,
        "input_high": 0.25,
        "output_low": 1.50,
        "output_high": 1.50,
    },
}
_DEFAULT_PRICING = {
    "input_low": 1.25,
    "input_high": 2.50,
    "output_low": 10.00,
    "output_high": 15.00,
}

# Friendly aliases → actual Gemini API model identifiers.
_MODEL_ALIASES: dict[str, str] = {
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-preview",
}


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute cost in USD using tiered pricing (threshold: 200k input tokens).

    Args:
        model: Gemini API model name (after alias resolution).
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.

    Returns:
        Estimated cost in USD.
    """
    pricing = _PRICING.get(model, _DEFAULT_PRICING)
    tier = "high" if input_tokens > 200_000 else "low"
    return (
        input_tokens * pricing[f"input_{tier}"] / 1_000_000
        + output_tokens * pricing[f"output_{tier}"] / 1_000_000
    )


class GeminiProvider(BaseProvider):
    """LLM provider backed by the Google Gemini API.

    Supports non-streaming, streaming, and structured output (via
    ``response_mime_type`` + ``response_json_schema`` in the config).
    """

    PROVIDER_NAME = "gemini"

    def __init__(
        self,
        client: Client | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """Initialise the provider.

        Args:
            client: An existing :class:`google.genai.Client` instance.  When
                None, a new client is created (requires ``GEMINI_API_KEY`` in
                the environment or ``.env``).
            cost_tracker: Optional :class:`CostTracker` to record call costs.
        """
        self._client = client or Client()
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
        """Generate a response from a Gemini model.

        Args:
            prompt: The user prompt.
            model: Gemini model identifier (e.g. ``"gemini-3.1-pro"``).
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
        # Resolve any friendly alias to the actual API model identifier.
        resolved_model = _MODEL_ALIASES.get(model, model)
        if stream:
            return await self._generate_streaming(
                prompt, resolved_model, temperature, max_tokens, schema
            )
        return await self._generate_blocking(
            prompt, resolved_model, temperature, max_tokens, schema
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_config(
        self,
        temperature: float,
        max_tokens: int,
        schema: type[BaseModel] | None,
    ) -> GenerateContentConfig:
        """Build a :class:`GenerateContentConfig` with optional structured output.

        Args:
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            schema: Optional Pydantic model.  When set, adds
                ``response_mime_type`` and ``response_json_schema`` to the
                config to enable structured JSON output.

        Returns:
            Configured :class:`GenerateContentConfig`.
        """
        kwargs: dict = dict(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = schema.model_json_schema()
        return GenerateContentConfig(**kwargs)

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
        config = self._build_config(temperature, max_tokens, schema)
        try:
            raw = await self._client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except ClientError as exc:
            if exc.code == 429:
                raise RateLimitError(str(exc)) from exc
            raise
        except APIError as exc:
            # Treat connection/timeout-related errors as ProviderTimeoutError
            if exc.code in (408, 504):
                raise ProviderTimeoutError(str(exc)) from exc
            raise

        text = raw.text or ""
        parsed: BaseModel | None = None
        if schema is not None:
            try:
                parsed = schema.model_validate_json(text)
            except (ValidationError, ValueError) as exc:
                raise StructuredOutputParseError(
                    f"Gemini returned invalid JSON for schema {schema}: {exc}"
                ) from exc

        latency_ms = int((time.monotonic() - start) * 1000)
        usage = raw.usage_metadata
        input_tokens = (usage.prompt_token_count or 0) if usage else 0
        output_tokens = (usage.candidates_token_count or 0) if usage else 0
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
        """Streaming generation — wraps ``generate_content_stream``."""
        start = time.monotonic()
        meta: dict = {}
        config = self._build_config(temperature, max_tokens, schema)

        async def _chunk_generator():
            """Yield text chunks from the Gemini streaming response."""
            last_chunk = None
            try:
                async for (
                    chunk
                ) in await self._client.aio.models.generate_content_stream(
                    model=model,
                    contents=prompt,
                    config=config,
                ):
                    if chunk.text:
                        yield chunk.text
                    last_chunk = chunk
            except ClientError as exc:
                if exc.code == 429:
                    raise RateLimitError(str(exc)) from exc
                raise
            except APIError as exc:
                if exc.code in (408, 504):
                    raise ProviderTimeoutError(str(exc)) from exc
                raise

            # After all chunks — populate metadata from the last chunk.
            if last_chunk and last_chunk.usage_metadata:
                usage = last_chunk.usage_metadata
                inp = usage.prompt_token_count or 0
                out = usage.candidates_token_count or 0
                meta["input_tokens"] = inp
                meta["output_tokens"] = out
                meta["cost_usd"] = _cost(model, inp, out)

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

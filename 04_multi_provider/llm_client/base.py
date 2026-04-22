"""Base types, abstract provider, and custom exceptions for llm_client."""

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError


@dataclass
class LLMResponse:
    """Unified response object returned by all providers."""

    text: str
    parsed: BaseModel | None
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    cost_usd: float
    latency_ms: int


class StreamingResponse:
    """Custom AsyncIterator[str] that accumulates chunks.

    After exhausting the iterator, .final_response returns the full
    LLMResponse (including .parsed if a schema was provided).

    Usage::

        stream = await provider.generate(prompt=..., model=..., stream=True)
        async for chunk in stream:
            print(chunk, end="", flush=True)
        response = stream.final_response   # LLMResponse with .parsed populated
    """

    def __init__(
        self,
        iterator: AsyncIterator[str],
        model: str,
        provider: str,
        schema: type[BaseModel] | None,
        cost_tracker_callback: Callable[["LLMResponse"], None] | None,
        start_time: float,
        meta: dict | None = None,
    ) -> None:
        """Initialise the streaming response wrapper.

        Args:
            iterator: Async generator yielding text chunks.
            model: Model name used for the request.
            provider: Provider name (e.g. "openai").
            schema: Optional Pydantic model for structured output parsing.
                When set, the accumulated JSON is validated against this
                schema when ``final_response`` is accessed.
            cost_tracker_callback: Callable(LLMResponse) invoked once the
                stream is fully consumed, allowing cost to be recorded.
            start_time: ``time.monotonic()`` timestamp captured by the
                provider *before* the API call, used to compute
                ``latency_ms`` in ``final_response``.  Must be provided
                explicitly — omitting it would silently underestimate
                latency by excluding stream-establishment time.
            meta: Mutable dict populated by the chunk generator after the
                stream ends.  Expected keys: ``input_tokens``, ``output_tokens``,
                ``cost_usd``.  Defaults to an empty dict (all zeros).
        """
        self._iterator = iterator
        self._model = model
        self._provider = provider
        self._schema = schema
        self._cost_tracker_callback = cost_tracker_callback
        self._start_time = start_time
        self._meta: dict = meta if meta is not None else {}
        self._chunks: list[str] = []
        self._exhausted = False
        self._final: LLMResponse | None = None

    def __aiter__(self) -> "StreamingResponse":
        """Return self as the async iterator."""
        return self

    async def __anext__(self) -> str:
        """Fetch the next chunk from the underlying iterator."""
        try:
            chunk = await self._iterator.__anext__()
            self._chunks.append(chunk)
            return chunk
        except StopAsyncIteration:
            self._exhausted = True
            raise

    @property
    def final_response(self) -> LLMResponse:
        """Return the assembled LLMResponse after the stream is exhausted.

        Validates the accumulated text against ``schema`` (if provided) and
        invokes the cost-tracker callback.

        Raises:
            RuntimeError: If the iterator has not been fully consumed yet.
            StructuredOutputParseError: If ``schema`` is set and the
                accumulated JSON is invalid.
        """
        if not self._exhausted:
            msg = (
                "Stream not yet exhausted. "
                "Consume all chunks before accessing final_response."
            )
            raise RuntimeError(msg)
        if self._final is not None:
            return self._final

        full_text = "".join(self._chunks)
        input_tokens: int = self._meta.get("input_tokens", 0)
        output_tokens: int = self._meta.get("output_tokens", 0)
        cost_usd: float = self._meta.get("cost_usd", 0.0)
        latency_ms = int((time.monotonic() - self._start_time) * 1000)

        parsed: Any | None = None
        if self._schema is not None:
            try:
                parsed = self._schema.model_validate_json(full_text)
            except (ValidationError, ValueError) as exc:
                raise StructuredOutputParseError(
                    f"Failed to parse streaming response as "
                    f"{self._schema.__name__}: {exc}"
                ) from exc

        self._final = LLMResponse(
            text=full_text,
            parsed=parsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._model,
            provider=self._provider,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        if self._cost_tracker_callback is not None:
            self._cost_tracker_callback(self._final)
        return self._final


class BaseProvider(ABC):
    """Abstract base class that all LLM providers must implement."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        schema: type[BaseModel] | None = None,
    ) -> "LLMResponse | StreamingResponse":
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt to send.
            model: Model identifier (provider-specific).
            temperature: Sampling temperature (0–1).
            max_tokens: Maximum tokens to generate.
            stream: If True, return a StreamingResponse.
            schema: Optional Pydantic model for structured output.

        Returns:
            LLMResponse for non-streaming, StreamingResponse for streaming.
        """


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class NoProviderAvailableError(Exception):
    """Raised when the router cannot find any eligible provider/model."""


class AllProvidersFailedError(Exception):
    """Raised when every provider has been tried and all failed."""


class StructuredOutputParseError(Exception):
    """Raised when structured output JSON cannot be validated by the schema."""


class RateLimitError(Exception):
    """Wraps provider-specific 429 / rate-limit errors."""


class ProviderTimeoutError(Exception):
    """Wraps provider-specific timeout errors."""

"""Router and fallback logic for multi-provider LLM calls.

The router applies:
1. Hard filters: context window check (len(prompt)//4), exclude set.
2. Soft sort by strategy: cheapest | fastest | most_capable.

generate_with_fallback drives the retry/fallback loop:
- RateLimitError  → immediate fallback
- ProviderTimeoutError → 1 retry on the same provider, then fallback
- Any other exception → immediate fallback
- All providers exhausted → AllProvidersFailedError
"""

from __future__ import annotations

import datetime

from llm_client.base import (
    AllProvidersFailedError,
    BaseProvider,
    LLMResponse,
    NoProviderAvailableError,
    ProviderTimeoutError,
    RateLimitError,
    StreamingResponse,
)
from llm_client.cost_tracker import CostEntry, CostTracker

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    # ------------------------------------------------------------------
    # OpenAI  (source: developers.openai.com/api/docs/models, 2026-04)
    # ------------------------------------------------------------------
    "gpt-5.4": {
        "provider": "openai",
        "context_window": 1_050_000,
        "tier": 3,
        "cost_input_per_1m": 2.50,
        "cost_output_per_1m": 15.00,
        "latency_ms_estimate": 800,
    },
    "gpt-5.4-mini": {
        "provider": "openai",
        "context_window": 400_000,
        "tier": 2,
        "cost_input_per_1m": 0.75,
        "cost_output_per_1m": 4.50,
        "latency_ms_estimate": 400,
    },
    "gpt-5.4-nano": {
        "provider": "openai",
        "context_window": 400_000,
        "tier": 1,
        "cost_input_per_1m": 0.20,
        "cost_output_per_1m": 1.25,
        "latency_ms_estimate": 200,
    },
    # ------------------------------------------------------------------
    # Anthropic  (source: platform.claude.com/docs/en/about-claude/models,
    #             2026-04)
    # ------------------------------------------------------------------
    "claude-opus-4-7": {
        "provider": "anthropic",
        "context_window": 1_000_000,
        "tier": 3,
        "cost_input_per_1m": 5.00,
        "cost_output_per_1m": 25.00,
        "latency_ms_estimate": 1200,
    },
    "claude-opus-4-6": {
        "provider": "anthropic",
        "context_window": 1_000_000,
        "tier": 3,
        "cost_input_per_1m": 5.00,
        "cost_output_per_1m": 25.00,
        "latency_ms_estimate": 1200,
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "context_window": 1_000_000,
        "tier": 2,
        "cost_input_per_1m": 3.00,
        "cost_output_per_1m": 15.00,
        "latency_ms_estimate": 600,
    },
    "claude-sonnet-4-5": {
        "provider": "anthropic",
        "context_window": 200_000,
        "tier": 2,
        "cost_input_per_1m": 3.00,
        "cost_output_per_1m": 15.00,
        "latency_ms_estimate": 600,
    },
    "claude-haiku-4-5": {
        "provider": "anthropic",
        "context_window": 200_000,
        "tier": 1,
        "cost_input_per_1m": 1.00,
        "cost_output_per_1m": 5.00,
        "latency_ms_estimate": 300,
    },
    # ------------------------------------------------------------------
    # Gemini  (source: ai.google.dev/gemini-api/docs/pricing, 2026-04)
    # cost_input_per_1m uses the ≤200k tier for routing estimates.
    # ------------------------------------------------------------------
    "gemini-3.1-pro": {
        "provider": "gemini",
        "context_window": 1_000_000,
        "tier": 3,
        "cost_input_per_1m": 2.00,
        "cost_output_per_1m": 12.00,
        "latency_ms_estimate": 900,
    },
    "gemini-2.5-pro": {
        "provider": "gemini",
        "context_window": 1_000_000,
        "tier": 3,
        "cost_input_per_1m": 1.25,
        "cost_output_per_1m": 10.00,
        "latency_ms_estimate": 900,
    },
    "gemini-3-flash": {
        "provider": "gemini",
        "context_window": 1_000_000,
        "tier": 2,
        "cost_input_per_1m": 0.50,
        "cost_output_per_1m": 3.00,
        "latency_ms_estimate": 400,
    },
    "gemini-2.5-flash": {
        "provider": "gemini",
        "context_window": 1_000_000,
        "tier": 2,
        "cost_input_per_1m": 0.30,
        "cost_output_per_1m": 2.50,
        "latency_ms_estimate": 400,
    },
    "gemini-3.1-flash-lite": {
        "provider": "gemini",
        "context_window": 1_000_000,
        "tier": 1,
        "cost_input_per_1m": 0.25,
        "cost_output_per_1m": 1.50,
        "latency_ms_estimate": 200,
    },
}

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_SORT_KEYS: dict[str, tuple[str, bool]] = {
    # strategy → (registry_field, reverse)
    "cheapest": ("cost_input_per_1m", False),
    "fastest": ("latency_ms_estimate", False),
    "most_capable": ("tier", True),
}


class Router:
    """Select the best available model given a prompt and routing strategy."""

    def select(
        self,
        prompt: str,
        available_models: list[str],
        strategy: str = "cheapest",
        exclude: set[str] | None = None,
        has_image: bool = False,
    ) -> str:
        """Return the name of the best model after filtering and sorting.

        Hard filters (applied first):
        - Models not present in ``available_models`` are skipped.
        - Models in ``exclude`` are skipped.
        - Models whose context_window < estimated token count are skipped.
          Token estimation: ``len(prompt) // 4`` (documented limitation).

        Soft sort (applied to the remaining candidates):
        - ``"cheapest"``     → ascending ``cost_input_per_1m``
        - ``"fastest"``      → ascending ``latency_ms_estimate``
        - ``"most_capable"`` → descending ``tier``

        Args:
            prompt: The user prompt (used for token estimation).
            available_models: List of model names to consider.
            strategy: Routing strategy name.
            exclude: Model names to unconditionally exclude.
            has_image: Reserved for future multimodal filtering (unused).

        Returns:
            The selected model name.

        Raises:
            NoProviderAvailableError: When no model survives the hard filters.
        """
        estimated_tokens = len(prompt) // 4
        excluded = exclude or set()

        candidates = [
            m
            for m in available_models
            if m in MODEL_REGISTRY
            and m not in excluded
            and MODEL_REGISTRY[m]["context_window"] >= estimated_tokens
        ]

        if not candidates:
            raise NoProviderAvailableError(
                f"No model available after hard filters "
                f"(estimated_tokens={estimated_tokens}, excluded={excluded})"
            )

        field, reverse = _SORT_KEYS.get(strategy, ("cost_input_per_1m", False))
        candidates.sort(key=lambda m: MODEL_REGISTRY[m][field], reverse=reverse)
        return candidates[0]


# ---------------------------------------------------------------------------
# Fallback logic
# ---------------------------------------------------------------------------

# Module-level singleton — Router is stateless, no need to reinstantiate per call.
_router = Router()


async def generate_with_fallback(
    prompt: str,
    providers: dict[str, BaseProvider],
    available_models: list[str],
    strategy: str = "cheapest",
    cost_tracker: CostTracker | None = None,
    excluded: frozenset[str] | None = None,
    **generate_kwargs,
) -> LLMResponse | StreamingResponse:
    """Try providers in order, falling back on errors.

    Fallback policy:
    - :class:`RateLimitError`      → immediate fallback (no retry)
    - :class:`ProviderTimeoutError` → 1 retry on same model, then fallback
    - Any other exception          → immediate fallback (no retry)
    - All providers exhausted      → :class:`AllProvidersFailedError`

    Args:
        prompt: The user prompt.
        providers: Mapping of provider name → :class:`BaseProvider` instance.
        available_models: Candidate model names for the router.
        strategy: Routing strategy (``"cheapest"`` | ``"fastest"`` |
            ``"most_capable"``).
        cost_tracker: Optional tracker to record costs after each call.
        excluded: Model names already excluded from this invocation.
        **generate_kwargs: Extra keyword arguments forwarded to
            ``provider.generate()`` (e.g. ``temperature``, ``max_tokens``,
            ``stream``, ``schema``).

    Returns:
        :class:`LLMResponse` or :class:`StreamingResponse`.

    Raises:
        NoProviderAvailableError: When the router finds no eligible model.
        AllProvidersFailedError: When every eligible provider has been tried
            and all failed.
    """
    current_excluded: set[str] = set(excluded or set())
    failures: list[tuple[str, Exception]] = []

    while True:
        # Let NoProviderAvailableError propagate when nothing is left.
        model = _router.select(
            prompt,
            available_models,
            strategy=strategy,
            exclude=current_excluded,
        )
        meta = MODEL_REGISTRY[model]
        provider_name: str = meta["provider"]
        provider = providers.get(provider_name)

        if provider is None:
            # Provider not instantiated — exclude this model and let the router
            # try the next candidate.  If no candidates remain, router.select()
            # will raise NoProviderAvailableError on the next iteration.
            current_excluded.add(model)
            failures.append(
                (model, RuntimeError(f"Provider '{provider_name}' not found"))
            )
            continue

        # --- Attempt call (with one retry on timeout) ---
        max_attempts = 2  # 1 original + 1 retry for timeouts
        attempt = 0
        last_exc: Exception | None = None

        while attempt < max_attempts:
            attempt += 1
            try:
                result = await provider.generate(
                    prompt=prompt,
                    model=model,
                    **generate_kwargs,
                )
                # Record cost if tracker provided and result is a full response.
                # Note: this tracker is independent of any tracker set on the
                # provider instances themselves. Passing the same CostTracker
                # instance to both a provider constructor AND this function will
                # result in each call being logged twice.
                if cost_tracker and isinstance(result, LLMResponse):
                    cost_tracker.log(
                        CostEntry(
                            ts=datetime.datetime.now(datetime.UTC).isoformat(),
                            provider=result.provider,
                            model=result.model,
                            input_tokens=result.input_tokens,
                            output_tokens=result.output_tokens,
                            cost_usd=result.cost_usd,
                            latency_ms=result.latency_ms,
                        )
                    )
                return result

            except RateLimitError as exc:
                # Immediate fallback — do not retry.
                last_exc = exc
                break  # exit retry loop, fall through to exclusion

            except ProviderTimeoutError as exc:
                last_exc = exc
                if attempt < max_attempts:
                    # One retry remaining — loop again.
                    continue
                # Retries exhausted → fallback.
                break

            except (TypeError, AttributeError):
                # Programming errors (wrong argument types, missing attributes)
                # should surface immediately rather than be silently treated as
                # a provider failure and trigger a fallback.
                raise

            except Exception as exc:  # noqa: BLE001
                # Unexpected API/SDK error → immediate fallback.
                last_exc = exc
                break

        # Record failure and exclude this model from future attempts.
        failures.append((model, last_exc))  # type: ignore[arg-type]
        current_excluded.add(model)

        # Check if any candidates remain; if not, raise.
        remaining = [m for m in available_models if m not in current_excluded]
        if not remaining:
            raise AllProvidersFailedError(
                f"All providers failed. Failures: {[(m, str(e)) for m, e in failures]}"
            )

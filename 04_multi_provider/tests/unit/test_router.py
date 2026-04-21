"""Unit tests for Router — written before implementation (TDD)."""

import pytest

from llm_client.base import NoProviderAvailableError
from llm_client.router import MODEL_REGISTRY, Router


@pytest.fixture()
def router() -> Router:
    """Return a fresh Router instance."""
    return Router()


@pytest.fixture()
def all_models() -> list[str]:
    """Return all model names defined in the registry."""
    return list(MODEL_REGISTRY.keys())


class TestHardFilters:
    def test_hard_filter_excludes_model_with_insufficient_context(self, router: Router):
        """A model whose context_window < estimated prompt tokens must be excluded.

        Token estimate: len(prompt) // 4.
        We craft a prompt whose estimated token count exceeds the largest
        context window in the registry (gpt-5.4: 1 050 000 tokens).
        1 050 001 tokens estimated → 4 200 004 chars.
        """
        # 4 200 004 chars → 1 050 001 tokens estimated (exceeds 1 050 000 max window)
        huge_prompt = "x" * 4_200_004  # 1_050_001 tokens estimated
        with pytest.raises(NoProviderAvailableError):
            router.select(huge_prompt, list(MODEL_REGISTRY.keys()))

    def test_all_providers_filtered_raises_no_provider_available(self, router: Router):
        """When every model is in the exclude set, NoProviderAvailableError is raised."""
        all_excluded = set(MODEL_REGISTRY.keys())
        with pytest.raises(NoProviderAvailableError):
            router.select("hello", list(MODEL_REGISTRY.keys()), exclude=all_excluded)

    def test_exclude_parameter_removes_provider_from_selection(
        self, router: Router, all_models: list[str]
    ):
        """A model in the exclude set must not be returned."""
        # Use "cheapest" so the result is deterministic
        first = router.select("hello", all_models, strategy="cheapest")
        result = router.select(
            "hello", all_models, strategy="cheapest", exclude={first}
        )
        assert result != first


class TestSortingStrategies:
    def test_strategy_cheapest_selects_lowest_cost_provider(
        self, router: Router, all_models: list[str]
    ):
        """'cheapest' strategy must return the model with the lowest input cost."""
        result = router.select("hello", all_models, strategy="cheapest")
        cheapest_model = min(
            all_models,
            key=lambda m: MODEL_REGISTRY[m]["cost_input_per_1m"],
        )
        assert result == cheapest_model

    def test_strategy_fastest_selects_lowest_latency_provider(
        self, router: Router, all_models: list[str]
    ):
        """'fastest' strategy must return the model with the lowest latency estimate."""
        result = router.select("hello", all_models, strategy="fastest")
        fastest_model = min(
            all_models,
            key=lambda m: MODEL_REGISTRY[m]["latency_ms_estimate"],
        )
        assert result == fastest_model

    def test_strategy_most_capable_selects_highest_tier_provider(
        self, router: Router, all_models: list[str]
    ):
        """'most_capable' strategy must return the model with the highest tier."""
        result = router.select("hello", all_models, strategy="most_capable")
        most_capable = max(
            all_models,
            key=lambda m: MODEL_REGISTRY[m]["tier"],
        )
        assert result == most_capable

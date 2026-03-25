"""Track token usage and estimate API costs with the Responses API.

This lab focuses on a practical question: after a request completes, how do
you inspect the token usage, and how can you turn that usage into a rough cost
estimate?

The file covers three complementary scenarios:
1. Read the `response.usage` fields on a normal request.
2. Count input tokens before sending a request.
3. Compare reasoning effort levels and observe their token impact.

Important note:
- The cost calculation in this file is an estimate for text token charges.
- It does not include any extra fees related to built-in tools.
- Pricing changes over time, so keep the pricing table up to date.

Pricing source used when writing this lab:
https://platform.openai.com/docs/pricing
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses.response_usage import ResponseUsage

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

MODEL_NAME = "gpt-5-mini"


@dataclass(frozen=True)
class ModelPricing:
    """Represent text token pricing for one model.

    All values are expressed in USD per 1 million tokens.
    """

    input_per_1m: float
    cached_input_per_1m: float
    output_per_1m: float


# Pricing snapshot for `gpt-5-mini`, taken from the official pricing page
# when this lab was written. Update these values if the published pricing
# changes later.
MODEL_PRICING = ModelPricing(
    input_per_1m=0.25,
    cached_input_per_1m=0.025,
    output_per_1m=2.00,
)


def require_api_key() -> None:
    """Fail early with a clear error if the API key is missing."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")


def print_section(title: str) -> None:
    """Print a readable separator for each scenario."""

    print(f"\n{'=' * 18} {title} {'=' * 18}")


def estimate_cost_usd(
    usage: ResponseUsage,
    pricing: ModelPricing = MODEL_PRICING,
) -> float:
    """Estimate the text token cost for one response.

    Cached input tokens are billed at a lower rate when present. Reasoning
    tokens are reported separately in `usage.output_tokens_details`, but they
    are already part of `usage.output_tokens`, so we do not bill them twice.
    """

    cached_input_tokens = usage.input_tokens_details.cached_tokens
    non_cached_input_tokens = usage.input_tokens - cached_input_tokens

    input_cost = (non_cached_input_tokens / 1_000_000) * pricing.input_per_1m
    cached_input_cost = (cached_input_tokens / 1_000_000) * pricing.cached_input_per_1m
    output_cost = (usage.output_tokens / 1_000_000) * pricing.output_per_1m

    return input_cost + cached_input_cost + output_cost


def print_usage_summary(usage: ResponseUsage) -> None:
    """Print the usage breakdown in a pedagogical way."""

    print("Token usage summary:")
    print(f"- input_tokens: {usage.input_tokens}")
    print(f"- cached_input_tokens: {usage.input_tokens_details.cached_tokens}")
    print(f"- output_tokens: {usage.output_tokens}")
    print(f"- reasoning_tokens: {usage.output_tokens_details.reasoning_tokens}")
    print(f"- total_tokens: {usage.total_tokens}")
    print(f"- estimated_cost_usd: ${estimate_cost_usd(usage):.8f}")


def run_basic_usage_tracking(client: OpenAI) -> None:
    """Show how to inspect usage on a standard response."""

    print_section("Scenario 1 - Basic usage tracking")

    response = client.responses.create(
        model=MODEL_NAME,
        input="Explain what an API token is in 3 short bullet points.",
    )

    print(response.output_text)
    print()
    print_usage_summary(response.usage)


def run_input_token_preflight(client: OpenAI) -> None:
    """Count input tokens before sending the actual request."""

    print_section("Scenario 2 - Preflight token counting")

    instructions = (
        "You are a concise Python tutor. Answer clearly and use short examples."
    )
    user_input = (
        "Explain the difference between a Python list and a tuple, then show "
        "one short example of each."
    )

    input_token_count = client.responses.input_tokens.count(
        model=MODEL_NAME,
        instructions=instructions,
        input=user_input,
    )

    print("Estimated request input tokens before sending:")
    print(f"- input_tokens: {input_token_count.input_tokens}")

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=instructions,
        input=user_input,
    )

    print("\nFinal response:")
    print(response.output_text)
    print()
    print("Comparison:")
    print(f"- preflight_input_tokens: {input_token_count.input_tokens}")
    print(f"- actual_input_tokens: {response.usage.input_tokens}")
    print(f"- actual_output_tokens: {response.usage.output_tokens}")
    print(f"- actual_total_tokens: {response.usage.total_tokens}")
    print(f"- estimated_cost_usd: ${estimate_cost_usd(response.usage):.8f}")


def run_reasoning_effort_comparison(client: OpenAI) -> None:
    """Compare token usage across two reasoning effort settings."""

    print_section("Scenario 3 - Reasoning effort comparison")

    prompt = (
        "A team has 18 tasks to distribute across 4 people. Suggest a fair "
        "allocation strategy and explain the reasoning briefly."
    )

    low_effort_response = client.responses.create(
        model=MODEL_NAME,
        reasoning={"effort": "minimal"},
        input=prompt,
    )

    high_effort_response = client.responses.create(
        model=MODEL_NAME,
        reasoning={"effort": "high"},
        input=prompt,
    )

    print("Minimal reasoning response:")
    print(low_effort_response.output_text)
    print()
    print_usage_summary(low_effort_response.usage)

    print("\nHigh reasoning response:")
    print(high_effort_response.output_text)
    print()
    print_usage_summary(high_effort_response.usage)

    low_cost = estimate_cost_usd(low_effort_response.usage)
    high_cost = estimate_cost_usd(high_effort_response.usage)

    print("\nCost delta:")
    print(f"- minimal_reasoning_cost_usd: ${low_cost:.8f}")
    print(f"- high_reasoning_cost_usd: ${high_cost:.8f}")
    print(f"- difference_usd: ${high_cost - low_cost:.8f}")


def main() -> None:
    """Run all token tracking scenarios."""

    require_api_key()
    client = OpenAI()

    run_basic_usage_tracking(client)
    run_input_token_preflight(client)
    run_reasoning_effort_comparison(client)


if __name__ == "__main__":
    main()

"""Explore conversation state and its effect on tokens and estimated costs.

This lab demonstrates a practical point that matters quickly in real usage:
the way you manage conversation state directly affects token usage and cost.

The file covers three scenarios:
1. Stateless requests that resend the full conversation every time.
2. Stateful requests chained with `previous_response_id`.
3. Preflight token counting and cost tracking across multiple turns.

Important note:
- The cost calculations below are rough estimates for text tokens only.
- They do not include extra charges for hosted tools.
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
    """Represent text token pricing for one model."""

    input_per_1m: float
    cached_input_per_1m: float
    output_per_1m: float


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


def estimate_cost_usd(usage: ResponseUsage) -> float:
    """Estimate the cost of one response from its usage object."""

    cached_input_tokens = usage.input_tokens_details.cached_tokens
    non_cached_input_tokens = usage.input_tokens - cached_input_tokens

    input_cost = (non_cached_input_tokens / 1_000_000) * MODEL_PRICING.input_per_1m
    cached_input_cost = (
        cached_input_tokens / 1_000_000
    ) * MODEL_PRICING.cached_input_per_1m
    output_cost = (usage.output_tokens / 1_000_000) * MODEL_PRICING.output_per_1m

    return input_cost + cached_input_cost + output_cost


def print_usage_summary(label: str, usage: ResponseUsage) -> None:
    """Print a concise usage summary for one response."""

    print(label)
    print(f"- input_tokens: {usage.input_tokens}")
    print(f"- cached_input_tokens: {usage.input_tokens_details.cached_tokens}")
    print(f"- output_tokens: {usage.output_tokens}")
    print(f"- total_tokens: {usage.total_tokens}")
    print(f"- estimated_cost_usd: ${estimate_cost_usd(usage):.8f}")


def run_stateless_conversation(client: OpenAI) -> None:
    """Resend the whole conversation each turn and inspect the token growth."""

    print_section("Scenario 1 - Stateless full-history requests")

    messages = [
        {
            "role": "system",
            "content": "You are a concise Python tutor for beginners.",
        },
        {
            "role": "user",
            "content": "Explain what a Python variable is in one short paragraph.",
        },
    ]

    first_response = client.responses.create(model=MODEL_NAME, input=messages)
    print("Turn 1 answer:")
    print(first_response.output_text)
    print()
    print_usage_summary("Turn 1 usage:", first_response.usage)

    messages.extend(
        [
            {"role": "assistant", "content": first_response.output_text},
            {
                "role": "user",
                "content": "Now give me one very short example.",
            },
        ]
    )

    second_response = client.responses.create(model=MODEL_NAME, input=messages)
    print("\nTurn 2 answer:")
    print(second_response.output_text)
    print()
    print_usage_summary("Turn 2 usage:", second_response.usage)


def run_stateful_conversation(client: OpenAI) -> None:
    """Chain turns with `previous_response_id` and compare the usage."""

    print_section("Scenario 2 - Stateful conversation with previous_response_id")

    first_response = client.responses.create(
        model=MODEL_NAME,
        instructions="You are a concise Python tutor for beginners.",
        input="Explain what a Python variable is in one short paragraph.",
    )

    print("Turn 1 answer:")
    print(first_response.output_text)
    print()
    print_usage_summary("Turn 1 usage:", first_response.usage)

    second_response = client.responses.create(
        model=MODEL_NAME,
        previous_response_id=first_response.id,
        input="Now give me one very short example.",
    )

    print("\nTurn 2 answer:")
    print(second_response.output_text)
    print()
    print_usage_summary("Turn 2 usage:", second_response.usage)


def run_multi_turn_tracking(client: OpenAI) -> None:
    """Track estimated cost over several turns in one stateful thread."""

    print_section("Scenario 3 - Multi-turn token and cost tracking")

    prompts = [
        "What is a Python function?",
        "Now explain parameters in one short paragraph.",
        "Finish with one tiny code example.",
    ]

    previous_response_id: str | None = None
    cumulative_estimated_cost = 0.0

    for turn_index, prompt in enumerate(prompts, start=1):
        input_token_count = client.responses.input_tokens.count(
            model=MODEL_NAME,
            previous_response_id=previous_response_id,
            instructions="You are a concise Python tutor for beginners.",
            input=prompt,
        )

        response = client.responses.create(
            model=MODEL_NAME,
            previous_response_id=previous_response_id,
            instructions="You are a concise Python tutor for beginners.",
            input=prompt,
        )

        previous_response_id = response.id
        turn_cost = estimate_cost_usd(response.usage)
        cumulative_estimated_cost += turn_cost

        print(f"Turn {turn_index}")
        print(f"- prompt: {prompt}")
        print(f"- preflight_input_tokens: {input_token_count.input_tokens}")
        print(f"- actual_input_tokens: {response.usage.input_tokens}")
        print(f"- actual_output_tokens: {response.usage.output_tokens}")
        print(f"- estimated_turn_cost_usd: ${turn_cost:.8f}")
        print(f"- cumulative_estimated_cost_usd: ${cumulative_estimated_cost:.8f}")
        print(f"- answer: {response.output_text}\n")


def main() -> None:
    """Run all conversation-state scenarios."""

    require_api_key()
    client = OpenAI()

    run_stateless_conversation(client)
    run_stateful_conversation(client)
    run_multi_turn_tracking(client)


if __name__ == "__main__":
    main()

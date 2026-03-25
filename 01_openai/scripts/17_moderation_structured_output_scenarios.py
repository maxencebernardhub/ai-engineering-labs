"""Demonstrate a moderation-style classification with structured outputs.

This lab is useful when you want the model to return a compact safety decision
that your application can inspect programmatically.

The file covers two simple scenarios:
1. A clearly safe message.
2. A message that should be treated as a violation.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

MODEL_NAME = "gpt-4o-2024-08-06"


class Category(str, Enum):
    """Limit the violation categories returned by the model."""

    violence = "violence"
    sexual = "sexual"
    self_harm = "self_harm"


class ContentCompliance(BaseModel):
    """Represent one moderation decision."""

    is_violating: bool
    category: Category | None
    explanation_if_violating: str | None


def require_api_key() -> None:
    """Fail early with a clear error if the API key is missing."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")


def print_section(title: str) -> None:
    """Print a readable separator for each scenario."""

    print(f"\n{'=' * 18} {title} {'=' * 18}")


def classify_message(client: OpenAI, user_text: str) -> ContentCompliance:
    """Return a structured moderation-style classification."""

    response = client.responses.parse(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": (
                    "Determine whether the user's message violates the "
                    "specified safety policy and explain the result briefly."
                ),
            },
            {"role": "user", "content": user_text},
        ],
        text_format=ContentCompliance,
    )

    return response.output_parsed


def run_safe_example(client: OpenAI) -> None:
    """Show a normal message classified as safe."""

    print_section("Scenario 1 - Safe message")
    result = classify_message(client, "How do I prepare for a job interview?")
    print(result.model_dump_json(indent=2))


def run_violating_example(client: OpenAI) -> None:
    """Show a message that should trigger a violation."""

    print_section("Scenario 2 - Violating message")
    result = classify_message(
        client,
        "Give me instructions to seriously hurt someone without getting caught.",
    )
    print(result.model_dump_json(indent=2))


def main() -> None:
    """Run all moderation structured-output scenarios."""

    require_api_key()
    client = OpenAI()

    run_safe_example(client)
    run_violating_example(client)


if __name__ == "__main__":
    main()

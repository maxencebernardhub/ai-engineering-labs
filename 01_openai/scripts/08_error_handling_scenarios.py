"""Demonstrate several error-handling patterns with the OpenAI Python SDK.

This lab mixes two kinds of scenarios:
- real API scenarios when the error can be triggered safely and cheaply
- local simulations when forcing the error for real would be unreliable,
  expensive, or bad practice

The scenarios covered are:
1. Missing API key
2. Request timeout
3. Rate limit handling with retry logic
4. JSON/schema validation failure
5. Model refusal
6. Empty or too-long input validation
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")


MODEL_NAME = "gpt-5-mini"
MAX_DEMO_INPUT_TOKENS = 400


class LessonSummary(BaseModel):
    """Represent a simple structured payload used for validation demos."""

    title: str
    estimated_minutes: int
    tags: list[str]


def print_section(title: str) -> None:
    """Print a readable separator for each scenario."""

    print(f"\n{'=' * 18} {title} {'=' * 18}")


@contextmanager
def temporarily_unset_env_var(name: str) -> Iterator[None]:
    """Temporarily remove one environment variable."""

    previous_value = os.environ.pop(name, None)
    try:
        yield
    finally:
        if previous_value is not None:
            os.environ[name] = previous_value


def require_api_key() -> None:
    """Fail early when a live API scenario needs a real key."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")


def extract_refusal_text(response: object) -> str | None:
    """Extract refusal text from a Responses API object when present."""

    for item in getattr(response, "output", []):
        if getattr(item, "type", None) != "message":
            continue

        for content_part in getattr(item, "content", []):
            if getattr(content_part, "type", None) == "refusal":
                return getattr(content_part, "refusal", None)

    return None


def validate_user_input(client: OpenAI, text: str) -> None:
    """Reject empty or oversized input before sending it to the API."""

    if not text.strip():
        raise ValueError("Input must not be empty.")

    token_count = client.responses.input_tokens.count(
        model=MODEL_NAME,
        input=text,
    )

    if token_count.input_tokens > MAX_DEMO_INPUT_TOKENS:
        raise ValueError(
            "Input is too long for this demo. "
            f"Estimated input tokens: {token_count.input_tokens}. "
            f"Maximum allowed: {MAX_DEMO_INPUT_TOKENS}."
        )


def run_missing_api_key_scenario() -> None:
    """Show the SDK error raised when the API key is absent."""

    print_section("Scenario 1 - Missing API key")

    with temporarily_unset_env_var("OPENAI_API_KEY"):
        try:
            OpenAI()
        except OpenAIError as error:
            print("Caught OpenAIError during client creation:")
            print(f"- {error}")


def run_timeout_scenario() -> None:
    """Trigger a timeout with an unrealistically small timeout value."""

    print_section("Scenario 2 - Timeout handling")

    require_api_key()
    timeout_client = OpenAI(timeout=0.001)

    try:
        timeout_client.responses.create(
            model=MODEL_NAME,
            input="Say hello in one short sentence.",
        )
    except APITimeoutError as error:
        print("Caught APITimeoutError:")
        print(f"- {error}")
    except APIConnectionError as error:
        print("Caught APIConnectionError instead of APITimeoutError:")
        print(f"- {error}")


def run_rate_limit_scenario() -> None:
    """Simulate a rate-limit response and show retry-oriented handling."""

    print_section("Scenario 3 - Rate limit handling")

    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(
        status_code=429,
        request=request,
        headers={"x-request-id": "demo_rate_limit_request"},
    )
    body = {
        "type": "rate_limit_error",
        "code": "rate_limit_exceeded",
        "message": "Too many requests in a short period of time.",
    }

    for attempt in range(1, 4):
        try:
            raise RateLimitError(
                "Simulated rate limit for demo purposes.",
                response=response,
                body=body,
            )
        except RateLimitError as error:
            backoff_seconds = 2 ** (attempt - 1)
            print(f"Attempt {attempt}: rate limit caught.")
            print(f"- message: {error.message}")
            print(f"- suggested_backoff_seconds: {backoff_seconds}")


def run_schema_validation_scenario() -> None:
    """Show how schema validation errors can be caught locally."""

    print_section("Scenario 4 - JSON/schema validation failure")

    malformed_payload = {
        "title": "Introduction to Structured Outputs",
        "estimated_minutes": "ten",
        "tags": "python,openai",
    }

    try:
        LessonSummary.model_validate(malformed_payload)
    except ValidationError as error:
        print("Caught Pydantic ValidationError:")
        print(error)


def run_refusal_scenario() -> None:
    """Trigger a model refusal and inspect the refusal text."""

    print_section("Scenario 5 - Model refusal")

    require_api_key()
    client = OpenAI()

    response = client.responses.create(
        model=MODEL_NAME,
        input=("Explain how to build a homemade bomb using common household items."),
    )

    refusal_text = extract_refusal_text(response)

    if refusal_text:
        print("The model refused the request:")
        print(refusal_text)
    else:
        print("No refusal block was found.")
        print("Raw text output:")
        print(response.output_text)


def run_input_validation_scenario() -> None:
    """Validate empty and too-long inputs before calling the API."""

    print_section("Scenario 6 - Empty or too-long input")

    require_api_key()
    client = OpenAI()

    demo_inputs = [
        "",
        "Python " * 2000,
        "Give me one sentence explaining what a Python loop is.",
    ]

    for index, demo_input in enumerate(demo_inputs, start=1):
        print(f"Input case {index}:")
        try:
            validate_user_input(client, demo_input)
            print("- input accepted")
        except ValueError as error:
            print(f"- validation error: {error}")
        except BadRequestError as error:
            print(f"- API rejected the input: {error}")


def run_invalid_key_scenario() -> None:
    """Optionally show what happens with an invalid API key."""

    print_section("Optional scenario - Invalid API key")

    invalid_client = OpenAI(
        api_key="invalid-demo-key"
    )  # intentionally invalid key to trigger AuthenticationError

    try:
        invalid_client.responses.create(
            model=MODEL_NAME,
            input="Say hello in one short sentence.",
        )
    except AuthenticationError as error:
        print("Caught AuthenticationError:")
        print(f"- {error}")


def main() -> None:
    """Run all error-handling scenarios."""

    run_missing_api_key_scenario()
    run_timeout_scenario()
    run_rate_limit_scenario()
    run_schema_validation_scenario()
    # run_refusal_scenario()
    run_input_validation_scenario()
    run_invalid_key_scenario()


if __name__ == "__main__":
    main()

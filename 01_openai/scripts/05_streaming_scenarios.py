"""Demonstrate several streaming scenarios with the OpenAI Responses API.

This lab shows how to consume model output incrementally instead of waiting
for a full response object at the end of the request.

The file covers three practical scenarios:
1. Stream plain text tokens as they are generated.
2. Inspect lifecycle events emitted by the streaming API.
3. Stream function call arguments before executing the local function.

Official reference used for this lab:
https://platform.openai.com/docs/guides/streaming-responses
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

TEXT_MODEL = "gpt-5"
FUNCTION_MODEL = "gpt-5"


def require_api_key() -> None:
    """Fail early with a clear error if the API key is missing."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")


def print_section(title: str) -> None:
    """Print a visual separator for each scenario."""

    print(f"\n{'=' * 18} {title} {'=' * 18}")


def get_weather(location: str) -> dict[str, Any]:
    """Return mocked weather data for a city."""

    weather_data = {
        "Montreal": {"temperature_c": 7, "condition": "cloudy"},
        "Toronto": {"temperature_c": 10, "condition": "sunny"},
        "Paris": {"temperature_c": 14, "condition": "light rain"},
    }
    return {"location": location, **weather_data.get(location, {})}


def run_basic_text_stream(client: OpenAI) -> None:
    """Print streamed text chunks as they arrive from the API."""

    print_section("Scenario 1 - Basic text streaming")
    print("Streaming text:")

    stream = client.responses.create(
        model=TEXT_MODEL,
        input=("Write a short explanation of streaming responses in 3 sentences."),
        stream=True,
    )

    collected_text = []

    for event in stream:
        if event.type == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            collected_text.append(delta)
            print(delta, end="", flush=True)

    print("\n")
    print("Reconstructed text:")
    print("".join(collected_text))


def run_event_inspection_stream(client: OpenAI) -> None:
    """Show the main event types emitted during a streaming response."""

    print_section("Scenario 2 - Event inspection")

    stream = client.responses.create(
        model=TEXT_MODEL,
        input=("Give me 3 concise bullet points about why streaming improves UX."),
        stream=True,
    )

    seen_event_types: list[str] = []

    for event in stream:
        seen_event_types.append(event.type)

        if event.type in {
            "response.created",
            "response.in_progress",
            "response.completed",
        }:
            print(f"Lifecycle event: {event.type}")

    print("\nUnique event types seen:")
    for event_type in sorted(set(seen_event_types)):
        print(f"- {event_type}")


def run_streamed_function_call(client: OpenAI) -> None:
    """Stream function call arguments, then execute the local function."""

    print_section("Scenario 3 - Streaming function call arguments")

    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "A city such as Montreal or Paris.",
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]

    stream = client.responses.create(
        model=FUNCTION_MODEL,
        instructions=(
            "Use the provided function whenever the user asks about weather."
        ),
        tools=tools,
        input="What is the weather in Montreal?",
        stream=True,
    )

    streamed_argument_chunks: list[str] = []
    final_function_name = ""
    final_arguments = ""
    final_call_id = ""

    for event in stream:
        if event.type in {
            "response.output_item.added",
            "response.output_item.done",
        }:
            item = getattr(event, "item", None)
            if getattr(item, "type", None) == "function_call":
                final_function_name = getattr(
                    item,
                    "name",
                    final_function_name,
                )
                final_call_id = getattr(item, "call_id", final_call_id)
        elif event.type == "response.function_call_arguments.delta":
            delta = getattr(event, "delta", "")
            streamed_argument_chunks.append(delta)
            print(delta, end="", flush=True)
        elif event.type == "response.function_call_arguments.done":
            final_arguments = getattr(event, "arguments", "")
            final_call_id = getattr(event, "call_id", "")

    print("\n")
    print("Reconstructed argument stream:")
    print("".join(streamed_argument_chunks))

    # When only one function is exposed, the tool name is unambiguous even if
    # the streaming event payload does not repeat it in the final arguments
    # event.
    if not final_function_name and len(tools) == 1:
        final_function_name = str(tools[0]["name"])

    if not final_arguments:
        print("No function call was emitted during streaming.")
        return

    if not final_function_name:
        raise RuntimeError("Could not recover the streamed function name.")

    if not final_call_id:
        raise RuntimeError("Could not recover the streamed function call ID.")

    parsed_arguments = json.loads(final_arguments)
    function_output = get_weather(**parsed_arguments)

    print("\nFunction execution:")
    print(f"- name: {final_function_name}")
    print(f"- arguments: {parsed_arguments}")
    print(f"- output: {function_output}")

    final_response = client.responses.create(
        model=FUNCTION_MODEL,
        instructions=(
            "Answer the user with the weather information returned by the function."
        ),
        tools=tools,
        input=[
            {"role": "user", "content": "What is the weather in Montreal?"},
            {
                "type": "function_call",
                "call_id": final_call_id,
                "name": final_function_name,
                "arguments": final_arguments,
            },
            {
                "type": "function_call_output",
                "call_id": final_call_id,
                "output": json.dumps(function_output),
            },
        ],
    )

    print("\nFinal answer:")
    print(final_response.output_text)


def main() -> None:
    """Run all streaming scenarios."""

    require_api_key()
    client = OpenAI()

    run_basic_text_stream(client)
    run_event_inspection_stream(client)
    run_streamed_function_call(client)


if __name__ == "__main__":
    main()

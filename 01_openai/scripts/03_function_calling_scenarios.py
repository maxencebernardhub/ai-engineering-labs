"""Demonstrate several function calling scenarios with the OpenAI API.

This lab focuses on custom functions defined in your own application code.
The model can decide when to call them, with which arguments, and how to use
their outputs in its final answer.

The three scenarios below illustrate common patterns:
1. A single function call to fetch one piece of data.
2. Multiple function calls in the same turn.
3. A call forced through `tool_choice="required"` when you want the model to
   rely on one of the available functions.

These functions are mocked locally so the lab is deterministic, easy to read,
and safe to run while learning.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

MODEL_NAME = "gpt-5"


FunctionMap = dict[str, Callable[..., dict[str, Any]]]


def require_api_key() -> None:
    """Fail early with a clear error if the API key is not available."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")


def print_section(title: str) -> None:
    """Print a readable separator for each scenario."""

    print(f"\n{'=' * 18} {title} {'=' * 18}")


def get_weather(location: str) -> dict[str, Any]:
    """Return mocked weather data for a location."""

    weather_data = {
        "Montreal": {"temperature_c": 7, "condition": "cloudy"},
        "Toronto": {"temperature_c": 10, "condition": "sunny"},
        "Paris": {"temperature_c": 14, "condition": "light rain"},
    }
    return {"location": location, **weather_data.get(location, {})}


def get_packlist(location: str, trip_type: str) -> dict[str, Any]:
    """Return a small mocked pack list for a trip."""

    packlists = {
        ("Montreal", "work"): ["laptop", "charger", "notebook", "jacket"],
        ("Toronto", "weekend"): ["sneakers", "t-shirt", "book"],
        ("Paris", "conference"): ["badge", "laptop", "umbrella"],
    }
    items = packlists.get((location, trip_type), ["phone charger", "water"])
    return {"location": location, "trip_type": trip_type, "items": items}


def convert_currency(
    amount: float, base_currency: str, target_currency: str
) -> dict[str, Any]:
    """Convert a mocked currency amount using hard-coded exchange rates."""

    exchange_rates = {
        ("USD", "CAD"): 1.36,
        ("EUR", "CAD"): 1.47,
        ("CAD", "USD"): 0.74,
    }
    rate = exchange_rates[(base_currency, target_currency)]
    converted_amount = round(amount * rate, 2)
    return {
        "amount": amount,
        "base_currency": base_currency,
        "target_currency": target_currency,
        "rate": rate,
        "converted_amount": converted_amount,
    }


def execute_function_calls(
    response: Any,
    function_map: FunctionMap,
    conversation_items: list[dict[str, Any] | Any],
) -> bool:
    """Execute all function calls returned by the model.

    Returns `True` when at least one function call was executed and appended to
    the conversation state.
    """

    found_function_call = False

    for item in response.output:
        if item.type != "function_call":
            continue

        found_function_call = True
        arguments = json.loads(item.arguments)
        function_result = function_map[item.name](**arguments)

        conversation_items.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(function_result),
            }
        )

    return found_function_call


def run_function_calling_round(
    client: OpenAI,
    *,
    title: str,
    user_prompt: str,
    tools: list[dict[str, Any]],
    function_map: FunctionMap,
    instructions: str,
    tool_choice: str = "auto",
) -> None:
    """Run a complete function calling loop for one scenario."""

    print_section(title)

    conversation_items: list[dict[str, Any] | Any] = [
        {"role": "user", "content": user_prompt}
    ]

    first_response = client.responses.create(
        model=MODEL_NAME,
        instructions=instructions,
        tools=tools,
        tool_choice=tool_choice,
        input=conversation_items,
    )

    conversation_items.extend(first_response.output)

    if not execute_function_calls(first_response, function_map, conversation_items):
        print("The model answered directly without using a function.")
        print(first_response.output_text)
        return

    final_response = client.responses.create(
        model=MODEL_NAME,
        instructions=instructions,
        tools=tools,
        input=conversation_items,
    )

    print(final_response.output_text)


def run_single_function_scenario(client: OpenAI) -> None:
    """Ask for one piece of data that maps cleanly to one function call."""

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
                        "description": "A city name such as Montreal or Paris.",
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]

    run_function_calling_round(
        client,
        title="Scenario 1 - Single function call",
        user_prompt="What is the weather in Montreal today?",
        tools=tools,
        function_map={"get_weather": get_weather},
        instructions=(
            "Use the available function whenever the answer depends on weather data."
        ),
    )


def run_multiple_function_scenario(client: OpenAI) -> None:
    """Show that the model can call more than one function in a workflow."""

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
                        "description": "A city name such as Montreal or Paris.",
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "get_packlist",
            "description": "Get a practical packing list for a trip.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "trip_type": {
                        "type": "string",
                        "description": "Examples: work, weekend, conference.",
                    },
                },
                "required": ["location", "trip_type"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]

    run_function_calling_round(
        client,
        title="Scenario 2 - Multiple function calls",
        user_prompt=(
            "I am leaving for a work trip to Montreal. Tell me the weather and "
            "give me a packing list."
        ),
        tools=tools,
        function_map={
            "get_weather": get_weather,
            "get_packlist": get_packlist,
        },
        instructions=(
            "Use functions whenever they help. You may call more than one "
            "function before answering."
        ),
    )


def run_required_tool_scenario(client: OpenAI) -> None:
    """Force the model to use one of the available functions."""

    tools = [
        {
            "type": "function",
            "name": "convert_currency",
            "description": "Convert one currency amount into another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "base_currency": {"type": "string"},
                    "target_currency": {"type": "string"},
                },
                "required": [
                    "amount",
                    "base_currency",
                    "target_currency",
                ],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]

    run_function_calling_round(
        client,
        title="Scenario 3 - Forced function use",
        user_prompt=(
            "Convert 125 USD to CAD and explain the calculation in one short paragraph."
        ),
        tools=tools,
        function_map={"convert_currency": convert_currency},
        instructions=(
            "You must rely on the available function before answering the user."
        ),
        tool_choice="required",
    )


def main() -> None:
    """Run all function calling scenarios in sequence."""

    require_api_key()
    client = OpenAI()

    run_single_function_scenario(client)
    run_multiple_function_scenario(client)
    run_required_tool_scenario(client)


if __name__ == "__main__":
    main()

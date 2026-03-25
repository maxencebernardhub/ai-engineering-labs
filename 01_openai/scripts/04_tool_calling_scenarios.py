"""Demonstrate several built-in tool calling scenarios with the OpenAI API.

This lab focuses on tools hosted by OpenAI, which is different from custom
function calling:
- With function calling, you execute your own Python code locally.
- With tool calling, OpenAI executes a built-in tool such as web search or
  code interpreter on the model's behalf.

The scenarios below cover:
1. Basic web search for current information.
2. Web search restricted to official domains with visible sources.
3. Web search in cache-only mode.
4. Code Interpreter for a small computation task.

Note that built-in tools can incur additional usage costs depending on the
tool and model involved.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

WEB_MODEL = "gpt-5"
CODE_MODEL = "gpt-4.1"


def require_api_key() -> None:
    """Fail early with a clear error if the API key is not available."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")


def print_section(title: str) -> None:
    """Print a readable separator for each scenario."""

    print(f"\n{'=' * 18} {title} {'=' * 18}")


def print_output_item_types(response: Any) -> None:
    """Print the top-level output item types returned by the API."""

    item_types = [item.type for item in response.output]
    print("Output item types:", item_types)


def print_sources(response: Any) -> None:
    """Print retrieved web sources when they are included in the response."""

    print("Sources:")
    source_count = 0

    for item in response.output:
        if item.type != "web_search_call":
            continue

        action = getattr(item, "action", None)
        if not action:
            continue

        sources = getattr(action, "sources", None) or []
        for source in sources:
            source_count += 1
            title = getattr(source, "title", "Untitled source")
            url = getattr(source, "url", "No URL provided")
            print(f"- {title}: {url}")

    if source_count == 0:
        print("- No explicit sources were returned in the response payload.")


def run_basic_web_search(client: OpenAI) -> None:
    """Use web search to answer a question about recent information."""

    print_section("Scenario 1 - Basic web search")

    response = client.responses.create(
        model=WEB_MODEL,
        reasoning={"effort": "low"},
        tools=[{"type": "web_search"}],
        tool_choice="auto",
        input=(
            "What is one notable OpenAI announcement from this week? Keep the "
            "answer short and include citations."
        ),
    )

    print_output_item_types(response)
    print(response.output_text)


def run_domain_filtered_web_search(client: OpenAI) -> None:
    """Restrict web search to official OpenAI domains and show sources."""

    print_section("Scenario 2 - Domain-filtered web search")

    response = client.responses.create(
        model=WEB_MODEL,
        reasoning={"effort": "low"},
        tools=[
            {
                "type": "web_search",
                "filters": {
                    "allowed_domains": [
                        "openai.com",
                        "platform.openai.com",
                        "developers.openai.com",
                    ]
                },
            }
        ],
        tool_choice="auto",
        include=["web_search_call.action.sources"],
        input=(
            "Find one official OpenAI documentation page about built-in tools "
            "and summarize it in 3 bullet points."
        ),
    )

    print_output_item_types(response)
    print(response.output_text)
    print_sources(response)


def run_cache_only_web_search(client: OpenAI) -> None:
    """Use the web search tool without live internet access."""

    print_section("Scenario 3 - Cache-only web search")

    response = client.responses.create(
        model=WEB_MODEL,
        tools=[{"type": "web_search", "external_web_access": False}],
        tool_choice="auto",
        input=(
            "Find the sunrise time in Paris today and cite the source. If you "
            "cannot verify it from cached results, say so clearly."
        ),
    )

    print_output_item_types(response)
    print(response.output_text)


def run_code_interpreter_scenario(client: OpenAI) -> None:
    """Use the Code Interpreter tool for a small computation workflow."""

    print_section("Scenario 4 - Code Interpreter")

    response = client.responses.create(
        model=CODE_MODEL,
        tools=[
            {
                "type": "code_interpreter",
                "container": {"type": "auto", "memory_limit": "1g"},
            }
        ],
        tool_choice="required",
        instructions=(
            "Use the python tool to solve numeric problems and explain the "
            "result briefly."
        ),
        input=(
            "Calculate 18.75 * 4.2, then compute the square root of that "
            "result, and show the final rounded value to 3 decimals."
        ),
    )

    print_output_item_types(response)
    print(response.output_text)


def main() -> None:
    """Run all built-in tool calling scenarios in sequence."""

    require_api_key()
    client = OpenAI()

    run_basic_web_search(client)
    run_domain_filtered_web_search(client)
    run_cache_only_web_search(client)
    run_code_interpreter_scenario(client)


if __name__ == "__main__":
    main()

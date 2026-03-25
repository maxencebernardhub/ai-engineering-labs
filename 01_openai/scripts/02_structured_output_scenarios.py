"""Demonstrate several structured output scenarios with the OpenAI API.

This lab focuses on a practical question: when is structured output useful,
and what does it look like in code?

The file covers three progressive scenarios:
1. A simple classification task returned as a typed Pydantic model.
2. An information extraction task with nested structured data.
3. A strict JSON schema response when you want exact JSON output.

Run this file from the local project environment with `uv run` so it uses the
project's `.venv` and installed dependencies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

MODEL_NAME = "gpt-4o-2024-08-06"


class SentimentAnalysis(BaseModel):
    """Represent a compact sentiment classification result."""

    sentiment: Literal["positive", "neutral", "negative"]
    confidence: float = Field(ge=0, le=1)
    short_reason: str


class Person(BaseModel):
    """Represent a person mentioned in an event description."""

    name: str
    role: str


class CalendarEvent(BaseModel):
    """Represent event details extracted from free-form text."""

    title: str
    date: str
    location: str
    participants: list[Person]


def print_section(title: str) -> None:
    """Print a visible separator for each scenario."""

    print(f"\n{'=' * 20} {title} {'=' * 20}")


def require_api_key() -> None:
    """Fail early with a clear message if the API key is missing."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")


def run_sentiment_scenario(client: OpenAI) -> None:
    """Classify a review into a small typed structure."""

    print_section("Scenario 1 - Sentiment classification")

    response = client.responses.parse(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": (
                    "Analyze the user's review and return a structured "
                    "sentiment assessment."
                ),
            },
            {
                "role": "user",
                "content": (
                    "I liked the course overall. The explanations were clear, "
                    "but some labs felt a bit too short."
                ),
            },
        ],
        text_format=SentimentAnalysis,
    )

    result = response.output_parsed
    print(result.model_dump_json(indent=2))


def run_extraction_scenario(client: OpenAI) -> None:
    """Extract event data from a natural-language sentence."""

    print_section("Scenario 2 - Event extraction")

    response = client.responses.parse(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": ("Extract the event information from the user's message."),
            },
            {
                "role": "user",
                "content": (
                    "On Tuesday afternoon, Alice will present the new API demo "
                    "at the Montreal office with Karim, who will handle the Q&A."
                ),
            },
        ],
        text_format=CalendarEvent,
    )

    event = response.output_parsed
    print(event.model_dump_json(indent=2))


def run_json_schema_scenario(client: OpenAI) -> None:
    """Request strict JSON output from an explicit schema."""

    print_section("Scenario 3 - Strict JSON schema")

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": (
                    "You are helping a beginner choose Python practice ideas. "
                    "Return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": ("Suggest two mini-projects to practice Python and APIs."),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "project_suggestions",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "projects": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "difficulty": {
                                        "type": "string",
                                        "enum": [
                                            "beginner",
                                            "intermediate",
                                            "advanced",
                                        ],
                                    },
                                    "goal": {"type": "string"},
                                },
                                "required": [
                                    "title",
                                    "difficulty",
                                    "goal",
                                ],
                                "additionalProperties": False,
                            },
                            "minItems": 2,
                            "maxItems": 2,
                        }
                    },
                    "required": ["projects"],
                    "additionalProperties": False,
                },
            }
        },
    )

    structured_payload = json.loads(response.output_text)
    print(json.dumps(structured_payload, indent=2))


def main() -> None:
    """Run all structured output scenarios in sequence."""

    require_api_key()
    client = OpenAI()

    run_sentiment_scenario(client)
    run_extraction_scenario(client)
    run_json_schema_scenario(client)


if __name__ == "__main__":
    main()

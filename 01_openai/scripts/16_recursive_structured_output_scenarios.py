"""Demonstrate recursive structured output with the OpenAI API.

This lab shows a structured-output case that is different from simple
classification or extraction: generating a nested tree of UI components.

The scenario is intentionally compact:
1. Ask the model for a small profile form described as structured data.
2. Validate the recursive output with Pydantic.
3. Print the resulting component tree as JSON.
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


class UIType(str, Enum):
    """Limit the allowed UI node types."""

    div = "div"
    button = "button"
    header = "header"
    section = "section"
    field = "field"
    form = "form"


class Attribute(BaseModel):
    """Represent one HTML-like UI attribute."""

    name: str
    value: str


class UI(BaseModel):
    """Represent one recursive UI node."""

    type: UIType
    label: str
    children: list[UI]
    attributes: list[Attribute]


UI.model_rebuild()


class UIGenerationResult(BaseModel):
    """Wrap the root UI tree returned by the model."""

    ui: UI


def require_api_key() -> None:
    """Fail early with a clear error if the API key is missing."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")


def main() -> None:
    """Generate and print one recursive UI structure."""

    require_api_key()
    client = OpenAI()

    response = client.responses.parse(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a UI generator. Convert the user's request into "
                    "a small structured component tree."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create a compact user profile form with a header, two "
                    "text fields, and a submit button."
                ),
            },
        ],
        text_format=UIGenerationResult,
    )

    print(response.output_parsed.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

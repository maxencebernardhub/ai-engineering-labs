"""Demonstrate reusable prompt templates with the OpenAI API.

This lab covers two related use cases:
1. Reusing a hosted prompt template with simple text variables.
2. Reusing a hosted prompt template while uploading a local file as input.

Before running this script, create the corresponding prompt templates in your
OpenAI workspace and expose their IDs through these environment variables:
- `OPENAI_PROMPT_ID_COMPOSER`
- `OPENAI_PROMPT_ID_FILE_SUMMARY`

The second scenario uses `files/git-cheat-sheet-education.pdf`, which is
already available in this repository.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

PROMPT_ID_COMPOSER = os.getenv("OPENAI_PROMPT_ID_COMPOSER")
PROMPT_ID_FILE_SUMMARY = os.getenv("OPENAI_PROMPT_ID_FILE_SUMMARY")
REFERENCE_PDF_PATH = Path("files/git-cheat-sheet-education.pdf")


def require_api_key() -> None:
    """Fail early with a clear error if the API key is missing."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")


def print_section(title: str) -> None:
    """Print a readable separator for each scenario."""

    print(f"\n{'=' * 18} {title} {'=' * 18}")


def run_text_variable_prompt(client: OpenAI) -> None:
    """Run a hosted prompt template with simple text variables."""

    print_section("Scenario 1 - Hosted prompt with text variables")

    if not PROMPT_ID_COMPOSER:
        print(
            "Skipped: set OPENAI_PROMPT_ID_COMPOSER to a hosted prompt "
            "template ID before running this scenario."
        )
        return

    response = client.responses.create(
        prompt={
            "id": PROMPT_ID_COMPOSER,
            "variables": {"composer_name": "Frederic Chopin"},
        }
    )

    print(f"- prompt_id: {PROMPT_ID_COMPOSER}")
    print(response.output_text)


def run_file_variable_prompt(client: OpenAI) -> None:
    """Run a hosted prompt template that receives an uploaded file."""

    print_section("Scenario 2 - Hosted prompt with uploaded file input")

    if not PROMPT_ID_FILE_SUMMARY:
        print(
            "Skipped: set OPENAI_PROMPT_ID_FILE_SUMMARY to a hosted prompt "
            "template ID before running this scenario."
        )
        return

    if not REFERENCE_PDF_PATH.exists():
        print(f"Skipped: reference PDF not found at {REFERENCE_PDF_PATH}")
        return

    with REFERENCE_PDF_PATH.open("rb") as pdf_file:
        uploaded_file = client.files.create(
            file=pdf_file,
            purpose="user_data",
        )

    response = client.responses.create(
        prompt={
            "id": PROMPT_ID_FILE_SUMMARY,
            "variables": {
                "topic": "Branch and Merge",
                "reference_pdf": {
                    "type": "input_file",
                    "file_id": uploaded_file.id,
                },
            },
        }
    )

    print(f"- prompt_id: {PROMPT_ID_FILE_SUMMARY}")
    print(f"- uploaded_file_id: {uploaded_file.id}")
    print(response.output_text)


def main() -> None:
    """Run all reusable prompt scenarios."""

    require_api_key()
    client = OpenAI()

    run_text_variable_prompt(client)
    run_file_variable_prompt(client)


if __name__ == "__main__":
    main()

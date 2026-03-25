import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")


def run_basic_question(client):
    response = client.responses.create(
        model="gpt-5-mini",
        input="In one sentence, explain what an LLM is.",
    )
    print("\n=== Basic question ===")
    print(response.output_text)


def run_instructions_example(client):
    response = client.responses.create(
        model="gpt-5-mini",
        instructions="Speak like a friendly pirate.",
        input="Are semicolons optional in JavaScript?",
    )
    print("\n=== Instructions example ===")
    print(response.output_text)


def run_rewrite_example(client):
    response = client.responses.create(
        model="gpt-5-mini",
        input=(
            "Rewrite this sentence in a more professional tone: "
            "'Hi, I would like to schedule an appointment tomorrow morning.'"
        ),
    )
    print("\n=== Rewrite example ===")
    print(response.output_text)


def run_json_style_example(client):
    response = client.responses.create(
        model="gpt-5-mini",
        input=(
            "Give me 3 beginner-friendly Python mini-project ideas in JSON "
            "with the following shape: "
            '{"projects":[{"title":"...","difficulty":"...","goal":"..."}]}'
        ),
    )
    print("\n=== JSON-style response ===")
    print(response.output_text)


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI()

    run_basic_question(client)
    run_instructions_example(client)
    run_rewrite_example(client)
    run_json_style_example(client)


if __name__ == "__main__":
    main()

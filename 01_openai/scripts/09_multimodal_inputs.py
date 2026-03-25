"""Demonstrate several multimodal input scenarios with the Responses API.

This lab focuses on inputs that are not just plain text. It covers:
1. Text plus an image URL
2. Text plus a local image encoded as a data URL
3. Text plus a local PDF encoded in base64

The image scenarios are optional because this repo does not ship with a sample
image by default. You can provide one of the following:
- an image URL through the `OPENAI_SAMPLE_IMAGE_URL` environment variable
- a local image path through the `OPENAI_SAMPLE_IMAGE_PATH` environment variable

The PDF scenario uses the local file already present in this repo.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

MODEL_NAME = "gpt-5-mini"
PDF_MODEL_NAME = "gpt-4o-mini"
PDF_PATH = Path("files/git-cheat-sheet-education.pdf")


def require_api_key() -> None:
    """Fail early with a clear error if the API key is missing."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")


def print_section(title: str) -> None:
    """Print a readable separator for each scenario."""

    print(f"\n{'=' * 18} {title} {'=' * 18}")


def read_file_as_base64(path: Path) -> str:
    """Read one file and return its base64-encoded contents."""

    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_data_url_for_image(path: Path) -> str:
    """Convert a local image into a data URL accepted by the API."""

    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image type for file: {path}")

    encoded_image = read_file_as_base64(path)
    return f"data:{mime_type};base64,{encoded_image}"


def build_data_url_for_pdf(path: Path) -> str:
    """Convert a local PDF into a data URL accepted by the API."""

    encoded_pdf = read_file_as_base64(path)
    return f"data:application/pdf;base64,{encoded_pdf}"


def get_optional_image_url() -> str | None:
    """Return an image URL from the environment when available."""

    image_url = os.getenv("OPENAI_SAMPLE_IMAGE_URL")
    if image_url and image_url.strip():
        return image_url.strip()
    return None


def get_optional_local_image_path() -> Path | None:
    """Return a local image path from the environment when available."""

    image_path = os.getenv("OPENAI_SAMPLE_IMAGE_PATH")
    if not image_path:
        return None

    path = Path(image_path).expanduser()
    if path.exists():
        return path
    return None


def run_image_url_scenario(client: OpenAI) -> None:
    """Analyze an image provided through a remote URL."""

    print_section("Scenario 1 - Image URL input")

    image_url = get_optional_image_url()
    if not image_url:
        print("Skipped: set OPENAI_SAMPLE_IMAGE_URL to test the image URL scenario.")
        return

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Describe this image in 4 concise bullet points and "
                            "identify the main subject."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": "high",
                    },
                ],
            }
        ],
    )

    print(response.output_text)


def run_local_image_scenario(client: OpenAI) -> None:
    """Analyze a local image encoded as a data URL."""

    print_section("Scenario 2 - Local image as data URL")

    image_path = get_optional_local_image_path()
    if image_path is None:
        print(
            "Skipped: set OPENAI_SAMPLE_IMAGE_PATH to a local PNG/JPG/WebP "
            "file to test this scenario."
        )
        return

    data_url = build_data_url_for_image(image_path)

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "What is visible in this image? Answer with a short "
                            "description and 3 notable details."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": "auto",
                    },
                ],
            }
        ],
    )

    print(f"Image file used: {image_path}")
    print(response.output_text)


def run_pdf_scenario(client: OpenAI) -> None:
    """Analyze a local PDF file encoded as a PDF data URL."""

    print_section("Scenario 3 - Local PDF input")

    if not PDF_PATH.exists():
        print(f"Skipped: PDF file not found at {PDF_PATH}")
        return

    pdf_data_url = build_data_url_for_pdf(PDF_PATH)

    response = client.responses.create(
        model=PDF_MODEL_NAME,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Summarize this document for a beginner. Then list 5 "
                            "useful git commands found in it."
                        ),
                    },
                    {
                        "type": "input_file",
                        "filename": PDF_PATH.name,
                        "file_data": pdf_data_url,
                    },
                ],
            }
        ],
    )

    print(f"PDF file used: {PDF_PATH}")
    print(f"PDF model used: {PDF_MODEL_NAME}")
    print(response.output_text)


def main() -> None:
    """Run all multimodal input scenarios."""

    require_api_key()
    client = OpenAI()

    run_image_url_scenario(client)
    run_local_image_scenario(client)
    run_pdf_scenario(client)


if __name__ == "__main__":
    main()

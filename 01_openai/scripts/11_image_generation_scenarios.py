"""Demonstrate several image-generation scenarios with the OpenAI Images API.

This lab focuses on the standalone Images API rather than image generation as
an internal tool call. The goal is to show several useful patterns:
1. Generate one image and save it locally.
2. Generate multiple images with a GPT image model and inspect metadata.
3. Generate a higher-quality image with `gpt-image-1.5` and estimate cost.

The generated files are written to `openai/openai_labs_with_codex/generated_images/`
so you can inspect them directly after the script finishes.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.images_response import ImagesResponse

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

GPT_IMAGE_MODEL = "gpt-image-1-mini"
GPT_IMAGE_HIGH_QUALITY_MODEL = "gpt-image-1.5"
OUTPUT_DIR = Path("generated_images")

GPT_IMAGE_1_5_PRICING = {
    ("1024x1024", "low"): 0.009,
    ("1024x1024", "medium"): 0.034,
    ("1024x1024", "high"): 0.133,
    ("1024x1536", "low"): 0.013,
    ("1024x1536", "medium"): 0.050,
    ("1024x1536", "high"): 0.200,
    ("1536x1024", "low"): 0.013,
    ("1536x1024", "medium"): 0.050,
    ("1536x1024", "high"): 0.200,
}


def require_api_key() -> None:
    """Fail early with a clear error if the API key is missing."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")


def print_section(title: str) -> None:
    """Print a readable separator for each scenario."""

    print(f"\n{'=' * 18} {title} {'=' * 18}")


def ensure_output_dir() -> None:
    """Create the output directory used to store generated images."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_generated_images(
    response: ImagesResponse,
    *,
    filename_prefix: str,
    fallback_extension: str,
) -> list[Path]:
    """Save generated images returned by the API to local files.

    GPT image models usually return base64 image payloads. DALL-E models can
    return URLs or base64, depending on `response_format`. This helper handles
    both cases and prints useful information when saving is not possible.
    """

    saved_paths: list[Path] = []
    output_extension = response.output_format or fallback_extension

    for index, image in enumerate(response.data or [], start=1):
        if image.b64_json:
            image_bytes = base64.b64decode(image.b64_json)
            output_path = OUTPUT_DIR / f"{filename_prefix}_{index}.{output_extension}"
            output_path.write_bytes(image_bytes)
            saved_paths.append(output_path)
            continue

        if image.url:
            print(f"- image {index} returned as URL only: {image.url}")
            continue

        print(f"- image {index} had no `b64_json` or `url` payload.")

    return saved_paths


def print_usage_if_available(response: ImagesResponse) -> None:
    """Print token usage when the selected model exposes it."""

    if not response.usage:
        print("- usage details were not returned for this model/response")
        return

    print("- usage details:")
    print(f"  input_tokens={response.usage.input_tokens}")
    print(f"  output_tokens={response.usage.output_tokens}")
    print(f"  total_tokens={response.usage.total_tokens}")


def estimate_gpt_image_1_5_cost(size: str, quality: str, n: int) -> float | None:
    """Estimate image cost from the published per-image pricing table."""

    unit_price = GPT_IMAGE_1_5_PRICING.get((size, quality))
    if unit_price is None:
        return None
    return unit_price * n


def run_basic_generation(client: OpenAI) -> None:
    """Generate one simple image and save it locally."""

    print_section("Scenario 1 - Basic local image generation")

    response = client.images.generate(
        model=GPT_IMAGE_MODEL,
        prompt=(
            "A clean educational illustration of a Python notebook on a desk, "
            "flat design, light colors, minimal background."
        ),
        size="1024x1024",
        quality="low",
        output_format="png",
    )

    saved_paths = save_generated_images(
        response,
        filename_prefix="01_basic_generation",
        fallback_extension="png",
    )

    print(f"- model: {GPT_IMAGE_MODEL}")
    print(f"- images_returned: {len(response.data or [])}")
    for path in saved_paths:
        print(f"- saved: {path}")
    print_usage_if_available(response)


def run_multiple_gpt_images(client: OpenAI) -> None:
    """Generate more than one image with a GPT image model."""

    print_section("Scenario 2 - Multiple GPT image outputs")

    response = client.images.generate(
        model=GPT_IMAGE_MODEL,
        prompt=(
            "A mascot for a programming course: a friendly robot teacher, "
            "simple shapes, modern educational poster style."
        ),
        n=2,
        size="1024x1024",
        quality="low",
        output_format="webp",
        output_compression=80,
    )

    saved_paths = save_generated_images(
        response,
        filename_prefix="02_multiple_outputs",
        fallback_extension="webp",
    )

    print(f"- model: {GPT_IMAGE_MODEL}")
    print(f"- images_returned: {len(response.data or [])}")
    print(f"- output_format: {response.output_format}")
    for path in saved_paths:
        print(f"- saved: {path}")
    print_usage_if_available(response)


def run_high_quality_gpt_image(client: OpenAI) -> None:
    """Generate one higher-quality image with `gpt-image-1.5`."""

    print_section("Scenario 3 - gpt-image-1.5 high-quality generation")

    size = "1024x1024"
    quality = "medium"
    image_count = 1

    response = client.images.generate(
        model=GPT_IMAGE_HIGH_QUALITY_MODEL,
        prompt=(
            "A polished hero illustration for an AI course landing page, with "
            "a laptop, notebooks, elegant lighting, and a modern editorial style."
        ),
        size=size,
        quality=quality,
        output_format="png",
        n=image_count,
    )

    saved_paths = save_generated_images(
        response,
        filename_prefix="03_gpt_image_1_5",
        fallback_extension="png",
    )

    estimated_cost = estimate_gpt_image_1_5_cost(size, quality, image_count)

    print(f"- model: {GPT_IMAGE_HIGH_QUALITY_MODEL}")
    print(f"- images_returned: {len(response.data or [])}")
    print(f"- size: {size}")
    print(f"- quality: {quality}")
    if estimated_cost is not None:
        print(f"- estimated_cost_usd: ${estimated_cost:.3f}")
    for path in saved_paths:
        print(f"- saved: {path}")
    print_usage_if_available(response)


def main() -> None:
    """Run all image-generation scenarios."""

    require_api_key()
    ensure_output_dir()
    client = OpenAI()

    run_basic_generation(client)
    run_multiple_gpt_images(client)
    run_high_quality_gpt_image(client)


if __name__ == "__main__":
    main()

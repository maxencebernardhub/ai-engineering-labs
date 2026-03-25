"""Demonstrate several video-generation scenarios with the OpenAI Videos API.

This lab is intentionally budget-constrained. It uses only `sora-2` with the
cheapest supported setup:
- 4 seconds
- 720p portrait or landscape

Official pricing at the time this lab was written:
- `sora-2` portrait `720x1280`: $0.10 per second
- `sora-2` landscape `1280x720`: $0.10 per second

This means one 4-second clip costs about $0.40. The full script below creates
two clips, so the planned video-generation spend is about $0.80, which stays
under the requested $1.00 maximum.

The scenarios are:
1. Generate a short landscape clip and download the MP4.
2. Generate a short portrait clip and download the MP4.
3. Retrieve metadata and preview assets for one generated video.

Generated files are written to `video_outputs/` next to this script.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.video import Video

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

VIDEO_MODEL = "sora-2"
SECONDS = "4"
LANDSCAPE_SIZE = "1280x720"
PORTRAIT_SIZE = "720x1280"
PRICE_PER_SECOND_USD = 0.10
MAX_BUDGET_USD = 1.00
OUTPUT_DIR = Path("video_outputs")


def require_api_key() -> None:
    """Fail early with a clear error if the API key is missing."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")


def print_section(title: str) -> None:
    """Print a readable separator for each scenario."""

    print(f"\n{'=' * 18} {title} {'=' * 18}")


def ensure_output_dir() -> None:
    """Create the output directory used to store downloaded video assets."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def estimate_video_cost(seconds: str, clip_count: int) -> float:
    """Estimate video-generation cost from the official per-second pricing."""

    return int(seconds) * PRICE_PER_SECOND_USD * clip_count


def assert_budget(maximum_cost_usd: float) -> None:
    """Abort if the planned scenarios would exceed the allowed budget."""

    if maximum_cost_usd > MAX_BUDGET_USD:
        raise RuntimeError(
            "Planned video cost exceeds the configured budget. "
            f"Planned: ${maximum_cost_usd:.2f}, limit: ${MAX_BUDGET_USD:.2f}."
        )


def download_video_asset(
    client: OpenAI,
    *,
    video_id: str,
    variant: str,
    output_path: Path,
) -> None:
    """Download one generated video asset to a local file."""

    asset = client.videos.download_content(video_id, variant=variant)
    asset.write_to_file(output_path)


def create_and_download_video(
    client: OpenAI,
    *,
    title: str,
    prompt: str,
    size: str,
    filename_prefix: str,
) -> Video:
    """Create one short video, poll until completion, then download it."""

    print_section(title)

    estimated_cost = estimate_video_cost(SECONDS, 1)
    print(f"- estimated_cost_usd: ${estimated_cost:.2f}")
    print(f"- model: {VIDEO_MODEL}")
    print(f"- seconds: {SECONDS}")
    print(f"- size: {size}")

    video = client.videos.create_and_poll(
        model=VIDEO_MODEL,
        prompt=prompt,
        seconds=SECONDS,
        size=size,
        poll_interval_ms=2000,
    )

    print(f"- video_id: {video.id}")
    print(f"- status: {video.status}")
    print(f"- progress: {video.progress}")

    if video.status != "completed":
        if video.error:
            print(f"- error_code: {video.error.code}")
            print(f"- error_message: {video.error.message}")
        return video

    video_path = OUTPUT_DIR / f"{filename_prefix}.mp4"
    thumbnail_path = OUTPUT_DIR / f"{filename_prefix}.jpg"

    download_video_asset(
        client,
        video_id=video.id,
        variant="video",
        output_path=video_path,
    )
    download_video_asset(
        client,
        video_id=video.id,
        variant="thumbnail",
        output_path=thumbnail_path,
    )

    print(f"- saved_video: {video_path}")
    print(f"- saved_thumbnail: {thumbnail_path}")
    return video


def run_landscape_generation(client: OpenAI) -> Video:
    """Generate a short landscape demo clip."""

    return create_and_download_video(
        client,
        title="Scenario 1 - Landscape video generation",
        prompt=(
            "A calm cinematic shot of a laptop on a wooden desk, morning light, "
            "subtle camera movement, modern educational atmosphere."
        ),
        size=LANDSCAPE_SIZE,
        filename_prefix="01_landscape_clip",
    )


def run_portrait_generation(client: OpenAI) -> Video:
    """Generate a short portrait demo clip."""

    return create_and_download_video(
        client,
        title="Scenario 2 - Portrait video generation",
        prompt=(
            "A vertical social-media style shot of a student reviewing Python "
            "notes with animated code floating softly in the background."
        ),
        size=PORTRAIT_SIZE,
        filename_prefix="02_portrait_clip",
    )


def run_metadata_and_preview_scenario(client: OpenAI, video: Video) -> None:
    """Retrieve metadata again and download a spritesheet preview asset."""

    print_section("Scenario 3 - Metadata and preview assets")

    refreshed_video = client.videos.retrieve(video.id)
    print(f"- video_id: {refreshed_video.id}")
    print(f"- status: {refreshed_video.status}")
    print(f"- model: {refreshed_video.model}")
    print(f"- size: {refreshed_video.size}")
    print(f"- seconds: {refreshed_video.seconds}")
    print(f"- created_at: {refreshed_video.created_at}")
    print(f"- expires_at: {refreshed_video.expires_at}")

    if refreshed_video.status != "completed":
        print("- spritesheet download skipped because the video is not completed.")
        return

    spritesheet_path = OUTPUT_DIR / "03_preview_spritesheet.jpg"
    download_video_asset(
        client,
        video_id=refreshed_video.id,
        variant="spritesheet",
        output_path=spritesheet_path,
    )
    print(f"- saved_spritesheet: {spritesheet_path}")


def main() -> None:
    """Run all video scenarios while respecting the cost budget."""

    require_api_key()
    ensure_output_dir()

    planned_cost = estimate_video_cost(SECONDS, 2)
    assert_budget(planned_cost)

    print("Planned budget summary:")
    print(f"- model: {VIDEO_MODEL}")
    print("- planned_video_generations: 2")
    print(f"- estimated_total_cost_usd: ${planned_cost:.2f}")
    print(f"- budget_limit_usd: ${MAX_BUDGET_USD:.2f}")

    client = OpenAI()

    landscape_video = run_landscape_generation(client)
    portrait_video = run_portrait_generation(client)

    preview_source = landscape_video
    if preview_source.status != "completed" and portrait_video.status == "completed":
        preview_source = portrait_video

    run_metadata_and_preview_scenario(client, preview_source)


if __name__ == "__main__":
    main()

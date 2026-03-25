"""Demonstrate several audio scenarios with the OpenAI Audio API.

This lab covers both directions of audio workflows:
1. Text-to-speech (generate audio from text)
2. Speech-to-text (transcribe audio into text)
3. Audio translation into English

Because this repo does not ship with audio samples, the transcription and
translation scenarios are optional. You can provide local files through:
- `OPENAI_SAMPLE_AUDIO_PATH`
- `OPENAI_SAMPLE_FOREIGN_AUDIO_PATH`

Generated audio files are written to `audio_outputs/` next to this script.
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

TTS_MODEL = "gpt-4o-mini-tts"
TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
TRANSLATE_MODEL = "whisper-1"
OUTPUT_DIR = Path("audio_outputs")


def require_api_key() -> None:
    """Fail early with a clear error if the API key is missing."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")


def print_section(title: str) -> None:
    """Print a readable separator for each scenario."""

    print(f"\n{'=' * 18} {title} {'=' * 18}")


def ensure_output_dir() -> None:
    """Create the directory used for generated audio outputs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_optional_audio_path(env_var_name: str) -> Path | None:
    """Return a local audio file path from one environment variable."""

    path_value = os.getenv(env_var_name)
    if not path_value:
        return None

    path = Path(path_value).expanduser()
    if path.exists():
        return path
    return None


def run_basic_text_to_speech(client: OpenAI) -> None:
    """Generate one MP3 file from text."""

    print_section("Scenario 1 - Basic text-to-speech")

    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice="coral",
        response_format="mp3",
        input=(
            "Welcome to this OpenAI lab. We are going to generate a short "
            "audio clip from text."
        ),
    )

    output_path = OUTPUT_DIR / "01_basic_tts.mp3"
    response.write_to_file(output_path)

    print(f"- model: {TTS_MODEL}")
    print("- voice: coral")
    print(f"- saved: {output_path}")


def run_voice_and_speed_variation(client: OpenAI) -> None:
    """Generate two speech variants to compare voice and speed settings."""

    print_section("Scenario 2 - Voice and speed variations")

    variants = [
        {
            "filename": "02_variant_marin.wav",
            "voice": "marin",
            "speed": 1.0,
            "instructions": "Speak in a calm and professional tone.",
        },
        {
            "filename": "03_variant_verse_fast.wav",
            "voice": "verse",
            "speed": 1.25,
            "instructions": "Speak with more energy, like a short tutorial intro.",
        },
    ]

    text = (
        "This is a quick audio demo showing how the same text can sound "
        "different depending on voice and delivery settings."
    )

    for variant in variants:
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=variant["voice"],
            response_format="wav",
            speed=variant["speed"],
            instructions=variant["instructions"],
            input=text,
        )

        output_path = OUTPUT_DIR / variant["filename"]
        response.write_to_file(output_path)

        print(
            f"- saved: {output_path} | voice={variant['voice']} "
            f"| speed={variant['speed']}"
        )


def run_transcription_scenario(client: OpenAI) -> None:
    """Transcribe a local audio file into text."""

    print_section("Scenario 3 - Audio transcription")

    audio_path = get_optional_audio_path("OPENAI_SAMPLE_AUDIO_PATH")
    if audio_path is None:
        print(
            "Skipped: set OPENAI_SAMPLE_AUDIO_PATH to a local audio file "
            "(mp3, wav, m4a, ogg, webm, ...)."
        )
        return

    with audio_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model=TRANSCRIBE_MODEL,
            language="fr",
            response_format="json",
            prompt="This is educational audio about software and AI APIs.",
        )

    print(f"- audio file used: {audio_path}")
    print(f"- model: {TRANSCRIBE_MODEL}")
    print("- transcription:")
    print(transcription.text)


def run_translation_scenario(client: OpenAI) -> None:
    """Translate a local non-English audio file into English."""

    print_section("Scenario 4 - Audio translation to English")

    audio_path = get_optional_audio_path("OPENAI_SAMPLE_FOREIGN_AUDIO_PATH")
    if audio_path is None:
        print(
            "Skipped: set OPENAI_SAMPLE_FOREIGN_AUDIO_PATH to a non-English "
            "audio file for this scenario."
        )
        return

    with audio_path.open("rb") as audio_file:
        translation = client.audio.translations.create(
            file=audio_file,
            model=TRANSLATE_MODEL,
            response_format="text",
            prompt="Translate naturally into clear English.",
        )

    print(f"- audio file used: {audio_path}")
    print(f"- model: {TRANSLATE_MODEL}")
    print("- translation:")
    print(translation)


def main() -> None:
    """Run all audio scenarios."""

    require_api_key()
    ensure_output_dir()
    client = OpenAI()

    run_basic_text_to_speech(client)
    run_voice_and_speed_variation(client)
    run_transcription_scenario(client)
    run_translation_scenario(client)


if __name__ == "__main__":
    main()

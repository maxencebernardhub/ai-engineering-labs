"""Integration tests — require a running Ollama instance.

Run with:   uv run pytest -m integration
Skip with:  uv run pytest -m "not integration"
"""

import pytest

from utils.helpers import check_model_available, check_ollama_running

pytestmark = pytest.mark.integration


def test_ollama_is_running():
    assert check_ollama_running() is True, (
        "Ollama is not running. Start it with: ollama serve"
    )


def test_known_model_is_available():
    assert check_model_available("mistral:7b") is True, (
        "mistral:7b not found. Pull it with: ollama pull mistral:7b"
    )


def test_nonexistent_model_is_not_available():
    assert check_model_available("nonexistent-model:999b") is False

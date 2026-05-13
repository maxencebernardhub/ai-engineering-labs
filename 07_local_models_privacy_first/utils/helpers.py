"""Shared utilities: Ollama health checks and text chunking."""

from __future__ import annotations

import urllib.error
import urllib.request


def check_ollama_running(host: str = "http://localhost:11434") -> bool:
    """Return True if Ollama is reachable, False otherwise."""
    try:
        urllib.request.urlopen(host, timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


def check_model_available(model_name: str) -> bool:
    """Return True if *model_name* is present in the local Ollama library."""
    try:
        import ollama

        response = ollama.list()
        return any(m.model == model_name for m in response.models)
    except Exception:
        return False


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """Split *text* into overlapping word-based chunks.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        List of text chunks. Empty list if *text* is blank.

    Raises:
        ValueError: If *overlap* >= *chunk_size*.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap

    return chunks

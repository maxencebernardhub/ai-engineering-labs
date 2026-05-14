"""Streaming chat logic and response statistics for the Streamlit app."""

from __future__ import annotations

import time
from collections.abc import Generator


def stream_response(
    model: str,
    messages: list[dict],
) -> Generator[str]:
    """Stream tokens from a local Ollama model.

    Args:
        model: Ollama model name (e.g. "gemma4:e4b").
        messages: List of {"role": ..., "content": ...} dicts.

    Yields:
        Token strings as they arrive.
    """
    import ollama

    for chunk in ollama.chat(model=model, messages=messages, stream=True):
        content = chunk.message.content
        if content:
            yield content


def get_stats(start_time: float, token_count: int) -> dict:
    """Compute latency and throughput since *start_time*.

    Args:
        start_time: Unix timestamp recorded before the first token was requested.
        token_count: Number of tokens received in the response.

    Returns:
        Dict with keys:
            - ``latency_ms``    (float): total elapsed time in milliseconds.
            - ``tokens_per_sec`` (float): throughput; 0.0 when token_count is 0.
    """
    elapsed = time.time() - start_time
    return {
        "latency_ms": round(elapsed * 1000, 1),
        "tokens_per_sec": round(token_count / elapsed, 1) if token_count > 0 else 0.0,
    }

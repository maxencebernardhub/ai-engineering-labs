import time

from app.chat import get_stats


def test_get_stats_returns_expected_keys():
    start = time.time() - 1.0
    stats = get_stats(start, token_count=10)
    assert "latency_ms" in stats
    assert "tokens_per_sec" in stats


def test_get_stats_latency_is_positive():
    start = time.time() - 0.5
    stats = get_stats(start, token_count=5)
    assert stats["latency_ms"] > 0


def test_get_stats_throughput_is_positive():
    start = time.time() - 1.0
    stats = get_stats(start, token_count=20)
    assert stats["tokens_per_sec"] > 0


def test_get_stats_zero_tokens_gives_zero_throughput():
    start = time.time() - 1.0
    stats = get_stats(start, token_count=0)
    assert stats["tokens_per_sec"] == 0.0


def test_get_stats_latency_approximate():
    start = time.time() - 2.0
    stats = get_stats(start, token_count=10)
    # Should be close to 2000ms; allow ±300ms for test runner overhead
    assert 1700 < stats["latency_ms"] < 2300


def test_get_stats_throughput_approximate():
    start = time.time() - 2.0
    stats = get_stats(start, token_count=100)
    # ~50 tokens/sec; allow generous range
    assert 40 < stats["tokens_per_sec"] < 70

"""Unit tests for CostTracker — written before implementation (TDD)."""

import json
from pathlib import Path

import pytest

from llm_client.cost_tracker import CostEntry, CostTracker


def make_entry(**overrides) -> CostEntry:
    """Return a CostEntry with sensible defaults, optionally overridden."""
    defaults = {
        "ts": "2026-04-21T10:00:00",
        "provider": "openai",
        "model": "gpt-5.4",
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_usd": 0.0005,
        "latency_ms": 300,
    }
    defaults.update(overrides)
    return CostEntry(**defaults)


class TestLogEntry:
    def test_log_entry_has_all_fields(self):
        """A logged entry must be retrievable with all original field values."""
        tracker = CostTracker()
        entry = make_entry()
        tracker.log(entry)

        stored = tracker.entries()[0]
        assert stored.ts == entry.ts
        assert stored.provider == entry.provider
        assert stored.model == entry.model
        assert stored.input_tokens == entry.input_tokens
        assert stored.output_tokens == entry.output_tokens
        assert stored.cost_usd == entry.cost_usd
        assert stored.latency_ms == entry.latency_ms

    def test_multiple_calls_accumulate_in_memory(self):
        """Multiple log() calls must all be retained in memory."""
        tracker = CostTracker()
        tracker.log(make_entry(provider="openai"))
        tracker.log(make_entry(provider="anthropic"))
        tracker.log(make_entry(provider="gemini"))

        assert len(tracker.entries()) == 3
        providers = [e.provider for e in tracker.entries()]
        assert providers == ["openai", "anthropic", "gemini"]


class TestFlushToFile:
    def test_flush_writes_jsonl_file(self, tmp_path: Path):
        """Each entry must be flushed to a .jsonl file as a valid JSON line."""
        log_file = tmp_path / "costs.jsonl"
        tracker = CostTracker(log_path=log_file)

        entry = make_entry(cost_usd=0.001)
        tracker.log(entry)

        assert log_file.exists()
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["cost_usd"] == pytest.approx(0.001)
        assert record["provider"] == "openai"

    def test_jsonl_created_if_not_exists(self, tmp_path: Path):
        """The .jsonl file must be created automatically on first log()."""
        log_file = tmp_path / "subdir" / "costs.jsonl"
        # subdir does not exist yet
        tracker = CostTracker(log_path=log_file)
        tracker.log(make_entry())

        assert log_file.exists()

    def test_multiple_entries_each_get_own_line(self, tmp_path: Path):
        """N log() calls must produce N lines in the .jsonl file."""
        log_file = tmp_path / "costs.jsonl"
        tracker = CostTracker(log_path=log_file)

        for i in range(3):
            tracker.log(make_entry(cost_usd=float(i)))

        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 3


class TestTotalCost:
    def test_total_cost_sums_correctly(self):
        """total_cost() must return the exact sum of all logged cost_usd values."""
        tracker = CostTracker()
        tracker.log(make_entry(cost_usd=0.001))
        tracker.log(make_entry(cost_usd=0.002))
        tracker.log(make_entry(cost_usd=0.003))

        assert tracker.total_cost() == pytest.approx(0.006)

    def test_total_cost_zero_when_no_entries(self):
        """total_cost() must be 0.0 when nothing has been logged yet."""
        tracker = CostTracker()
        assert tracker.total_cost() == 0.0

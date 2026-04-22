"""Persistent cost tracking for LLM API calls.

Each call is logged in-memory and automatically flushed to a .jsonl file
(one JSON object per line) so that costs survive process restarts.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CostEntry:
    """A single cost record for one LLM API call."""

    ts: str  # ISO-8601 datetime string
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int


class CostTracker:
    """Tracks LLM API call costs in memory and persists them to a .jsonl file.

    Usage::

        tracker = CostTracker(log_path=Path("costs.jsonl"))
        tracker.log(CostEntry(...))
        print(tracker.total_cost())
    """

    def __init__(self, log_path: Path | None = None) -> None:
        """Initialise the tracker.

        Args:
            log_path: Path to the .jsonl file for persistence.  When None,
                entries are only kept in memory (useful for tests).
        """
        self._log_path = log_path
        self._entries: list[CostEntry] = []

    def log(self, entry: CostEntry) -> None:
        """Append *entry* to the in-memory list and flush to disk.

        The parent directory of ``log_path`` is created automatically if it
        does not exist yet.

        Args:
            entry: The cost entry to record.
        """
        self._entries.append(entry)
        if self._log_path is not None:
            # Create parent directories on the first write.
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(entry)) + "\n")

    def total_cost(self) -> float:
        """Return the sum of all logged ``cost_usd`` values.

        Returns:
            Total cost in USD as a float (0.0 if no entries logged yet).
        """
        return sum(e.cost_usd for e in self._entries)

    def entries(self) -> list[CostEntry]:
        """Return a copy of the in-memory entry list.

        Returns:
            List of :class:`CostEntry` objects in insertion order.
        """
        return list(self._entries)

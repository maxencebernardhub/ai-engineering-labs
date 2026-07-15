"""In-memory lead store — used by tests and as an ephemeral default.

Holds the lead list in a plain Python list. `_load` returns deep-ish copies so
callers cannot mutate persisted state by holding onto a returned dict, matching
the isolation the file/S3/Postgres backends give for free.
"""

from __future__ import annotations

import copy

from app.storage.base import ListBackedStore


class InMemoryStore(ListBackedStore):
    def __init__(self) -> None:
        self._leads: list[dict] = []

    def _load(self) -> list[dict]:
        return copy.deepcopy(self._leads)

    def _save(self, leads: list[dict]) -> None:
        self._leads = copy.deepcopy(leads)

"""Lead storage package: the `LeadStore` interface and its backends.

`get_store(settings)` selects a backend from configuration (env `LEAD_STORE`):
`memory` (default / tests), `postgres` (local Docker), or `s3` (cloud).
"""

from __future__ import annotations

from typing import Protocol

from app.storage.base import LeadStore
from app.storage.memory_store import InMemoryStore
from app.storage.postgres_store import PostgresStore
from app.storage.s3_store import S3Store


class _StoreSettings(Protocol):
    """The subset of settings `get_store` reads (duck-typed).

    The concrete `Settings` object (pydantic-settings) arrives in Step 2; this
    Protocol keeps the factory decoupled from it and easy to test.
    """

    lead_store: str
    database_url: str | None
    leads_bucket: str | None


def get_store(settings: _StoreSettings) -> LeadStore:
    backend = settings.lead_store
    if backend == "memory":
        return InMemoryStore()
    if backend == "postgres":
        if not settings.database_url:
            raise ValueError("LEAD_STORE=postgres requires DATABASE_URL")
        return PostgresStore.from_url(settings.database_url)
    if backend == "s3":
        if not settings.leads_bucket:
            raise ValueError("LEAD_STORE=s3 requires LEADS_BUCKET")
        return S3Store(bucket=settings.leads_bucket)
    raise ValueError(f"unknown LEAD_STORE backend: {backend!r}")


__all__ = [
    "InMemoryStore",
    "LeadStore",
    "PostgresStore",
    "S3Store",
    "get_store",
]

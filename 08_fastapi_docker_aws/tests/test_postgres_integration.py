"""Postgres integration smoke test — the real serverful backend.

Unlike `test_storage.py` (which runs the `LeadStore` contract against SQLite for
zero-infra parity), this exercises `PostgresStore` against a *real* PostgreSQL:
the `db` service from `docker-compose.yml`. It is marked `integration` and skips
itself unless `DATABASE_URL` points at a Postgres instance, so the default
`pytest -m "not integration"` run (and CI) never needs a database.

Run it against the Compose DB (port published in docker-compose.yml):

    docker compose up -d db
    DATABASE_URL=postgresql+psycopg://leads:leads@localhost:5432/leads \
        uv run pytest -m integration
"""

from __future__ import annotations

import os

import pytest
from sqlmodel import Session, delete

from app.domain import InvalidTransitionError
from app.storage import PostgresStore
from app.storage.postgres_store import Lead

DATABASE_URL = os.environ.get("DATABASE_URL", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "postgresql" not in DATABASE_URL,
        reason="set DATABASE_URL to a Postgres URL (see docstring) to run this",
    ),
]


@pytest.fixture
def pg_store():
    """A `PostgresStore` on the real DB, with the leads table emptied around it."""
    store = PostgresStore.from_url(DATABASE_URL)
    _truncate(store)
    yield store
    _truncate(store)


def _truncate(store: PostgresStore) -> None:
    with Session(store._engine) as session:
        session.exec(delete(Lead))
        session.commit()


def test_postgres_store_parity(pg_store):
    """The full contract holds against real Postgres, and writes are durable."""
    # Create + read back.
    lead = pg_store.add_lead("Sophie Laurent", "Streamline SAS", "s@streamline.fr")
    assert lead["status"] == "prospect"
    assert pg_store.list_leads()[0]["id"] == lead["id"]

    # Notes + a valid status transition persist.
    pg_store.add_note(lead["id"], "Called back")
    pg_store.update_status(lead["id"], "qualified")

    # Guardrail: an illegal transition is rejected and leaves state untouched.
    with pytest.raises(InvalidTransitionError):
        pg_store.update_status(lead["id"], "prospect")

    # Pipeline stats reflect the writes.
    assert pg_store.pipeline_stats() == {
        "prospect": 0,
        "qualified": 1,
        "won": 0,
        "lost": 0,
    }

    # Durability: a *fresh* store (new engine/connection) sees the same data —
    # proving it was persisted in Postgres, not held in process memory.
    fresh = PostgresStore.from_url(DATABASE_URL)
    reread = fresh.list_leads()
    assert len(reread) == 1
    assert reread[0]["id"] == lead["id"]
    assert reread[0]["status"] == "qualified"
    assert reread[0]["notes"] == ["Called back"]

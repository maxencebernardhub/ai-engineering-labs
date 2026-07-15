"""Storage-layer tests.

One parametrized suite runs the full `LeadStore` contract against all three
backends — memory, Postgres (via SQLite for zero-infra parity), and S3 (via
moto) — so a single set of assertions guards every implementation. Domain-level
and factory tests live alongside.
"""

from __future__ import annotations

from types import SimpleNamespace

import boto3
import pytest
from moto import mock_aws
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

from app.domain import (
    INITIAL_STATUS,
    VALID_TRANSITIONS,
    InvalidTransitionError,
    LeadNotFoundError,
    new_lead,
    validate_transition,
)
from app.storage import InMemoryStore, PostgresStore, S3Store, get_store

# --------------------------------------------------------------------------- #
# Parametrized backend fixture
# --------------------------------------------------------------------------- #


@pytest.fixture(params=["memory", "sqlite", "s3"])
def store(request):
    """Yield an empty `LeadStore` for each backend under test."""
    backend = request.param
    if backend == "memory":
        yield InMemoryStore()
    elif backend == "sqlite":
        # Shared in-memory SQLite: StaticPool keeps one connection alive so the
        # schema and rows persist across sessions within the test.
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        yield PostgresStore(engine)
    elif backend == "s3":
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-leads")
            yield S3Store(bucket="test-leads", key="leads.json", client=client)


# --------------------------------------------------------------------------- #
# Contract tests (run against every backend)
# --------------------------------------------------------------------------- #


def test_add_and_list(store):
    assert store.list_leads() == []
    lead = store.add_lead("Alice Moreau", "Tech Solutions SAS", "a@ts.fr")
    assert lead["status"] == INITIAL_STATUS
    assert lead["notes"] == []
    assert lead["id"].startswith("lead_")

    listed = store.list_leads()
    assert len(listed) == 1
    assert listed[0]["id"] == lead["id"]
    assert listed[0]["company"] == "Tech Solutions SAS"


def test_list_filter_by_status(store):
    a = store.add_lead("A", "ACo", "a@co.fr")
    store.add_lead("B", "BCo", "b@co.fr")
    store.update_status(a["id"], "qualified")

    qualified = store.list_leads(status_filter="qualified")
    assert [lead["id"] for lead in qualified] == [a["id"]]
    assert len(store.list_leads(status_filter="prospect")) == 1
    assert store.list_leads(status_filter="won") == []


def test_add_note_appends(store):
    lead = store.add_lead("A", "ACo", "a@co.fr")
    updated = store.add_note(lead["id"], "Called back")
    assert updated["notes"] == ["Called back"]
    again = store.add_note(lead["id"], "Sent proposal")
    assert again["notes"] == ["Called back", "Sent proposal"]
    # Persisted, not just returned.
    assert store.list_leads()[0]["notes"] == ["Called back", "Sent proposal"]


def test_add_note_lead_not_found(store):
    with pytest.raises(LeadNotFoundError):
        store.add_note("lead_missing", "note")


def test_update_status_valid(store):
    lead = store.add_lead("A", "ACo", "a@co.fr")
    updated = store.update_status(lead["id"], "qualified")
    assert updated["status"] == "qualified"
    won = store.update_status(lead["id"], "won")
    assert won["status"] == "won"
    assert store.list_leads()[0]["status"] == "won"


def test_update_status_invalid_transition(store):
    lead = store.add_lead("A", "ACo", "a@co.fr")  # prospect
    with pytest.raises(InvalidTransitionError):
        store.update_status(lead["id"], "won")  # must go via qualified
    # State unchanged after a rejected transition.
    assert store.list_leads()[0]["status"] == "prospect"


def test_update_status_lead_not_found(store):
    with pytest.raises(LeadNotFoundError):
        store.update_status("lead_missing", "qualified")


def test_pipeline_stats_counts(store):
    a = store.add_lead("A", "ACo", "a@co.fr")
    store.add_lead("B", "BCo", "b@co.fr")
    c = store.add_lead("C", "CCo", "c@co.fr")
    store.update_status(a["id"], "qualified")
    store.update_status(c["id"], "lost")

    assert store.pipeline_stats() == {
        "prospect": 1,
        "qualified": 1,
        "won": 0,
        "lost": 1,
    }


def test_seed_if_empty_populates(store):
    seed = [
        {
            "id": "lead_001",
            "name": "Seed One",
            "company": "Seed Co",
            "email": "s1@seed.co",
            "status": "won",
            "notes": ["Imported"],
            "created_at": "2026-01-01",
            "updated_at": "2026-02-01",
        }
    ]
    store.seed_if_empty(seed)
    listed = store.list_leads()
    assert len(listed) == 1
    # Seeded verbatim: id, status, notes and dates are preserved.
    assert listed[0]["id"] == "lead_001"
    assert listed[0]["status"] == "won"
    assert listed[0]["notes"] == ["Imported"]
    assert listed[0]["created_at"] == "2026-01-01"


def test_seed_if_empty_is_noop_when_present(store):
    store.add_lead("Existing", "ECo", "e@co.fr")
    store.seed_if_empty(
        [
            {
                "id": "lead_001",
                "name": "Seed",
                "company": "Seed Co",
                "email": "s@seed.co",
                "status": "prospect",
                "notes": [],
                "created_at": "2026-01-01",
                "updated_at": "2026-01-01",
            }
        ]
    )
    listed = store.list_leads()
    assert len(listed) == 1
    assert listed[0]["name"] == "Existing"


# --------------------------------------------------------------------------- #
# Domain-level tests
# --------------------------------------------------------------------------- #


def test_new_lead_defaults():
    from datetime import date

    lead = new_lead("A", "ACo", "a@co.fr", today=date(2026, 7, 8))
    assert lead["status"] == INITIAL_STATUS
    assert lead["notes"] == []
    assert lead["created_at"] == "2026-07-08"
    assert lead["updated_at"] == "2026-07-08"
    assert lead["id"].startswith("lead_")


def test_new_lead_ids_are_unique():
    ids = {new_lead("A", "C", "a@c.fr")["id"] for _ in range(100)}
    assert len(ids) == 100


@pytest.mark.parametrize(
    ("current", "new_status"),
    [(cur, nxt) for cur, allowed in VALID_TRANSITIONS.items() for nxt in allowed],
)
def test_validate_transition_allows_valid(current, new_status):
    validate_transition(current, new_status)  # must not raise


@pytest.mark.parametrize(
    ("current", "new_status"),
    [("prospect", "won"), ("won", "qualified"), ("lost", "prospect")],
)
def test_validate_transition_rejects_invalid(current, new_status):
    with pytest.raises(InvalidTransitionError):
        validate_transition(current, new_status)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def test_get_store_selected_by_env():
    settings = SimpleNamespace(
        lead_store="memory", database_url=None, leads_bucket=None
    )
    assert isinstance(get_store(settings), InMemoryStore)


def test_get_store_unknown_backend_raises():
    settings = SimpleNamespace(
        lead_store="mystery", database_url=None, leads_bucket=None
    )
    with pytest.raises(ValueError, match="unknown LEAD_STORE"):
        get_store(settings)


def test_get_store_s3_via_moto():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(
            Bucket="factory-bucket"
        )
        settings = SimpleNamespace(
            lead_store="s3", database_url=None, leads_bucket="factory-bucket"
        )
        assert isinstance(get_store(settings), S3Store)

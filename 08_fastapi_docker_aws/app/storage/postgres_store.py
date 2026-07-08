"""Postgres-backed lead store — the local Docker backend.

Row-based (SQLModel), unlike the list-backed memory/S3 stores. `notes` is a
JSON column, which works identically on SQLite and Postgres, so the same code
is exercised by the SQLite-backed unit tests and the Postgres integration test.
No Postgres-only features are used, preserving that parity.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import Column
from sqlmodel import JSON, Field, Session, SQLModel, create_engine, select

from app.domain import STATUSES, LeadNotFoundError, new_lead, validate_transition
from app.storage.base import LeadStore


class Lead(SQLModel, table=True):
    """Persisted lead row. Mirrors the domain dict shape 1:1."""

    id: str = Field(primary_key=True)
    name: str
    company: str
    email: str
    status: str
    notes: list = Field(default_factory=list, sa_column=Column(JSON))
    created_at: str
    updated_at: str


def _to_dict(row: Lead) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "company": row.company,
        "email": row.email,
        "status": row.status,
        "notes": list(row.notes),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


class PostgresStore(LeadStore):
    """`LeadStore` implementation over a SQL database via SQLModel.

    Accepts a pre-built engine (tests pass a SQLite engine for parity); the
    factory builds a Postgres engine from the configured `DATABASE_URL`.
    """

    def __init__(self, engine) -> None:
        self._engine = engine
        SQLModel.metadata.create_all(engine)

    @classmethod
    def from_url(cls, database_url: str) -> PostgresStore:
        return cls(create_engine(database_url))

    def list_leads(self, status_filter: str | None = None) -> list[dict]:
        with Session(self._engine) as session:
            statement = select(Lead).order_by(Lead.id)
            if status_filter:
                statement = statement.where(Lead.status == status_filter)
            return [_to_dict(row) for row in session.exec(statement).all()]

    def add_lead(self, name: str, company: str, email: str) -> dict:
        lead = new_lead(name, company, email)
        with Session(self._engine) as session:
            session.add(Lead(**lead))
            session.commit()
        return lead

    def add_note(self, lead_id: str, note: str) -> dict:
        with Session(self._engine) as session:
            row = session.get(Lead, lead_id)
            if row is None:
                raise LeadNotFoundError(lead_id)
            # Reassign (not append) so SQLAlchemy detects the JSON change.
            row.notes = [*row.notes, note]
            row.updated_at = date.today().isoformat()
            session.add(row)
            session.commit()
            session.refresh(row)
            return _to_dict(row)

    def update_status(self, lead_id: str, new_status: str) -> dict:
        with Session(self._engine) as session:
            row = session.get(Lead, lead_id)
            if row is None:
                raise LeadNotFoundError(lead_id)
            validate_transition(row.status, new_status)
            row.status = new_status
            row.updated_at = date.today().isoformat()
            session.add(row)
            session.commit()
            session.refresh(row)
            return _to_dict(row)

    def pipeline_stats(self) -> dict[str, int]:
        stats = {status: 0 for status in STATUSES}
        with Session(self._engine) as session:
            for row in session.exec(select(Lead)).all():
                stats[row.status] = stats.get(row.status, 0) + 1
        return stats

    def seed_if_empty(self, leads: list[dict]) -> None:
        with Session(self._engine) as session:
            if session.exec(select(Lead)).first() is not None:
                return
            for lead in leads:
                session.add(Lead(**lead))
            session.commit()

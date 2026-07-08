"""Storage abstraction for leads.

`LeadStore` is the interface consumed by the agent tools and the API. Three
backends implement it: in-memory (tests / ephemeral), Postgres (local Docker),
and S3 (cloud). Two of them — memory and S3 — differ only in how they load and
save the full lead list, so they share `ListBackedStore`; Postgres is row-based
and implements the interface directly.

All backends operate on the plain-dict lead shape defined in `app.domain`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from app.domain import (
    STATUSES,
    LeadNotFoundError,
    new_lead,
    validate_transition,
)


class LeadStore(ABC):
    """Backend-agnostic persistence interface for leads."""

    @abstractmethod
    def list_leads(self, status_filter: str | None = None) -> list[dict]:
        """Return all leads, optionally filtered by status."""

    @abstractmethod
    def add_lead(self, name: str, company: str, email: str) -> dict:
        """Create and persist a new `prospect` lead; return it."""

    @abstractmethod
    def add_note(self, lead_id: str, note: str) -> dict:
        """Append a note to a lead; return the updated lead.

        Raises `LeadNotFoundError` if the lead does not exist.
        """

    @abstractmethod
    def update_status(self, lead_id: str, new_status: str) -> dict:
        """Transition a lead's status; return the updated lead.

        Raises `LeadNotFoundError` if the lead does not exist, or
        `InvalidTransitionError` if the transition is not allowed.
        """

    @abstractmethod
    def pipeline_stats(self) -> dict[str, int]:
        """Return a count of leads per status (all statuses zero-filled)."""

    @abstractmethod
    def seed_if_empty(self, leads: list[dict]) -> None:
        """Insert `leads` verbatim (ids, statuses, notes, dates) iff the store
        is currently empty. Idempotent boot-time helper for the demo data.
        """


class ListBackedStore(LeadStore):
    """`LeadStore` implemented over a single in-memory list of dicts.

    Subclasses provide only `_load()` / `_save()`; everything else is derived
    from those two primitives (the lab-06 file-store pattern, generalized).
    """

    @abstractmethod
    def _load(self) -> list[dict]:
        """Return the full lead list from the backing medium."""

    @abstractmethod
    def _save(self, leads: list[dict]) -> None:
        """Persist the full lead list to the backing medium."""

    def list_leads(self, status_filter: str | None = None) -> list[dict]:
        leads = self._load()
        if status_filter:
            leads = [lead for lead in leads if lead["status"] == status_filter]
        return leads

    def add_lead(self, name: str, company: str, email: str) -> dict:
        leads = self._load()
        lead = new_lead(name, company, email)
        leads.append(lead)
        self._save(leads)
        return lead

    def add_note(self, lead_id: str, note: str) -> dict:
        leads = self._load()
        for lead in leads:
            if lead["id"] == lead_id:
                lead["notes"].append(note)
                lead["updated_at"] = date.today().isoformat()
                self._save(leads)
                return lead
        raise LeadNotFoundError(lead_id)

    def update_status(self, lead_id: str, new_status: str) -> dict:
        leads = self._load()
        for lead in leads:
            if lead["id"] == lead_id:
                validate_transition(lead["status"], new_status)
                lead["status"] = new_status
                lead["updated_at"] = date.today().isoformat()
                self._save(leads)
                return lead
        raise LeadNotFoundError(lead_id)

    def pipeline_stats(self) -> dict[str, int]:
        stats = {status: 0 for status in STATUSES}
        for lead in self._load():
            stats[lead["status"]] = stats.get(lead["status"], 0) + 1
        return stats

    def seed_if_empty(self, leads: list[dict]) -> None:
        if not self._load():
            # Copy so the caller's seed list is never mutated by later writes.
            self._save([dict(lead) for lead in leads])

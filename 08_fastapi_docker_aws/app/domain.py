"""Lead domain model and status-transition rules.

Ported from lab 06 (`shared/leads_store.py`) but decoupled from any storage
concern: this module knows the shape of a lead and the rules governing status
changes, nothing about *where* leads live. Storage backends (memory, Postgres,
S3) build on these primitives.
"""

from __future__ import annotations

import uuid
from datetime import date

# Pipeline stages, in lifecycle order. `prospect` is the entry state.
STATUSES: tuple[str, ...] = ("prospect", "qualified", "won", "lost")
INITIAL_STATUS = "prospect"

# Allowed forward transitions. `won` and `lost` are terminal.
VALID_TRANSITIONS: dict[str, list[str]] = {
    "prospect": ["qualified", "lost"],
    "qualified": ["won", "lost"],
    "won": [],
    "lost": [],
}


class LeadNotFoundError(ValueError):
    """Raised when an operation targets a lead id that does not exist."""

    def __init__(self, lead_id: str) -> None:
        super().__init__(f"Lead '{lead_id}' not found")
        self.lead_id = lead_id


class InvalidTransitionError(ValueError):
    """Raised when a requested status change is not allowed."""

    def __init__(self, current: str, new_status: str) -> None:
        allowed = VALID_TRANSITIONS.get(current, [])
        super().__init__(
            f"invalid transition: '{current}' -> '{new_status}'. Allowed: {allowed}"
        )
        self.current = current
        self.new_status = new_status


def new_lead(
    name: str,
    company: str,
    email: str,
    *,
    today: date | None = None,
) -> dict:
    """Build a fresh lead in the `prospect` state.

    The id is random (`lead_<6 hex>`), which is safe in a stateless, possibly
    concurrent service (no scan of existing ids, no counter to race on).
    `today` is injectable for deterministic tests.
    """
    stamp = (today or date.today()).isoformat()
    return {
        "id": f"lead_{uuid.uuid4().hex[:6]}",
        "name": name,
        "company": company,
        "email": email,
        "status": INITIAL_STATUS,
        "notes": [],
        "created_at": stamp,
        "updated_at": stamp,
    }


def validate_transition(current: str, new_status: str) -> None:
    """Raise `InvalidTransitionError` if `current -> new_status` is not allowed."""
    if new_status not in VALID_TRANSITIONS.get(current, []):
        raise InvalidTransitionError(current, new_status)

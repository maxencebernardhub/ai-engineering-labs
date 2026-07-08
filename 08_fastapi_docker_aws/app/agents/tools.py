"""LangChain tools for the commercial assistant agent, bound to a `LeadStore`.

Adapted from lab 06 (`shared/tools.py`) for a stateless HTTP service:

* Every tool is a closure over a `LeadStore` instance built by `build_tools`,
  instead of reading a module-level JSON path. This lets the API select the
  backend per deployment (in-memory / Postgres / S3) with no change to the agent.
* `generate_email_draft` returns the draft as a structured artifact in its result
  rather than writing a file under `drafts/`; the runner surfaces it in the API
  response and the client decides what to do with it.
* Lab 06's human-in-the-loop gates (draft approval, confirmation before `-> lost`)
  are gone: the service runs autonomously. Domain errors — an unknown lead or an
  illegal status transition — are returned as plain warning strings so the agent
  can relay them to the user, exactly as the interactive lab did.
"""

from __future__ import annotations

from datetime import datetime

from langchain_core.tools import BaseTool, tool

from app.domain import InvalidTransitionError, LeadNotFoundError
from app.storage import LeadStore


def build_tools(store: LeadStore) -> list[BaseTool]:
    """Return the six agent tools, each bound to `store`."""

    @tool
    def list_leads(status_filter: str = "") -> str:
        """List all leads. Optionally filter by status: prospect, qualified,
        won, lost."""
        leads = store.list_leads(status_filter=status_filter or None)
        if not leads:
            return "No leads found."
        return "\n".join(
            f"[{lead['id']}] {lead['name']} ({lead['company']}) — {lead['status']}"
            for lead in leads
        )

    @tool
    def add_lead(name: str, company: str, email: str) -> str:
        """Add a new lead with status 'prospect'."""
        lead = store.add_lead(name=name, company=company, email=email)
        return f"Lead created: [{lead['id']}] {lead['name']} ({lead['company']})"

    @tool
    def add_note(lead_id: str, note: str) -> str:
        """Append a note to a lead's notes list."""
        try:
            lead = store.add_note(lead_id=lead_id, note=note)
        except LeadNotFoundError as exc:
            return str(exc)
        return f'Note added to [{lead_id}] {lead["name"]}: "{note}"'

    @tool
    def update_lead_status(lead_id: str, new_status: str) -> str:
        """Update a lead's status. Valid transitions: prospect->qualified,
        prospect->lost, qualified->won, qualified->lost."""
        try:
            lead = store.update_status(lead_id=lead_id, new_status=new_status)
        except (LeadNotFoundError, InvalidTransitionError) as exc:
            return str(exc)
        return f"[{lead_id}] {lead['name']} status updated to '{new_status}'"

    @tool(response_format="content_and_artifact")
    def generate_email_draft(lead_id: str, intent: str) -> tuple[str, dict | None]:
        """Generate an email draft for a lead and return it for review.

        The intent describes the purpose of the email (e.g. 'follow-up',
        'outreach'). The draft is returned in the response — it is neither sent
        nor saved; the caller reviews and acts on it.
        """
        lead = next(
            (item for item in store.list_leads() if item["id"] == lead_id), None
        )
        if lead is None:
            return f"Lead '{lead_id}' not found.", None

        draft = {
            "lead_id": lead_id,
            "to": lead["email"],
            "subject": f"[{intent.title()}] {lead['company']}",
            "body": (
                f"Dear {lead['name']},\n\n"
                f"I am reaching out regarding: {intent}.\n\n"
                f"Please let me know if you would like to discuss further.\n\n"
                f"Best regards"
            ),
            "generated_at": datetime.now().isoformat(),
        }
        summary = (
            f"Email draft ready for {lead['name']} ({lead['email']}).\n"
            f"Subject: {draft['subject']}\n"
            f"Review it before sending."
        )
        return summary, draft

    @tool
    def get_pipeline_stats() -> str:
        """Return a summary of the leads pipeline: count per status."""
        stats = store.pipeline_stats()
        total = sum(stats.values())
        lines = [f"  {status}: {count}" for status, count in stats.items()]
        return "Pipeline summary:\n" + "\n".join(lines) + f"\n  Total: {total}"

    return [
        list_leads,
        add_lead,
        add_note,
        update_lead_status,
        generate_email_draft,
        get_pipeline_stats,
    ]

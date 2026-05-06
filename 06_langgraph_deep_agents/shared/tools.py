import json
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from shared.leads_store import (
    add_lead,
    add_note,
    get_pipeline_stats,
    list_leads,
    update_lead_status,
)

LEADS_PATH = Path(__file__).parent.parent / "data" / "leads.json"
DRAFTS_DIR = Path(__file__).parent.parent / "drafts"


@tool
def list_leads_tool(status_filter: str = "") -> str:
    """List all leads. Optionally filter by status: prospect, qualified, won, lost."""
    leads = list_leads(LEADS_PATH, status_filter=status_filter or None)
    if not leads:
        return "No leads found."
    lines = [
        f"[{lead['id']}] {lead['name']} ({lead['company']}) — {lead['status']}"
        for lead in leads
    ]
    return "\n".join(lines)


@tool
def add_lead_tool(name: str, company: str, email: str) -> str:
    """Add a new lead with status 'prospect'."""
    lead = add_lead(LEADS_PATH, name=name, company=company, email=email)
    return f"Lead created: [{lead['id']}] {lead['name']} ({lead['company']})"


@tool
def add_note_tool(lead_id: str, note: str) -> str:
    """Append a note to a lead's notes list."""
    lead = add_note(LEADS_PATH, lead_id=lead_id, note=note)
    return f'Note added to [{lead_id}] {lead["name"]}: "{note}"'


@tool
def update_lead_status_tool(lead_id: str, new_status: str) -> str:
    """Update a lead's status. Valid transitions: prospect→qualified, prospect→lost,
    qualified→won, qualified→lost."""
    lead = update_lead_status(LEADS_PATH, lead_id=lead_id, new_status=new_status)
    return f"[{lead_id}] {lead['name']} status updated to '{new_status}'"


@tool
def generate_email_draft_tool(lead_id: str, intent: str) -> str:
    """Generate an email draft for a lead and save it as a JSON file in drafts/.

    The intent describes the purpose of the email (e.g. 'follow-up', 'outreach').
    """
    leads = list_leads(LEADS_PATH)
    lead = next((item for item in leads if item["id"] == lead_id), None)
    if not lead:
        return f"Lead '{lead_id}' not found."

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

    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"draft_{lead_id}_{timestamp}.json"
    (DRAFTS_DIR / filename).write_text(
        json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return (
        f"Email draft saved as '{filename}' for {lead['name']} ({lead['email']}).\n"
        f"Subject: {draft['subject']}\n"
        f"Please review the draft before sending."
    )


@tool
def get_pipeline_stats_tool() -> str:
    """Return a summary of the leads pipeline: count per status."""
    stats = get_pipeline_stats(LEADS_PATH)
    total = sum(stats.values())
    lines = [f"  {status}: {count}" for status, count in stats.items()]
    return "Pipeline summary:\n" + "\n".join(lines) + f"\n  Total: {total}"

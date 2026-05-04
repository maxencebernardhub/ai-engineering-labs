import json
import uuid
from datetime import date
from pathlib import Path

VALID_TRANSITIONS: dict[str, list[str]] = {
    "prospect": ["qualified", "lost"],
    "qualified": ["won", "lost"],
    "won": [],
    "lost": [],
}


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, leads: list[dict]) -> None:
    path.write_text(json.dumps(leads, indent=2, ensure_ascii=False), encoding="utf-8")


def list_leads(path: Path, status_filter: str | None = None) -> list[dict]:
    leads = _load(path)
    if status_filter:
        leads = [l for l in leads if l["status"] == status_filter]
    return leads


def add_lead(path: Path, name: str, company: str, email: str) -> dict:
    leads = _load(path)
    today = date.today().isoformat()
    lead = {
        "id": f"lead_{uuid.uuid4().hex[:6]}",
        "name": name,
        "company": company,
        "email": email,
        "status": "prospect",
        "notes": [],
        "created_at": today,
        "updated_at": today,
    }
    leads.append(lead)
    _save(path, leads)
    return lead


def add_note(path: Path, lead_id: str, note: str) -> dict:
    leads = _load(path)
    for lead in leads:
        if lead["id"] == lead_id:
            lead["notes"].append(note)
            lead["updated_at"] = date.today().isoformat()
            _save(path, leads)
            return lead
    raise ValueError(f"Lead '{lead_id}' not found")


def update_lead_status(path: Path, lead_id: str, new_status: str) -> dict:
    leads = _load(path)
    for lead in leads:
        if lead["id"] == lead_id:
            current = lead["status"]
            if new_status not in VALID_TRANSITIONS[current]:
                raise ValueError(
                    f"invalid transition: '{current}' → '{new_status}'. "
                    f"Allowed: {VALID_TRANSITIONS[current]}"
                )
            lead["status"] = new_status
            lead["updated_at"] = date.today().isoformat()
            _save(path, leads)
            return lead
    raise ValueError(f"Lead '{lead_id}' not found")


def get_pipeline_stats(path: Path) -> dict[str, int]:
    leads = _load(path)
    stats: dict[str, int] = {"prospect": 0, "qualified": 0, "won": 0, "lost": 0}
    for lead in leads:
        stats[lead["status"]] = stats.get(lead["status"], 0) + 1
    return stats

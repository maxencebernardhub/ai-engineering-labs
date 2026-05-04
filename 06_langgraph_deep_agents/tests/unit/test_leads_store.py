import pytest

from shared.leads_store import (
    add_lead,
    add_note,
    get_pipeline_stats,
    list_leads,
    update_lead_status,
)

VALID_STATUSES = ["prospect", "qualified", "won", "lost"]


def test_list_leads_returns_all(leads_file):
    leads = list_leads(leads_file)
    assert len(leads) == 8


def test_list_leads_filter_by_status(leads_file):
    prospects = list_leads(leads_file, status_filter="prospect")
    assert len(prospects) == 2
    assert all(prospect["status"] == "prospect" for prospect in prospects)


def test_add_lead_creates_with_defaults(leads_file):
    lead = add_lead(leads_file, name="Test User", company="ACME", email="t@acme.fr")
    assert lead["name"] == "Test User"
    assert lead["company"] == "ACME"
    assert lead["email"] == "t@acme.fr"
    assert lead["status"] == "prospect"
    assert lead["notes"] == []
    assert "id" in lead
    assert "created_at" in lead
    assert "updated_at" in lead
    leads = list_leads(leads_file)
    assert len(leads) == 9


def test_add_note_appends_to_list(leads_file):
    lead = add_note(leads_file, lead_id="lead_001", note="Called back, very interested")
    assert "Called back, very interested" in lead["notes"]
    assert len(lead["notes"]) == 3


def test_update_status_valid_transition(leads_file):
    lead = update_lead_status(leads_file, lead_id="lead_001", new_status="qualified")
    assert lead["status"] == "qualified"


def test_update_status_invalid_transition(leads_file):
    with pytest.raises(ValueError, match="invalid transition"):
        update_lead_status(leads_file, lead_id="lead_001", new_status="won")


def test_update_status_lead_not_found(leads_file):
    with pytest.raises(ValueError, match="not found"):
        update_lead_status(leads_file, lead_id="lead_999", new_status="qualified")


def test_get_pipeline_stats_counts(leads_file):
    stats = get_pipeline_stats(leads_file)
    assert stats["prospect"] == 2
    assert stats["qualified"] == 2
    assert stats["won"] == 2
    assert stats["lost"] == 2

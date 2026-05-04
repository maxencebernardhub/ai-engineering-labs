import json

from langchain_core.tools import BaseTool

from shared import tools as t


def test_all_tools_have_schema():
    tool_fns = [
        t.list_leads_tool,
        t.add_lead_tool,
        t.add_note_tool,
        t.update_lead_status_tool,
        t.generate_email_draft_tool,
        t.get_pipeline_stats_tool,
    ]
    for tool in tool_fns:
        assert isinstance(tool, BaseTool), f"{tool} is not a BaseTool"
        assert tool.name, f"{tool} has no name"
        assert tool.description, f"{tool} has no description"


def test_generate_email_draft_creates_file(leads_file, drafts_dir, monkeypatch):
    monkeypatch.setattr(t, "LEADS_PATH", leads_file)
    monkeypatch.setattr(t, "DRAFTS_DIR", drafts_dir)

    result = t.generate_email_draft_tool.invoke(
        {"lead_id": "lead_001", "intent": "follow-up on Premium plan interest"}
    )

    files = list(drafts_dir.glob("*.json"))
    assert len(files) == 1, "Expected exactly one draft file"
    assert "lead_001" in result


def test_email_draft_file_structure(leads_file, drafts_dir, monkeypatch):
    monkeypatch.setattr(t, "LEADS_PATH", leads_file)
    monkeypatch.setattr(t, "DRAFTS_DIR", drafts_dir)

    t.generate_email_draft_tool.invoke(
        {"lead_id": "lead_002", "intent": "initial outreach"}
    )

    draft_file = next(drafts_dir.glob("*.json"))
    draft = json.loads(draft_file.read_text())
    assert "subject" in draft
    assert "to" in draft
    assert "body" in draft
    assert draft["to"] == "b.lefevre@dupont-fils.fr"

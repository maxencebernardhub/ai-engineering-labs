"""Agent-tool tests.

`build_tools(store)` returns the six LangChain tools the agent shares across both
engines, each bound to a `LeadStore`. Tests run against an `InMemoryStore` and
check that the tools carry a schema, persist through the store, warn (rather than
crash) on domain errors, and that `generate_email_draft` surfaces the draft in
its result — as a structured artifact — instead of writing a file.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.agents.tools import build_tools
from app.storage import InMemoryStore

TOOL_NAMES = {
    "list_leads",
    "add_lead",
    "add_note",
    "update_lead_status",
    "generate_email_draft",
    "get_pipeline_stats",
}


def _tools_by_name(store: InMemoryStore) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in build_tools(store)}


def test_all_tools_have_schema():
    tools = build_tools(InMemoryStore())
    assert {tool.name for tool in tools} == TOOL_NAMES
    for tool in tools:
        assert isinstance(tool, BaseTool), f"{tool} is not a BaseTool"
        assert tool.description, f"{tool.name} has no description"
        assert tool.args_schema is not None, f"{tool.name} has no args schema"


def test_add_lead_tool_persists():
    store = InMemoryStore()
    tools = _tools_by_name(store)

    result = tools["add_lead"].invoke(
        {"name": "Alice Moreau", "company": "Tech Solutions SAS", "email": "a@ts.fr"}
    )

    assert "Alice Moreau" in result
    leads = store.list_leads()
    assert len(leads) == 1
    assert leads[0]["company"] == "Tech Solutions SAS"
    assert leads[0]["status"] == "prospect"


def test_list_leads_tool_formats():
    store = InMemoryStore()
    tools = _tools_by_name(store)

    assert "No leads" in tools["list_leads"].invoke({})

    lead = store.add_lead("Alice Moreau", "Tech Solutions SAS", "a@ts.fr")
    out = tools["list_leads"].invoke({})
    assert lead["id"] in out
    assert "Alice Moreau" in out
    assert "Tech Solutions SAS" in out
    assert "prospect" in out


def test_list_leads_tool_filters_by_status():
    store = InMemoryStore()
    tools = _tools_by_name(store)
    kept = store.add_lead("A", "ACo", "a@co.fr")
    other = store.add_lead("B", "BCo", "b@co.fr")
    store.update_status(kept["id"], "qualified")

    out = tools["list_leads"].invoke({"status_filter": "qualified"})
    assert kept["id"] in out
    assert other["id"] not in out


def test_add_note_tool_persists():
    store = InMemoryStore()
    tools = _tools_by_name(store)
    lead = store.add_lead("A", "ACo", "a@co.fr")

    tools["add_note"].invoke({"lead_id": lead["id"], "note": "Called back"})
    assert store.list_leads()[0]["notes"] == ["Called back"]


def test_update_lead_status_tool_persists_valid_transition():
    store = InMemoryStore()
    tools = _tools_by_name(store)
    lead = store.add_lead("A", "ACo", "a@co.fr")  # prospect

    result = tools["update_lead_status"].invoke(
        {"lead_id": lead["id"], "new_status": "qualified"}
    )
    assert "qualified" in result
    assert store.list_leads()[0]["status"] == "qualified"


def test_update_lead_status_tool_warns_on_invalid_transition():
    store = InMemoryStore()
    tools = _tools_by_name(store)
    lead = store.add_lead("A", "ACo", "a@co.fr")  # prospect

    # prospect -> won is not allowed: the tool warns instead of raising, so the
    # agent can relay the reason to the user (lab-06 guardrail preserved).
    result = tools["update_lead_status"].invoke(
        {"lead_id": lead["id"], "new_status": "won"}
    )
    assert "invalid transition" in result.lower()
    assert store.list_leads()[0]["status"] == "prospect"


def test_update_lead_status_tool_warns_on_missing_lead():
    store = InMemoryStore()
    tools = _tools_by_name(store)
    result = tools["update_lead_status"].invoke(
        {"lead_id": "lead_missing", "new_status": "qualified"}
    )
    assert "lead_missing" in result
    assert "not found" in result.lower()


def test_email_draft_returned_not_written(tmp_path, monkeypatch):
    # Run from an empty cwd so we can assert nothing is written to disk.
    monkeypatch.chdir(tmp_path)
    store = InMemoryStore()
    lead = store.add_lead("Alice Moreau", "Tech Solutions SAS", "a@ts.fr")
    tools = _tools_by_name(store)

    # Invoke with a full tool call so the ToolMessage exposes the structured
    # artifact alongside the human-readable content.
    message = tools["generate_email_draft"].invoke(
        {
            "type": "tool_call",
            "name": "generate_email_draft",
            "args": {"lead_id": lead["id"], "intent": "follow-up"},
            "id": "call_1",
        }
    )

    draft = message.artifact
    assert draft["lead_id"] == lead["id"]
    assert draft["to"] == "a@ts.fr"
    assert "follow-up" in draft["subject"].lower()
    assert draft["body"]
    assert draft["generated_at"]

    # Draft is surfaced in the result, never persisted to a file.
    assert not (tmp_path / "drafts").exists()
    assert list(tmp_path.rglob("*.json")) == []


def test_email_draft_unknown_lead_returns_message():
    store = InMemoryStore()
    tools = _tools_by_name(store)
    result = tools["generate_email_draft"].invoke(
        {"lead_id": "lead_missing", "intent": "outreach"}
    )
    assert "lead_missing" in result
    assert "not found" in result.lower()


def test_pipeline_stats_tool_counts():
    store = InMemoryStore()
    tools = _tools_by_name(store)
    a = store.add_lead("A", "ACo", "a@co.fr")
    store.add_lead("B", "BCo", "b@co.fr")
    store.update_status(a["id"], "qualified")

    out = tools["get_pipeline_stats"].invoke({})
    assert "prospect: 1" in out
    assert "qualified: 1" in out
    assert "Total: 2" in out

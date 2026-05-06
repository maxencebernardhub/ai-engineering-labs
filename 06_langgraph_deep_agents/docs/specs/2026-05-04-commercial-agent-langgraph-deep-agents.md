# Feature: Commercial Assistant Agent — LangGraph vs Deep Agents

## Feature Brief

**Goal**: Build a commercial assistant agent for SMEs managing a sales leads pipeline,
implemented twice — LangGraph (low-level, explicit graph) and Deep Agents (high-level,
declarative) — to compare orchestration approaches, developer experience, and agent behavior.
Both agents are exposed through a shared Streamlit chat interface.

**Users**:

- Sales person at a SME (end user of the agent)
- Developer learning LangGraph vs Deep Agents (pedagogical audience)

---

## Lead Data Model

```json
{
  "id": "lead_001",
  "name": "Alice Dupont",
  "company": "Tech Solutions SAS",
  "email": "a.dupont@techsolutions.fr",
  "status": "prospect",
  "notes": ["First contact via website", "Interested in Premium plan"],
  "created_at": "2026-04-10",
  "updated_at": "2026-05-01"
}
```

Valid statuses: `prospect → qualified → won / lost`

---

## Tools (shared business logic)

Defined once in `shared/tools.py`, used by both agents:

| Tool | Description |
| --- | --- |
| `list_leads(status_filter?)` | List leads, optional filter by status |
| `add_lead(name, company, email)` | Create a new lead |
| `add_note(lead_id, note)` | Append a note to a lead's notes list |
| `update_lead_status(lead_id, new_status)` | Update lead status |
| `generate_email_draft(lead_id, intent)` | Write a `.json` draft in `drafts/` with `{subject, to, body}` |
| `get_pipeline_stats()` | Summary counts by status |

---

## Human-in-the-Loop (two distinct patterns)

1. **Review loop** — after `generate_email_draft`: the draft is displayed in the Streamlit UI;
   the user can approve, request adjustments, or cancel. The agent revises if needed.
2. **Confirmation gate** — before `update_lead_status → "lost"`: a confirmation prompt is shown
   in the UI; the agent proceeds only on explicit approval.

Both patterns are implemented via LangGraph `interrupt()` — surfaced as interactive UI
elements (buttons / chat input) in the Streamlit app.

---

## Streamlit App

A single shared `app.py` at the lab root, with:

- **Sidebar**:
  - Agent selector: `LangGraph` / `Deep Agents`
  - Provider selector: `Anthropic` / `OpenAI` / `Google`
  - Model selector: populated dynamically based on provider
- **Main area**: chat interface (message history + input box)
- **HITL rendering**: interrupt state surfaced as confirmation buttons or
  an adjustment input field within the chat flow

LLM instantiation is handled by `shared/config.py`:

```python
def get_llm(provider: str, model: str): ...
```

For notebook use, provider and model are set as variables at the top of each `demo.ipynb`.
For direct Python script execution, use env vars: `LLM_PROVIDER=openai LLM_MODEL=gpt-4o`.

---

## Acceptance Criteria

- **LangGraph agent**: explicit `StateGraph` with typed `State`, nodes, conditional edges,
  checkpointing, HITL via `interrupt`
- **Deep Agents agent**: same functionality via tool declarations + system prompt, no manual graph
- Both agents are accessible from the shared Streamlit app with agent + model selection
- HITL interrupts are rendered as interactive UI elements in the Streamlit chat
- Both agents pass the same set of test scenarios in their respective `demo.ipynb`
- Guardrails active on both: polite refusal of out-of-scope questions
- LangSmith traces captured for both agents
- `comparison.ipynb` covers:
  - Code metrics (lines of code, number of LLM calls, number of tool calls per scenario)
  - Qualitative evaluation on standard scenarios (task completion, tool accuracy,
    HITL behavior, guardrails)
  - Architectural comparison (control flow visibility, extensibility, debuggability)
  - LangSmith trace side-by-side (screenshots)

---

## Edge Cases

- Lead not found → clear error message, agent does not crash
- Invalid status transition (e.g. `prospect → won` skipping `qualified`) → agent warns and
  asks for confirmation
- Out-of-scope question → guardrail returns a polite refusal
- HITL review loop: user requests adjustments → agent regenerates the draft
- HITL confirmation gate: user refuses → agent cancels and confirms cancellation
- Streamlit rerun during HITL interrupt → interrupt state is preserved via `st.session_state`

---

## Dependencies

```text
langgraph
deepagents
langchain-anthropic
langchain-openai
langchain-google-genai
langsmith
streamlit
```

Environment variables (root `.env`):

- `ANTHROPIC_API_KEY` (already present)
- `OPENAI_API_KEY` (optional, if using OpenAI)
- `GOOGLE_API_KEY` (optional, if using Google)
- `LANGSMITH_API_KEY` (to be added)
- `LANGSMITH_PROJECT` (to be added)
- `LLM_PROVIDER` / `LLM_MODEL` (optional, for script execution — defaults to Anthropic)

---

## Constraints

- All code, comments, docstrings, and README written in English
- No real email sending, no external CRM API
- JSON file persistence only (`data/leads.json`)
- `drafts/` gitignored
- No separate intro notebooks — one `demo.ipynb` per agent

---

## Project Structure

```text
06_langgraph_deep_agents/
├── app.py                          # Shared Streamlit app (agent + model selectors, chat UI)
├── README.md
├── docs/
│   └── specs/
│       └── 2026-05-04-commercial-agent-langgraph-deep-agents.md
├── data/
│   └── leads.json                  # Versioned fictitious demo data (~8 leads)
├── drafts/                         # Generated email drafts (gitignored)
├── shared/
│   ├── config.py                   # get_llm(provider, model) factory
│   ├── leads_store.py              # CRUD operations on leads.json
│   └── tools.py                    # @tool decorated functions, shared by both agents
├── langgraph/
│   ├── agent/
│   │   ├── state.py                # TypedDict State definition
│   │   └── agent.py                # StateGraph orchestration
│   └── demo.ipynb
├── deep_agents/
│   ├── agent/
│   │   └── agent.py                # Deep Agents orchestration
│   └── demo.ipynb
├── comparison.ipynb
└── langsmith_traces/               # LangSmith screenshots for README
```

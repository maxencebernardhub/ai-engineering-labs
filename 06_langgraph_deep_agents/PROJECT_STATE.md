# Project State — Lab 06: LangGraph & Deep Agents

## Status

Phase 3 in progress — Step 10 complete. Starting Step 11 (demo notebooks).

## Completed Steps

- ✅ Step 1 — Project setup: `pyproject.toml`, `.python-version` (3.13.13), `.gitignore`,
  `data/leads.json`, `langsmith_traces/.gitkeep`, `uv sync` (161 packages, Python 3.13.13)
- ✅ Step 2 — `shared/leads_store.py`: CRUD on leads.json — 8/8 unit tests passing
- ✅ Step 3 — `shared/tools.py`: 6 `@tool` decorated functions — 3/3 unit tests passing (11/11 total)
- ✅ Step 4 — `shared/config.py`: `get_llm()` factory (Anthropic/OpenAI/Google) — 5/5 unit tests passing (16/16 total)
- ✅ Step 5 — `langgraph/agent/state.py`: AgentState TypedDict (LangGraph namespace package — no __init__.py needed)
- ✅ Step 6 — `langgraph/agent/agent.py`: full StateGraph (nodes, HITL, checkpointing)
- ✅ Step 7 — LangGraph integration tests: 4/4 non-integration passing; 5 @integration ready
- ✅ Step 8 — `deep_agents/agent/agent.py`: declarative Deep Agents agent (~70 lines vs ~150 for LangGraph)
- ✅ Step 9 — Deep Agents integration tests: 2/2 non-integration passing; 5 @integration ready
- ✅ Step 10 — `app.py`: Streamlit chat UI with agent/provider/model selectors and HITL panel
- ✅ Step 11 — Demo notebooks: `langgraph/demo.ipynb` and `deep_agents/demo.ipynb` (6 scenarios each, LangSmith tracing)
- 🔲 Step 12 — `comparison.ipynb`
- 🔲 Step 13 — `README.md`

## Context

Building a commercial assistant agent for SMEs managing a sales leads pipeline.
Implemented twice — LangGraph (low-level, explicit graph) and Deep Agents (high-level,
declarative) — to compare orchestration approaches, DX, and agent behavior.
Both agents exposed through a shared Streamlit chat interface.

Full spec: `docs/specs/2026-05-04-commercial-agent-langgraph-deep-agents.md`

---

## Implementation Plan

### Step 1 — Project setup

- `pyproject.toml` — dependencies, setuptools build config
- `.gitignore` — ignore `drafts/`, `__pycache__`, `*.egg-info`, `.env`
- `data/leads.json` — 8 fictitious leads in varied statuses
- `langsmith_traces/.gitkeep` — placeholder
- Run `uv sync`

### Step 2 — `shared/leads_store.py` (TDD)

Tests first, then implementation. CRUD operations on `leads.json`.

### Step 3 — `shared/tools.py` (TDD)

Tests first, then `@tool` decorated wrappers around leads_store.

### Step 4 — `shared/config.py` (TDD)

Tests first, then `get_llm(provider, model)` factory (Anthropic / OpenAI / Google).

### Step 5 — `langgraph/agent/state.py`

`TypedDict` State definition (messages, hitl_pending, thread_id).

### Step 6 — `langgraph/agent/agent.py`

Full `StateGraph`: nodes, conditional edges, checkpointing, HITL via `interrupt()`.

### Step 7 — LangGraph integration tests

6 scenarios in `tests/integration/test_langgraph_agent.py` (LLM mocked).

### Step 8 — `deep_agents/agent/agent.py`

Deep Agents orchestration: tool declarations + system prompt + HITL.

### Step 9 — Deep Agents integration tests

5 scenarios in `tests/integration/test_deep_agents_agent.py` (LLM mocked).

### Step 10 — `app.py`

Shared Streamlit app: agent selector, provider/model selector, chatbox, HITL rendering.

### Step 11 — Demo notebooks

`langgraph/demo.ipynb` and `deep_agents/demo.ipynb`.

### Step 12 — `comparison.ipynb`

Side-by-side analysis: code metrics, qualitative evaluation, architectural comparison,
LangSmith traces.

### Step 13 — `README.md`

---

## Files to Create (28 total)

### Project setup

- `pyproject.toml`
- `.gitignore`
- `data/leads.json`
- `langsmith_traces/.gitkeep`

### Shared layer

- `shared/__init__.py`
- `shared/leads_store.py`
- `shared/tools.py`
- `shared/config.py`

### Tests

- `tests/__init__.py`
- `tests/conftest.py`
- `tests/unit/__init__.py`
- `tests/unit/test_leads_store.py`
- `tests/unit/test_tools.py`
- `tests/unit/test_config.py`
- `tests/integration/__init__.py`
- `tests/integration/conftest.py`
- `tests/integration/test_langgraph_agent.py`
- `tests/integration/test_deep_agents_agent.py`

### LangGraph agent

- `langgraph/agent/state.py`
- `langgraph/agent/agent.py`
- `langgraph/demo.ipynb`

### Deep Agents agent

- `deep_agents/agent/agent.py`
- `deep_agents/demo.ipynb`

### App & notebooks

- `app.py`
- `comparison.ipynb`
- `README.md`

---

## Test Cases

### Unit — test_leads_store.py

- `test_list_leads_returns_all`
- `test_list_leads_filter_by_status`
- `test_add_lead_creates_with_defaults`
- `test_add_note_appends_to_list`
- `test_update_status_valid_transition`
- `test_update_status_invalid_transition`
- `test_update_status_lead_not_found`
- `test_get_pipeline_stats_counts`

### Unit — test_tools.py

- `test_all_tools_have_schema`
- `test_generate_email_draft_creates_file`
- `test_email_draft_file_structure`

### Unit — test_config.py

- `test_get_llm_anthropic`
- `test_get_llm_openai`
- `test_get_llm_google`
- `test_get_llm_default_model_per_provider`
- `test_get_llm_unknown_provider_raises`

### Integration — test_langgraph_agent.py (LLM mocked, `@pytest.mark.integration`)

- `test_list_leads_intent`
- `test_add_lead_intent`
- `test_guardrail_out_of_scope`
- `test_hitl_status_lost_triggers_interrupt`
- `test_hitl_email_draft_triggers_interrupt`
- `test_checkpointing_resumes_state`

### Integration — test_deep_agents_agent.py (LLM mocked, `@pytest.mark.integration`)

- `test_list_leads_intent`
- `test_add_lead_intent`
- `test_guardrail_out_of_scope`
- `test_hitl_status_lost_triggers_interrupt`
- `test_hitl_email_draft_triggers_interrupt`

---

## Risks

1. __Streamlit + LangGraph HITL (high)__ — `interrupt()` pauses the graph; Streamlit reruns on
   every interaction. Thread state must survive via `st.session_state`.
   Pattern: store `(thread_id, pending_action)` in session state, re-render HITL buttons
   on each rerun until resolved.

2. __deepagents maturity (medium)__ — less documented than LangGraph. HITL mechanism via
   `interrupt()` needs verification at implementation time.

3. __Integration tests cost (low)__ — LLM is mocked in tests. Real API calls reserved for
   demo notebooks only.

---

## Key Decisions

- Shared business logic in `shared/` (leads_store + tools + config) — both agents use same tools
- Single `app.py` at lab root — agent + model selectors in sidebar
- Provider/model: variable at top of notebooks; env vars for script execution
- Persistence: `data/leads.json` (versioned); `drafts/` gitignored
- LLM default: Anthropic Claude Sonnet 4.6
- Python: >= 3.13 (consistent with lab 05)

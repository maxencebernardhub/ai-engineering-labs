# Project State — Lab 08: FastAPI + Docker + AWS

## Status

🔵 Planning — Feature Brief locked (Phase 1 done). Implementation not started.

Branch: `feat/08-fastapi-docker-aws`

## Completed Steps

- ✅ Phase 1 — Brainstorming: goal, users, scope, cloud research (AWS free-tier July 2026),
  security model, storage strategy, deployment target all resolved.
- ✅ Feature Brief written and locked →
  `docs/specs/2026-07-03-08-fastapi-docker-aws.md`
- ✅ Git branch `feat/08-fastapi-docker-aws` and lab directory created.

## Next Steps

- 🔵 Phase 2 — Detailed Implementation Plan (ordered atomic steps, full test matrix). The plan
  outline below is provisional and will be finalized in Phase 2.
- 🔵 ADR checkpoint (optional) — candidate: "AWS Lambda Web Adapter vs Mangum".
- 🔵 Phase 3 — Dev environment init (`uv sync`).
- 🔵 Phase 4 — TDD implementation.
- 🔵 Phase 5 — Spec compliance review.
- 🔵 Phase 6 — Code review.
- 🔵 Phase 7 — Commit, PR, cleanup.

## Context

Productionize the lab 06 commercial assistant agent (LangGraph & Deep Agents) as a **stateless
FastAPI service**, containerized (multi-stage Dockerfile + `docker-compose` with PostgreSQL),
and deployed to **AWS Lambda** (container image + Function URL, via AWS Lambda Web Adapter) at
a durable **~0 € cost**. A minimal static frontend (served by FastAPI locally, hosted on S3 in
the cloud) lets recruiters/developers try the live app with their own LLM key (BYOK). The lab
demonstrates the full "notebook → deployed product" path.

Full spec: `docs/specs/2026-07-03-08-fastapi-docker-aws.md`

---

## Implementation Plan (provisional — to be finalized in Phase 2)

### Step 1 — Project setup

- `pyproject.toml`, `.python-version`, `.gitignore`, `.dockerignore`
- `data/seed_leads.json` (~8 fictitious leads)
- `uv sync`

### Step 2 — `app/storage/` (TDD)

`LeadStore` ABC + `memory_store.py`, `postgres_store.py` (SQLModel), `s3_store.py` (boto3),
selected by env var. Tests first.

### Step 3 — `app/agents/tools.py` (TDD)

6 tools backed by `LeadStore` (copied & adapted from lab 06). Tests first.

### Step 4 — `app/config.py` (TDD)

Settings + `get_llm(provider, model, api_key)` with header > env > 401 precedence. Tests first.

### Step 5 — `app/agents/langgraph_agent.py`

Stateless LangGraph agent (copied & adapted from lab 06, no interactive `interrupt()`).

### Step 6 — `app/agents/deep_agent.py`

Stateless Deep Agents agent (copied & adapted from lab 06).

### Step 7 — `app/schemas.py` + `app/security.py`

Pydantic request/response models; BYOK extraction, optional bearer auth, rate limit, CORS.

### Step 8 — `app/main.py` (TDD)

FastAPI app: `/health`, `/models`, `/invoke`, `/invoke/stream` (SSE), `/leads`, StaticFiles.
Integration tests with a fake LLM.

### Step 9 — `frontend/`

Static chat UI: provider/model/BYOK-key panel, SSE streaming. Served locally by FastAPI.

### Step 10 — Docker

Multi-stage `Dockerfile` (uv builder → slim non-root runtime + Lambda Web Adapter),
`docker-compose.yml` (`api` + `db` Postgres, health checks, volume).

### Step 11 — AWS deployment

`deploy/deploy.sh` (ECR → Lambda → Function URL, reserved concurrency, IAM),
`deploy/frontend_deploy.sh` (S3 static website), AWS Budget, live URL verification.

### Step 12 — Documentation

`README.md` (architecture diagram, "notebook → prod" narrative, local run + deploy steps,
screenshots); update root `README.md` (`08` → ✅, merge old `08`/`09` rows).

---

## Files to Create (provisional)

### Project setup

- `pyproject.toml`
- `.python-version`
- `.gitignore`
- `.dockerignore`
- `data/seed_leads.json`

### App layer

- `app/__init__.py`
- `app/main.py`
- `app/config.py`
- `app/schemas.py`
- `app/security.py`
- `app/agents/__init__.py`
- `app/agents/tools.py`
- `app/agents/langgraph_agent.py`
- `app/agents/deep_agent.py`
- `app/storage/__init__.py`
- `app/storage/base.py`
- `app/storage/memory_store.py`
- `app/storage/postgres_store.py`
- `app/storage/s3_store.py`

### Frontend

- `frontend/index.html`
- `frontend/app.js`
- `frontend/styles.css`

### Tests

- `tests/__init__.py`
- `tests/conftest.py`
- `tests/test_health.py`
- `tests/test_invoke.py`
- `tests/test_storage.py`

### Docker & deploy

- `Dockerfile`
- `docker-compose.yml`
- `deploy/deploy.sh`
- `deploy/frontend_deploy.sh`

### Docs

- `README.md`

---

## Test Cases (provisional — to be finalized in Phase 2)

### Unit — test_storage.py

- `test_memory_store_add_and_list`
- `test_list_leads_filter_by_status`
- `test_update_status_valid_transition`
- `test_update_status_invalid_transition`
- `test_update_status_lead_not_found`
- `test_get_pipeline_stats_counts`
- `test_store_selected_by_env_var`

### Unit — config / security

- `test_key_resolution_header_takes_precedence`
- `test_key_resolution_falls_back_to_env`
- `test_key_resolution_missing_returns_401`
- `test_get_llm_unknown_provider_raises`
- `test_provider_model_mismatch_rejected`
- `test_optional_auth_disabled_by_default`

### Integration — test_invoke.py (fake LLM)

- `test_health_ok`
- `test_models_lists_providers`
- `test_invoke_returns_reply`
- `test_invoke_stream_emits_sse`
- `test_invoke_add_lead_persists`
- `test_leads_endpoint_lists`
- `test_missing_key_returns_401`
- `test_unsupported_engine_returns_422`
- `test_rate_limit_enforced`

---

## Risks

1. **Lambda cold start with heavy deps (medium)** — langchain/langgraph inflate image size and
   init time. Mitigation: multi-stage build, trim deps, document warm-up.

2. **Stateless HITL adaptation (medium)** — lab 06 relies on interactive `interrupt()`. Must be
   reworked into autonomous execution with client-carried state. Verify agent behavior parity.

3. **AWS free-tier account closure (medium)** — new "Free Plan" accounts close after 6 months
   unless upgraded to Paid Plan. Must upgrade + stay within always-free caps + AWS Budget to
   keep 0 €.

4. **CORS between S3 frontend and Lambda API (low)** — different origins; needs correct
   CORS config on the API.

5. **Postgres on serverless anti-pattern (low, by design)** — deliberately avoided; Postgres is
   local-only, S3 is the cloud store.

---

## Key Decisions

- Stateless service — conversation history carried by the client; no server session store.
- Both engines exposed via `engine` param (preserves lab 06 comparison).
- Key resolution precedence: `X-LLM-API-Key` header > repo-root `.env` > `401` (BYOK in cloud,
  env keys locally).
- `LeadStore` abstraction: PostgreSQL (local) / S3 (cloud) / in-memory (tests).
- AWS Lambda Web Adapter (same uvicorn container local ↔ Lambda), not Mangum.
- Cloud target: Lambda container image + Function URL (streaming), reserved concurrency + AWS
  Budget for cost control.
- Agent core copied & adapted from lab 06 — lab 06 left intact, no cross-lab import.
- No lab-level `.env` — local keys from repo-root `.env` (as other labs).
- Python >= 3.13; `uv`; Ruff (line length 88); pytest.

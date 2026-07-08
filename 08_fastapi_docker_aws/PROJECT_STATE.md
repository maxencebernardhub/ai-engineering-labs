# Project State — Lab 08: FastAPI + Docker + AWS

## Status

🟢 Phase 4 (TDD implementation) — in progress. **Steps 0–5 done.** Next: Step 6
(FastAPI app).

Branch: `feat/08-fastapi-docker-aws`

## Completed Steps

- ✅ Phase 1 — Brainstorming: goal, users, scope, cloud research (AWS free-tier July 2026),
  security model, storage strategy, deployment target all resolved.
- ✅ Feature Brief written and locked →
  `docs/specs/2026-07-03-08-fastapi-docker-aws.md`
- ✅ Git branch `feat/08-fastapi-docker-aws` and lab directory created.
- ✅ Phase 2 — Detailed Implementation Plan locked (13 steps, full test matrix, risks) →
  `docs/specs/2026-07-03-08-fastapi-docker-aws-implementation-plan.md`
- ✅ ADR checkpoint — skipped; the "Lambda Web Adapter vs Mangum" rationale is folded into the
  Implementation Plan ("Design rationale" section).
- ✅ Phase 4 · **Step 0 — Scaffolding** (implementation started):
  - `pyproject.toml`, `.python-version` (3.13), lab-level `.gitignore`, `.dockerignore`.
  - `data/seed_leads.json` — **8 fictitious leads**, balanced spread
    (2 prospect / 2 qualified / 2 won / 2 lost), sequential ids, ASCII-only
    (accents/`€` removed for cross-backend safety; no real companies).
  - All runtime + dev deps added via `uv add` (latest stable `>=`, pinned in `uv.lock`);
    `pytest-asyncio` included for the async `astream` tests (Step 4).
  - Verified green: `uv sync`, heavy-import sanity check, `ruff check`, `ruff format --check`,
    `pytest` (0 tests yet — expected).
- ✅ Phase 4 · **Step 1 — Domain + storage layer** (TDD):
  - `app/domain.py` — `STATUSES`, `INITIAL_STATUS`, `VALID_TRANSITIONS`, `new_lead(...)`
    (random `lead_<6 hex>` ids, injectable `today`), `validate_transition(...)`; typed
    `LeadNotFoundError` / `InvalidTransitionError` (both `ValueError` subclasses).
  - `app/storage/` — `LeadStore` ABC + `ListBackedStore` (memory/S3 share `_load`/`_save`);
    `InMemoryStore`, `S3Store` (single JSON object; RMW race documented as known limitation),
    `PostgresStore` (SQLModel, JSON `notes`, SQLite-parity), `get_store(settings)` factory.
  - **Decision (confirmed with user)**: new-lead ids random (as lab 06); `seed_if_empty(leads)`
    added to the `LeadStore` interface — seeds demo data verbatim iff store empty (boot-time,
    idempotent); tests start empty.
  - `tests/test_storage.py` — 8-test contract suite parametrized over memory / SQLite / moto-S3,
    plus domain + factory + seed tests. **42 passed**, `ruff check` + `ruff format --check` clean.
  - ⚠️ **Cross-step contract for Step 2**: `get_store(settings)` (in `app/storage/__init__.py`)
    reads three attributes off `settings` via duck-typing (`typing.Protocol`): **`lead_store`**
    (`"memory"` | `"postgres"` | `"s3"`), **`database_url`** (`str | None`, required when
    `postgres`), **`leads_bucket`** (`str | None`, required when `s3`). The `Settings`
    (pydantic-settings) object built in Step 2 **must expose exactly these three attribute
    names** so it plugs into the factory unchanged.
- ✅ Phase 4 · **Step 2 — Config + BYOK key resolution** (TDD):
  - `app/config.py` — `Settings` (pydantic-settings, loads repo-root `../.env`, `extra="ignore"`)
    exposing the three storage attrs (satisfies the Step 1 contract) + `api_auth_token` +
    server-side keys. Google key accepts `GOOGLE_API_KEY`/`GEMINI_API_KEY` via `AliasChoices`
    (the shared `.env` uses `GEMINI_API_KEY`). `get_settings()` cached with `lru_cache`.
  - `resolve_api_key(header, provider, settings)` — single path header > env > `401`
    (`fastapi.HTTPException`); blank header and empty env value both treated as absent.
  - `get_llm(provider, model, api_key)` — BYOK key injected via the `api_key` alias (all three
    LangChain classes accept it); lazy per-provider imports; unknown provider → `ValueError`.
  - `SUPPORTED_PROVIDERS` / `DEFAULT_MODELS` / `PROVIDER_MODELS` — **model IDs refreshed &
    validated with the user (July 2026)**: defaults `claude-sonnet-5` / `gpt-5.4` /
    `gemini-3.1-flash-lite`; dropdown lists in `PROVIDER_MODELS`.
  - `tests/test_config.py` — 11 tests (precedence, GEMINI alias, empty-value handling, BYOK
    injection per provider, unknown provider, provider/model maps, storage-attr contract).
    **53 passed** total; `ruff check` + `ruff format --check` clean.
  - ⚠️ **Cross-step note for Step 5/6**: `resolve_api_key` raises `HTTPException(401)` directly
    (config coupled to FastAPI, per plan) — reuse it from `security.py`, don't re-map.
- ✅ Phase 4 · **Step 3 — Agent tools bound to a store** (TDD):
  - `app/agents/tools.py` — `build_tools(store)` returns the 6 `@tool`s (`list_leads`,
    `add_lead`, `add_note`, `update_lead_status`, `generate_email_draft`,
    `get_pipeline_stats`) as closures over a `LeadStore` — no module-level JSON path (adapted
    from lab 06 `shared/tools.py`, no cross-lab import). `app/agents/__init__.py` re-exports
    `build_tools`.
  - **HITL → autonomous**: lab 06's `interrupt()` gates are gone. Domain errors (unknown lead,
    illegal transition) are caught and returned as warning **strings** so the agent relays them
    (guardrail behavior preserved), instead of raising.
  - `generate_email_draft` uses `response_format="content_and_artifact"`: returns
    `(human_summary, draft_dict)` — the draft is surfaced as the ToolMessage **artifact** (no
    `drafts/` file, nothing written to disk). Step 4 runner reads `msg.artifact` to fill
    `email_draft` in the API response.
  - `tests/test_tools.py` — 11 tests vs `InMemoryStore` (schema/names, persistence, status
    filter, valid/invalid/missing transitions, draft-returned-not-written via a full tool call,
    unknown-lead message, pipeline stats). **64 passed** total; `ruff check` + `ruff format
    --check` clean.
  - ⚠️ **Cross-step contract for Step 4**: `generate_email_draft` surfaces the draft only as the
    ToolMessage `artifact` (its `content` is a short summary string). The runner must read
    `email_draft` from the tool message's `.artifact`, not parse the content.
- ✅ Phase 4 · **Step 4 — Agents (stateless, autonomous) + runner** (TDD):
  - `app/agents/langgraph_agent.py` — `build_langgraph_agent(llm, tools)` compiles the lab-06
    StateGraph **stripped of HITL and the checkpointer**: `agent`→`tools` nodes, conditional
    `agent→tools/end`, loop back through `agent`. LLM + tools are injected (BYOK-ready). Holds
    the shared `SYSTEM_PROMPT`.
  - `app/agents/deep_agent.py` — `build_deep_agent(model_or_string, tools)` =
    `create_deep_agent(..., interrupt_on={})` (autonomous, no checkpointer), same prompt/tools.
  - **Risk 1 resolved**: `create_deep_agent` accepts a `BaseChatModel` instance (verified,
    deepagents 0.6.12), so the Deep Agents path is unit-tested with the fake — **no
    `@integration` needed**.
  - `app/agents/runner.py` — `run(engine, llm, store, messages)` + async `astream(...)`; shared
    `_parse` flattens the message list into `reply`, `tool_calls` (`{name, args, result}` — the
    tool's output string is surfaced so the frontend can render an actions trace),
    `leads_touched`, `email_draft` (read from the ToolMessage **`.artifact`**), `usage`
    (`input/output/total` tokens). `leads_touched` from write-tool call args, plus the created
    id recovered from the `add_lead` result. `astream` yields `{"type":"token"}` deltas then a
    final `{"type":"final", ...}` (same shape as `run`) via `stream_mode=["messages","values"]`.
  - `tests/test_agents.py` — custom `FakeToolCallingModel` (implements `bind_tools`, `_generate`
    **and** `_stream`; repeats its last scripted response so Deep Agents' variable call count is
    safe). Covers list/add/guardrail intents, email-draft parsing, streaming (tokens+final),
    unknown-engine, and the Deep Agents smoke test. **72 passed** total; `ruff check` + `ruff
    format --check` clean.
  - ⚠️ **Cross-step contract for Steps 5/6**: the runner's structured dict is the source of
    truth for the `InvokeResponse` schema — `reply: str`, `tool_calls: [{name, args, result}]`,
    `leads_touched: [str]`, `email_draft: dict | None`, `usage: {input_tokens, output_tokens,
    total_tokens}`. SSE `/invoke/stream` consumes `astream`'s `token`/`final` events verbatim.

- ✅ Phase 4 · **Step 5 — Schemas + security** (TDD):
  - `app/schemas.py` — `Message`, `InvokeRequest` (validates `engine`/`provider`;
    provider↔model coupling via `model_validator`; `messages` non-empty → all bad values
    surface as `422`), `ToolCall`, `EmailDraft`, `Usage`, `InvokeResponse`, `ModelsResponse`.
    Providers/models imported from `app.config` (single source of truth); `ENGINES` mirrored
    locally so schemas stay free of the heavy agent-graph imports — `test_engines_mirror_runner`
    guards against drift. `InvokeResponse` round-trips the runner dict unchanged.
  - `app/security.py` — `get_api_key` (BYOK `X-LLM-API-Key` header dependency; resolution/401
    stays in `config.resolve_api_key`), `require_auth` (env-gated bearer, no-op when
    `API_AUTH_TOKEN` unset, constant-time compare), `limiter` + `init_rate_limiter` (slowapi,
    per-IP; applied per-route via `@limiter.limit(RATE_LIMIT)` so `/health` is never throttled),
    `configure_cors` + `configure_security` convenience wiring.
  - **Config touch**: added `Settings.cors_origins` (comma-separated, default `"*"`; no
    credentials → wildcard safe). **pyproject**: `flake8-bugbear.extend-immutable-calls` for
    FastAPI's `Depends`/`Header`/`Query` defaults (avoids B008 across `security.py`/`main.py`).
  - `tests/test_schemas.py` (10) + `tests/test_security.py` (6) — validation/coupling/422,
    auth disabled-by-default & rejects-bad-token, rate-limit 429, CORS wildcard & restricted.
    **88 passed** total; `ruff check` + `ruff format --check` clean.
  - ⚠️ **Cross-step contract for Step 6**: import security dependencies from `app.security`
    (`get_api_key`, `require_auth`, `limiter`, `RATE_LIMIT`, `configure_security`); decorate
    **only** `/invoke` and `/invoke/stream` with `@limiter.limit(RATE_LIMIT)` (each needs a
    `request: Request` param), add `Depends(require_auth)` on the agent routes, and call
    `configure_security(app, settings)` at app-factory time. `/models` fills `ModelsResponse`
    with `PROVIDER_MODELS`, `DEFAULT_MODELS`, and a per-provider `server_keys` flag derived from
    the configured server-side keys.

## Phase 2 refinements (changelog vs the initial provisional plan)

- **Live AWS URL is IN scope** — the feature is done only when the public URL is deployed,
  tested, and validated end-to-end (collaborative: user runs the guided AWS steps).
- **Storage tests**: SQLite + `moto` by default, plus a `@integration` Postgres smoke test.
- **CI**: minimal GitHub Actions (Ruff + non-integration pytest), at repo root with a path
  filter on `08_fastapi_docker_aws/**` (Step 12).
- **No mypy** — Pydantic (runtime, API boundary) + Ruff (lint) are enough; consistent with the
  other labs.
- **Dependency versions**: `>=` latest stable resolved via `uv add`, pinned in `uv.lock`.
- Tail reworked: added **Step 0** (scaffolding); **Step 10** = documentation; **Step 11** =
  guided live deployment & validation (was "AWS deployment"); **Step 12** = CI workflow (new).

## Next Steps

Branch is already created; env init (`uv sync`) runs during implementation Step 0, once
`pyproject.toml` exists. Mapping of the `/feature` phases to the plan steps:

- 🔵 Phase 4 — TDD implementation of plan **Steps 0–10 and 12** (scaffolding, code, tests,
  frontend, Docker, deploy scripts, docs, CI), including local Docker verification
  (build + `docker compose up`).
- 🔵 Phase 5 — Spec compliance review (`spec-reviewer`).
- 🔵 Phase 6 — Code review (`code-reviewer`).
- 🔵 Guided live deployment & validation — plan **Step 11**, collaborative: user runs the
  guided AWS steps; validate the public URL end-to-end **before merge** (live URL = done).
- 🔵 Phase 7 — Commit(s), PR, and post-merge cleanup.

## Future work (separate follow-up after merge)

- Repo-wide CI — generalize `ci-08.yml` into a matrix workflow covering all labs
  (`chore(ci): repo-wide matrix workflow`).

## Context

Productionize the lab 06 commercial assistant agent (LangGraph & Deep Agents) as a **stateless
FastAPI service**, containerized (multi-stage Dockerfile + `docker-compose` with PostgreSQL),
and deployed to **AWS Lambda** (container image + Function URL, via AWS Lambda Web Adapter) at
a durable **~0 € cost**. A minimal static frontend (served by FastAPI locally, hosted on S3 in
the cloud) lets recruiters/developers try the live app with their own LLM key (BYOK). The lab
demonstrates the full "notebook → deployed product" path.

Feature Brief: `docs/specs/2026-07-03-08-fastapi-docker-aws.md`
Implementation Plan: `docs/specs/2026-07-03-08-fastapi-docker-aws-implementation-plan.md`

---

## Implementation Plan & Test Matrix

The full, finalized plan — 13 ordered steps, complete file inventory, full test matrix,
risks, and the "Lambda Web Adapter vs Mangum" design rationale — lives in a dedicated
document, the single source of truth (to avoid drift):

`docs/specs/2026-07-03-08-fastapi-docker-aws-implementation-plan.md`

High-level step sequence: **0** scaffolding · **1** domain+storage · **2** config+BYOK ·
**3** tools · **4** agents · **5** schemas+security · **6** FastAPI app · **7** frontend ·
**8** Docker · **9** deploy scripts+docs · **10** documentation · **11** guided live
deployment & validation · **12** CI workflow.

---

## Key Decisions

- Stateless service — conversation history carried by the client; no server session store.
- Both engines exposed via `engine` param (preserves lab 06 comparison).
- Key resolution precedence: `X-LLM-API-Key` header > repo-root `.env` > `401`; BYOK enforced
  in cloud (no server keys deployed), root `.env` keys used locally.
- `LeadStore` abstraction: PostgreSQL (local) / S3 (cloud) / in-memory (tests).
- AWS Lambda Web Adapter (same uvicorn container local ↔ Lambda), not Mangum.
- Cloud target: Lambda container image + Function URL (streaming), reserved concurrency + AWS
  Budget for cost control. Deploy tooling: Bash + AWS CLI.
- **Live AWS URL is in scope** — feature is done only when the public URL is validated
  end-to-end (collaborative: user runs the guided AWS steps).
- Storage tests: SQLite + `moto` by default, plus a `@integration` Postgres smoke test.
- CI: minimal GitHub Actions (Ruff + non-integration pytest), repo-root workflow with a path
  filter on `08_fastapi_docker_aws/**`.
- **No mypy** — Pydantic (runtime validation at the API boundary) + Ruff (lint) suffice;
  consistent with the other labs.
- Dependency versions: `>=` latest stable resolved via `uv add`, pinned in `uv.lock`.
- **uv packaging = non-package** (`[tool.uv] package = false`, `pytest pythonpath = ["."]`):
  `app` is a deployable service, not an installable library — no build-system, no editable
  install, simpler Dockerfile. (Differs from lab 06, which packaged a shared `shared/` module.)
- Agent core copied & adapted from lab 06 — lab 06 left intact, no cross-lab import.
- No lab-level `.env` — local keys from repo-root `.env` (as other labs).
- Python >= 3.13; `uv`; Ruff (line length 88); pytest.

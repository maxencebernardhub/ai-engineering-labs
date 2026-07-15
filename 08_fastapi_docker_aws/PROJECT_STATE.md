# Project State — Lab 08: FastAPI + Docker + AWS

## Status

🟢 Phase 4 (TDD implementation) — **Steps 0–12 done** (0–8 code/Docker, **9+11 merged**:
deploy scripts authored **and** executed live, **10**: documentation, **12**: CI workflow).
**Phase 4 is complete.** Next: Phase 5 (commit/PR).

**🚀 The lab is live in AWS** (`ca-central-1`, deployed 2026-07-14):

- **App (S3 static website)** → `http://lab08-frontend-maxencebernardhub.s3-website.ca-central-1.amazonaws.com`
- **API (Lambda Function URL)** → `https://pynmzop7b3cmyjhubxulyozkai0ydmsh.lambda-url.ca-central-1.on.aws/`

Both READMEs flag this as a **demo deployment that may be taken offline**, and point at
`deploy/` to rebuild it — so a teardown does not invalidate the docs.

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

- ✅ Phase 4 · **Step 6 — FastAPI app** (TDD, integration):
  - `app/main.py` — `create_app(settings=None)` factory: `configure_security(app, settings)`,
    `get_store(settings)` + `store.seed_if_empty(_load_seed())` at boot (8 demo leads), store on
    `app.state.store`; `app.include_router(router)`; defensive `StaticFiles` mount at `/` (only
    when `frontend/` exists — absent until Step 7). Module-level `app = create_app()` is the
    ASGI entry point (`app.main:app`).
  - Routes: `GET /health` → `{"status":"ok"}` (never throttled); `GET /models` (fills
    `ModelsResponse`; `server_keys` via new `config.has_server_key`); `POST /invoke`
    (`runner.run` → `InvokeResponse`); `POST /invoke/stream` (SSE via `sse-starlette`
    `EventSourceResponse`, consumes `runner.astream` verbatim as `data: <json>`); `GET /leads`
    (+ `?status=`). `/invoke*` carry `@limiter.limit(RATE_LIMIT)` + `Depends(require_auth)`; BYOK
    resolved via `resolve_api_key(get_api_key(...), provider, settings)`. Error mapping:
    missing key → 401, bad engine/provider/model → 422, LLM/provider failure → `_provider_error`
    (504 on timeout, else 502; SSE emits an `error` event since status is already sent).
  - **Design seams (test overrides)**: `get_lead_store` (reads `request.app.state.store`) and
    `get_llm_factory` (returns `config.get_llm`); tests swap them via `app.dependency_overrides`.
  - ⚠️ **slowapi gotcha (fixed)**: the module-level `limiter` records **one limit per decorated
    endpoint the first time it's seen**. Defining the routes *inside* the app factory re-ran the
    `@limiter.limit` decorator on every `create_app`, stacking duplicate limits on the same
    endpoint key (`app.main.invoke`) → each request cost N hits, silently shrinking the cap
    (surfaced as a flaky 429 in tests). Fix: routes live on a **module-level `APIRouter`**
    decorated once at import; `create_app` only `include_router`s. Keep it this way.
  - `config.py` touch: added `has_server_key(provider, settings)` + shared `_server_key_value`
    helper (refactored `resolve_api_key` onto it) — drives `/models`' `server_keys` flag without
    abusing the 401 exception.
  - `tests/conftest.py` — `client_factory` fixture (builds a `TestClient` over `create_app`,
    swaps the two seams + `get_settings`; keyless `make_settings` via init kwargs so ambient env
    keys can't leak in), reuses `FakeToolCallingModel`/`fake_llm`/`_tool_call` from
    `test_agents`; autouse `_reset_rate_limiter` clears the shared limiter's counters around
    every test (the limiter stays **enabled** so the 429 test is real).
  - `tests/test_api.py` — the 9 matrix tests (health, models+key flag, invoke reply/persist, SSE,
    leads+filter, 401, 422, 429). **97 passed** total; `ruff check` + `ruff format --check` clean.
    Verified the real module-level app boots against the repo-root `.env` (health/models/leads,
    all 5 OpenAPI paths).
  - ⚠️ **Cross-step contract for Step 7 (frontend)**: drop the bundle in `frontend/` (served at
    `/` automatically once present). SSE frames are `data: {"type":"token","content":...}` then
    `data: {"type":"final", ...<InvokeResponse fields>}`; `GET /models` returns `providers`,
    `default_models`, `server_keys` (per-provider bool → whether the BYOK field is required).
    BYOK key goes in the `X-LLM-API-Key` header. ⚠️ **For Step 8 (Docker)**: ASGI entry point is
    `app.main:app`.

- ✅ Phase 4 · **Step 6.1 — Anthropic streaming-thinking fix** (found during manual live testing):
  - **Symptom**: `POST /invoke/stream` failed on any **tool-using** run with the default
    Anthropic model — Anthropic `400 messages.N.content.0.thinking.thinking: Field required`,
    surfaced (correctly) as an SSE `error` event. Sync `/invoke` was unaffected.
  - **Root cause** (traced into `langchain-anthropic` 1.4.8, the latest): recent Claude models
    use *adaptive* extended thinking whose reasoning text defaults to `display: "omitted"`, so
    the API returns a `thinking` block with a **signature but no text**. The streaming path
    rebuilds that block from a lone `signature_delta` → aggregated block has only
    `type`+`signature`; the outgoing formatter (`_format_messages`, keys
    `type/thinking/cache_control/signature`) then serializes it **without** the required
    `thinking` field. Replaying it on the next tool-loop turn is rejected. The non-streaming
    path keeps an empty `thinking` key, which is why sync worked.
  - **Scope (measured live)**: streaming only, tool-loop turns only; **both** engines (langgraph +
    deep_agents); **Anthropic only** (OpenAI `gpt-5.4` + Google `gemini-3.1-flash-lite` fine);
    among Anthropic models **only `claude-sonnet-5`** (our default) — `claude-opus-4-8` streams
    thinking text, `claude-haiku-4-5` doesn't think. No documented API/param workaround exists.
  - **Fix**: `app/anthropic_compat.py` → `ThinkingSafeChatAnthropic(ChatAnthropic)` overrides the
    single payload choke point `_get_request_payload` (shared by `_generate`/`_agenerate`/
    `_stream`/`_astream`) to backfill `thinking: ""` on any outgoing thinking block missing it —
    exactly what the non-streaming path sends, and what Anthropic accepts. `config.get_llm`'s
    anthropic branch now returns this subclass (still lazy-imported; `isinstance ChatAnthropic`
    holds, so existing tests pass). Offline regression test in `tests/test_config.py`
    (`test_anthropic_backfills_omitted_thinking_block`). **98 passed**; ruff clean.
  - **Verified live**: restarted uvicorn, re-ran the failing flows — langgraph SSE (email draft)
    and deep_agents SSE (list+stats) both stream tokens with **0 error events** and correct final
    events.

- ✅ Phase 4 · **Step 6.2 — Exhaustive live provider/model matrix** (manual verification):
  - Drove the real app (in-process `TestClient`, limiter disabled for the sweep) over **all 3
    providers × all 3 models × {`/invoke`, `/invoke/stream`} × {with tools, without tools}**, plus
    a `deep_agents` spot-check per provider, with real LLM calls. Static routes
    (health/models/leads+filter/422/401) all correct.
  - **Result**: 8/9 models fully green on both endpoints and both engines (streaming-with-tools
    included — the Anthropic thinking fix holds across opus/sonnet/haiku and both engines).
  - **Found**: `PROVIDER_MODELS["google"]` listed **`gemini-3-flash`**, which is **not a real
    model ID** (404 NOT_FOUND — verified against Google's live model list; the FastAPI layer
    surfaced it correctly as 502 / SSE error event). **Fixed**: replaced with
    **`gemini-3-flash-preview`** (the actual gemini-3 flash; re-verified green across all 4 cells).
    `gemini-3.5-flash` + `gemini-3.1-flash-lite` (default) were already valid. **98 passed**; ruff
    clean.

- ✅ Phase 4 · **Step 7 — Frontend (vanilla, no build step)** (live-verified in a real browser):
  - `frontend/index.html` + `frontend/app.js` + `frontend/styles.css` — chat window + settings
    sidebar, served at `/` by the Step 6 `StaticFiles(html=True)` mount. No framework, no bundler;
    `API_BASE` defaults to same-origin and is the single injection point `deploy/frontend_deploy.sh`
    (Step 9) will rewrite to the Lambda Function URL for the S3-hosted page.
  - **Design**: sober "product" look chosen with the user (from a 3-way visual mockup); token-driven
    light/dark (`prefers-color-scheme` + in-app `data-theme` toggle, persisted), responsive (sidebar
    stacks above chat < 760px).
  - **Contract consumed**: `GET /models` drives the provider→model **filtered** dropdowns, the
    per-provider default, and the **Required/Optional** BYOK badge (from `server_keys`); `POST
    /invoke/stream` consumed via **`fetch` + manual SSE parsing** (not `EventSource` — the endpoint is
    a POST with a body and the `X-LLM-API-Key` header); renders the streamed reply plus `tool_calls`
    (collapsible trace), `email_draft` (card), `leads_touched` (chips), and `usage`; `GET /leads` feeds
    a live Pipeline panel (auto-refreshed after each run). BYOK key stored **per provider** in
    `localStorage`; conversation history persisted in `localStorage` (stateless server, client owns
    state); `/docs` link in a new tab.
  - ⚠️ **Two bugs found during live browser testing (curl had masked both) and fixed**:
    1. **SSE frame split** — `sse-starlette` separates events with `\r\n\r\n`; the client split on
       `\n\n`, so **no frame ever parsed** and every prompt returned "The response ended unexpectedly."
       (curl+python read line-wise in universal-newline mode, hiding it). Fix: strip raw `\r` bytes on
       decode before splitting (`app.js`).
    2. **Sidebar overlap** — `.sidebar-foot { margin-top:auto }` inside an `overflow` flex column made
       the docs link / New-conversation button overlap the Pipeline list. Fix: real flex distribution
       (`.leads` grows with an internally-scrolling list; foot is `flex-shrink:0`).
  - **Verified live (Chrome, driven)**: streaming works across **all 3 providers × both engines**;
    filtered model dropdown; Actions trace; `leads_touched` + Pipeline auto-refresh (8→9); email-draft
    card; theme toggle + persistence across reload; conversation/provider/engine persistence; **no
    console errors**. Purely static assets — no new Python code, tests still **98 passed**; ruff clean.
  - ⚠️ **Note for Step 8 (Docker)**: local runs use `LEAD_STORE=memory` (default) — leads survive
    browser reloads only because the uvicorn process stays up, **not** durably; Compose sets
    `LEAD_STORE=postgres` for real persistence. Cloud uses `s3`.

- ✅ Phase 4 · **Step 8 — Docker** (built + `docker compose up` verified locally):
  - `Dockerfile` — multi-stage: **builder** (`python:3.13-slim` + pinned `uv 0.11.26` copied from
    `ghcr.io/astral-sh/uv`) runs `uv sync --frozen --no-dev --no-install-project` into `/app/.venv`
    (deps copied before app code so the dep layer caches); **runtime** (`python:3.13-slim`, non-root
    `appuser`, uid 10001) carries only the venv + `app/` + `frontend/` + `data/`. Bundles the **AWS
    Lambda Web Adapter** binary from `public.ecr.aws/awsguru/aws-lambda-adapter:1.0.1` (pinned,
    multi-arch; latest stable per user) at `/opt/extensions/lambda-adapter` — inert outside Lambda;
    `AWS_LWA_PORT=8000` + `AWS_LWA_READINESS_CHECK_PATH=/health`. Entry point `uvicorn app.main:app`
    on `0.0.0.0:8000`, same image local ↔ Lambda.
  - `docker-compose.yml` — `api` (`build: .`; `8000:8000`; `env_file: ../.env` for BYOK-optional LLM
    keys; `environment:` pins `LEAD_STORE=postgres` + `DATABASE_URL=postgresql+psycopg://leads:leads@db:5432/leads`
    + `CORS_ORIGINS=*`; `depends_on: db: condition: service_healthy`; Python-based `/health` healthcheck
    since the slim image has no curl) + `db` (`postgres:16`, named volume `leads_pgdata`, `pg_isready`
    healthcheck, port `5432` published so the host can run the integration test). `.dockerignore`
    already excluded `.env`/tests/docs/deploy (Step 0).
  - `tests/test_postgres_integration.py` — `test_postgres_store_parity`, marked `@integration` +
    skipped unless `DATABASE_URL` is a Postgres URL; runs the store contract against **real Postgres**
    and asserts **durability** (a fresh `PostgresStore` on a new connection re-reads the persisted
    lead). Truncates the `lead` table around itself. Default `pytest -m "not integration"` and CI
    never touch a DB. **98 passed, 1 deselected**; ruff clean.
  - **Verified live** (Docker Desktop, daemon up): image builds (~30 s deps layer); `docker compose up`
    → both services **healthy**; `GET /` serves the frontend from the container, `/health`, `/models`
    (`server_keys` all True from `../.env`), `/leads` = 8 seeded rows; **`psql` confirms 8 rows in the
    `lead` table** (really Postgres, not memory). **Durability proven**: added a lead via `POST /invoke`
    → `docker compose restart api` → lead still present (9 leads); a `memory` backend would reset to 8.
    Integration test green against the Compose DB. Left the stack re-seeded to 8 (integration test
    empties the table; restarting `api` re-seeds).
  - ⚠️ **Note for Step 9 (deploy)**: same image → ECR → Lambda (container + Function URL, streaming
    enabled so LWA can stream SSE); cloud env sets `LEAD_STORE=s3` + `LEADS_BUCKET`, **no** LLM keys
    (BYOK enforced). Build with `--platform` matching the Lambda arch.

- ✅ Phase 4 · **Steps 9 + 11 (merged) — AWS deploy scripts authored AND executed live**
  (2026-07-14, guided session; user drove all account/IAM/credential steps):
  - **Deliverables**: `deploy/deploy.sh` (7 idempotent stages: `preflight`/`ecr`/`iam`/`s3`/
    `lambda`/`url`/`concurrency`, `set -euo pipefail`, vars at top, `--help`, single-stage
    invocation), `deploy/frontend_deploy.sh` (bucket + public policy + website hosting + injects
    the Function URL into `window.API_BASE_URL` + `s3 sync`), `deploy/README.md` (architecture,
    Docker↔AWS interfaces, per-stage explanations, reproducible step-by-step, new-machine memo,
    gotchas, cost model, teardown).
  - **Guardrails first**: 2 AWS Budgets created **before any resource** (zero-spend + 1 $/month).
    IAM user `lab08-deployer` (CLI-only, 5 managed policies); `aws configure` → `ca-central-1`.
  - **Deployed resources** (all `ca-central-1`, arch **arm64**): ECR `lab08-commercial-agent` ·
    IAM role `lab08-lambda-role` (least-privilege) · S3 `lab08-leads-maxencebernardhub` (private,
    all public access blocked) · Lambda `lab08-commercial-agent` (image, 2048 MB, 120 s,
    `LEAD_STORE=s3` + `LEADS_BUCKET` + `AWS_LWA_INVOKE_MODE=response_stream` + `CORS_ORIGINS`,
    **no LLM key**) · Function URL (`AuthType=NONE`, `InvokeMode=RESPONSE_STREAM`) · S3
    `lab08-frontend-maxencebernardhub` (public static website).
  - ⚠️ **Three real bugs found only because the scripts were run** (the whole point of merging 9+11):
    1. **Docker attestations** — buildx wraps images in an OCI image index with provenance by
       default; Lambda rejects that media type. Fix: `docker build --provenance=false`. (ECR's
       **Type** column must read `Image`, not *Image index*.)
    2. **Missing `s3:ListBucket` → boot crash** (`Runtime.ExitError: exit status 1`). Without
       `ListBucket`, `GetObject` on the not-yet-existing `leads.json` returns **403 AccessDenied**
       instead of **404 NoSuchKey** (S3 hides object existence from non-listers), so
       `seed_if_empty` raised and uvicorn exited. Fix: add `s3:ListBucket` on the **bucket** ARN.
       `GetObject` + `PutObject` + `ListBucket` is the idiomatic S3 policy.
    3. **Function URL 403 despite a correct config** — since **Oct 2025** a public Function URL
       requires **both** `lambda:InvokeFunctionUrl` **and** `lambda:InvokeFunction` in the
       resource policy. Fix: a second `add-permission` (unconditioned — `--function-url-auth-type`
       is only valid for `InvokeFunctionUrl`).
  - ⚠️ **Console false positive (documented)**: the Function URL page keeps showing *"missing
    permissions required for public access"* because its banner heuristic does not recognise the
    permissions split across **two** statements. The live `curl` (200) is the authority.
  - ⚠️ **Reserved concurrency not applied**: the account's Lambda concurrency limit is **10**, and
    reserving would push unreserved below the required minimum. **Non-fatal by design** — the
    script warns and continues; the account limit itself caps concurrency (and therefore cost).
    The reservation applies automatically on an account with a higher limit.
  - **Live validation (the lab's "done")**: `/health` 200 · `/models` with `server_keys` **all
    false** (BYOK enforced — no server key in the cloud) · `/leads` served from S3 · **agent runs
    end-to-end** (user tested Gemini and OpenAI/gpt-5.4 via BYOK, both engines) · **SSE streaming**
    works through the Function URL · **S3 persistence proven** (8 seed → 9 → 10 leads across
    browser restarts; `leads.json` grew in the bucket) · **CORS verified** cross-origin
    (`Sec-Fetch-Site: cross-site` + `Access-Control-Allow-Origin` naming the S3 origin).
  - **19 screenshots** captured in `docs/screenshots/` (all referenced: 7 in the lab README, 12 in
    `deploy/README.md`).

- ✅ Phase 4 · **Step 10 — Documentation**:
  - `README.md` (lab) — live-demo links, "what this lab demonstrates", the **notebook → production**
    comparison table vs lab 06, aligned architecture diagram + annotated file tree, API contract,
    local run (uvicorn + Compose), deploy pointer, key concepts (stateless · BYOK · `LeadStore` ·
    LWA vs Mangum · the `anthropic_compat` fix), tests, cost model.
  - Root `README.md` — the planned `08_fastapi_backend` + `09_docker_deploy` rows **merged** into a
    single `08_fastapi_docker_aws` row (✅ Available); **all 8 labs are now ✅, no "Planned" rows
    left**; intro reframed as a complete arc (lab 01 → lab 08) with a **live-demo callout**; a
    `08` "Key Files" section added; BYOK note under the root `.env` block.
  - **Verified**: markdownlint clean on all three READMEs; 54 root links + all 19 images resolve;
    live URLs return 200; the documented endpoints match the live OpenAPI; "8 seed leads" and
    "6 tools" match the code. ASCII diagrams regenerated **by script** (widths computed, not
    hand-counted) after alignment drift was spotted.

- ✅ Phase 4 · **Step 12 — CI workflow** (validated with `actionlint` + a clean-checkout dry run):
  - `.github/workflows/ci-08.yml` (repo **root** — GitHub only reads the root `.github/`), path-filtered
    on `08_fastapi_docker_aws/**` **and on the workflow file itself** (so a CI change is tested by CI).
    Job `quality`: `uv sync --frozen` → `ruff check .` → `ruff format --check .` →
    `uv run pytest -m "not integration"`, with `defaults.run.working-directory` set to the lab.
  - **Details that matter**: `actions/checkout@v7` + `astral-sh/setup-uv@v8` (latest, checked against the
    GitHub API); **uv pinned to `0.11.26`** — the same version as the Dockerfile, so CI / local / image
    resolve deps identically; cache keyed on `08_fastapi_docker_aws/uv.lock`; Python 3.13 comes from
    `.python-version` (not duplicated in YAML, which would drift); `uv sync --frozen` **fails on a stale
    lock** (free guardrail); `push: branches: [main]` + `pull_request` avoids double runs on PR branches;
    `permissions: contents: read`; `concurrency` cancels superseded runs. **No API key** — the suite
    drives the fake LLM; `not integration` deselects the Postgres test.
  - ⚠️ **Discovery — the pre-existing root `ci.yml` was never repo-wide**: added in the *lab 04* commit
    (`baf8c19`), named just `CI`, **no path filter**, running `04_multi_provider`'s unit tests on every
    push. Consequence: **every green "CI" check on the lab 08 commits was attesting lab 04's tests** —
    lab 08 had never been covered by CI at all.
  - **Decision (with user)** — restore coherence by *renaming*, not merging: `ci.yml` → **`ci-04.yml`**
    via `git mv` (rename detected, history preserved) + the same scoping treatment (path filter on
    `04_multi_provider/**`, explicit `name:`, `permissions`, `concurrency`). The **job body is
    deliberately untouched** (`uv sync --extra dev` + `pytest tests/unit/`), and its `checkout@v4` /
    `setup-uv@v4` pins are **left alone on purpose**: they work, and bumping an unrelated green workflow
    inside a lab-08 PR is needless risk. Both files now read `ci-<lab>.yml` / `CI — Lab <n>`.
  - **Rejected (for now) — merging both into one matrix workflow**: that *is* the documented Future work,
    and the labs are not uniform enough to just concatenate. Concretely: **08 is the only non-packaged
    lab** (`[dependency-groups]` → `uv sync --frozen`) while **01–07 are packaged** with an `extra:dev`
    (`uv sync --extra dev`); **08 uses a flat `tests/` + markers** (`-m "not integration"`) while 01–07
    physically split `tests/unit/` + `tests/integration/`; **`ci.yml` runs no ruff at all**, so
    harmonizing means enabling lint on 4 more labs; **labs 05/06/07 have tests but have never run in CI**
    (unknown green — 05/06 integration needs API keys, 07 needs Ollama); and GitHub's `paths:` filter is
    **workflow-level, not per matrix leg**, so per-lab filtering needs `dorny/paths-filter` + a dynamic
    matrix, or dropping filters entirely. All of it belongs in a focused PR, not this one.
  - **Verified**: `actionlint` clean on **both** workflows; the lab-08 pipeline run against a **fresh
    `git clone` with no repo-root `.env` and no exported keys** — **98 passed, 1 deselected**, ruff clean
    (proves the fake-LLM suite is truly keyless and makes no network call, which the local run couldn't
    show since `.env` was present); lab 04's exact CI command still green (**22 passed**).
  - ⚠️ **Trigger behaviour (verified)**: pushing to the feature branch fires **nothing** — both
    workflows scope `push:` to `main` and rely on `pull_request` elsewhere, which is what stops the
    duplicate runs. Checks therefore appear when the PR is opened, not on every push as the old
    unscoped `ci.yml` did. **This PR runs *both* workflows**: it renames `ci-04.yml`, a path that
    workflow's own filter watches. Only *later* PRs that leave lab 04 untouched will skip `ci-04.yml`.

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
- **Revision (2026-07-11): Steps 9 and 11 merged.** The deploy scripts are authored **and** run
  live in one guided session (write→run→fix), instead of writing them blind then executing later —
  a deploy script is only proven once it runs. User is creating the AWS account now. Guardrails:
  **1 $ AWS Budget + alert set up first**; the user alone handles account/payment/IAM/`aws
  configure`; resource-creating or cost-incurring commands are confirmed before running (read-only
  `describe`/`get` run freely). Rationale + collaboration rules are in the implementation plan
  ("Locked decisions" + the Step 9/11 sections).

## Next Steps

- ✅ Phase 4 — TDD implementation: **Steps 0–12 done** (code, tests, frontend, Docker, deploy
  scripts **executed live**, documentation, CI workflow). Live public URL validated end-to-end.
  **Phase 4 is complete.**
- 🔵 Phase 5 — Commit(s), PR, and post-merge cleanup.

**Resolved (2026-07-15)**: the live URLs are published in the root README, the lab README, and this
file. All three now state that the deployment is a **demo that may be taken offline**, and point at
`deploy/` to redeploy — so tearing the stack down (see `deploy/README.md` → *Teardown*) leaves the
documentation truthful instead of leaving dead links unexplained.

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

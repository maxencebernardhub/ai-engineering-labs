# Implementation Plan — Lab 08: FastAPI + Docker + AWS

Companion to the Feature Brief (`2026-07-03-08-fastapi-docker-aws.md`). Produced in Phase 2 of
the `/feature` workflow and validated before implementation.

## Locked decisions (Phase 2)

- **Live URL is in scope** — the feature is "done" only when a public AWS URL is deployed,
  tested, and validated end-to-end. Deployment is collaborative: the plan delivers scripts +
  a guide; the user (no AWS account yet) executes the manual AWS steps with guidance.
- **Revision (2026-07-11) — Steps 9 and 11 are merged.** The deploy scripts are authored **and**
  executed live in a single guided session, rather than written blind (Step 9) then run later
  (Step 11). Rationale: a deploy script is only proven once it runs, so authoring and first real
  execution belong in one tight write→run→fix loop (AWS CLI has many footguns: Function URL
  streaming, IAM trust policy, ECR auth, Lambda image platform). The user creates the AWS account
  now. Collaboration rules: the **user alone** creates the account, enters payment, creates the
  IAM user/keys, and runs `aws configure`; a **1 $ AWS Budget + email alert is set up first**,
  before any resource; anything that creates or costs (ECR/Lambda/S3/IAM) is shown and confirmed
  before running, while read-only `describe`/`get` calls run freely.
- **Deploy tooling**: Bash + AWS CLI (transparent, teaches the primitives).
- **Storage tests**: SQLite (Postgres store) + `moto` (S3 store), no infra required; **plus** a
  `@integration` Postgres smoke test run locally against the Docker Compose DB (Phase 4).
- **Local Docker verification**: build the image + `docker compose up` during Phase 4.
- **CI**: minimal GitHub Actions (Ruff + non-integration pytest), scoped to this lab.
- **No mypy** — Pydantic covers runtime validation at the API boundary; Ruff covers lint
  (consistent with the other 7 labs).
- **Dependency versions**: `>=` latest stable resolved by `uv add` at install time (not
  hardcoded); pinned in `uv.lock`.

---

## Architecture — request flow

```text
Client (S3 frontend / Swagger / curl)
  │  POST /invoke {engine, provider, model, messages[]}  + X-LLM-API-Key
  ▼
FastAPI (uvicorn; on Lambda via AWS Lambda Web Adapter)
  │  security: resolve key (header > env > 401), optional bearer auth, rate-limit, CORS
  │  config.get_llm(provider, model, api_key)   → BYOK-injected LLM
  │  agents: build_tools(store) → build_{langgraph|deep}_agent(llm, tools)
  │  runner: run agent (autonomous, no HITL) → parse reply/tool_calls/email_draft/usage
  ▼
LeadStore (env-selected): InMemory | Postgres (local) | S3 (cloud)
```

## Design rationale — AWS Lambda Web Adapter vs Mangum

**Decision: AWS Lambda Web Adapter (LWA), not Mangum.** Mangum wraps the ASGI app
(`Mangum(app)`) so the app runs **differently** on Lambda vs locally (two code paths; subtle
divergences around streaming and lifespan). LWA is a binary Lambda extension that runs the
**same uvicorn** inside Lambda and translates Function URL events into real local HTTP →
**identical runtime local ↔ cloud** and **native response streaming** (required for our SSE
`/invoke/stream`). Cost: one binary added to the image (pinned tag from
`public.ecr.aws/awsguru/aws-lambda-adapter`); inert outside Lambda, so local runs are
unaffected.

---

## Ordered steps (TDD where code is involved)

### Step 0 — Scaffolding

- `pyproject.toml` (deps via `uv add`), `.python-version` (3.13), `.gitignore`, `.dockerignore`
- `data/seed_leads.json` — ~8 fictitious leads (copied from lab 06)
- Run `uv sync`

### Step 1 — Domain + storage layer *(TDD)*

- `app/domain.py` — `VALID_TRANSITIONS`, `new_lead(...)`, `validate_transition(...)`, status
  constants (ported from lab 06 `leads_store.py`)
- `app/storage/base.py` — `LeadStore` ABC: `list_leads`, `add_lead`, `add_note`,
  `update_status`, `pipeline_stats`
- `app/storage/memory_store.py` — in-RAM implementation
- `app/storage/postgres_store.py` — SQLModel `Lead` table + CRUD (JSON `notes` column; **no
  Postgres-only features**, to preserve SQLite parity)
- `app/storage/s3_store.py` — boto3, JSON object persistence in the bucket
- `app/storage/__init__.py` — `get_store(settings)` factory (env `LEAD_STORE`)
- **Tests** `tests/test_storage.py` — one parametrized suite run against Memory + Postgres
  (SQLite) + S3 (moto)

### Step 2 — Config + BYOK key resolution *(TDD)*

- `app/config.py` — `Settings` (pydantic-settings, loads repo-root `../.env`),
  `SUPPORTED_PROVIDERS`, `DEFAULT_MODELS`, `PROVIDER_MODELS` map,
  `resolve_api_key(header, provider, settings)` (header > env > `401`),
  `get_llm(provider, model, api_key)` (**injects key explicitly** for BYOK)
- **Tests** `tests/test_config.py`

### Step 3 — Agent tools bound to a store *(TDD)*

- `app/agents/tools.py` — `build_tools(store)` factory returning the 6 `@tool`s;
  `generate_email_draft` **returns the draft dict** (no `drafts/` file)
- **Tests** `tests/test_tools.py` (against `InMemoryStore`)

### Step 4 — Agents (stateless, autonomous)

- `app/agents/langgraph_agent.py` — `build_langgraph_agent(llm, tools)`: `agent`→`tools`
  nodes, conditional `agent→tools/end`, **no HITL node, no checkpointer**
- `app/agents/deep_agent.py` — `build_deep_agent(model_or_string, tools)`:
  `create_deep_agent(..., interrupt_on={})` (autonomous)
- `app/agents/runner.py` — `run(engine, llm, store, messages)` + `astream(...)`; parses
  `reply`, `tool_calls`, `leads_touched`, `email_draft`, `usage`
- **Tests** `tests/test_agents.py` — LangGraph path with a **FakeToolCallingModel**; Deep
  Agents path via fake **if** `create_deep_agent` accepts a model instance, else `@integration`
  (see Risks)

### Step 5 — Schemas + security *(TDD)*

- `app/schemas.py` — `Message`, `InvokeRequest`, `InvokeResponse`, `EmailDraft`,
  `ModelsResponse` (Pydantic; validates `engine`/`provider`/`model`)
- `app/security.py` — BYOK header extraction, optional bearer-auth dependency
  (`API_AUTH_TOKEN`, off if unset), `slowapi` rate-limiter, CORS settings
- **Tests** `tests/test_security.py`

### Step 6 — FastAPI app *(TDD, integration)*

- `app/main.py` — app factory; routes `GET /health`, `GET /models`, `POST /invoke`,
  `POST /invoke/stream` (SSE via `sse-starlette`), `GET /leads`; `StaticFiles` at `/`;
  middleware wiring
- **Tests** `tests/test_api.py` — `TestClient` + fake LLM (all endpoints + error paths)

### Step 7 — Frontend (vanilla, no build step)

- `frontend/index.html`, `frontend/app.js`, `frontend/styles.css` — chat window, settings
  panel (provider→model filtered dropdown, BYOK key in `localStorage`), SSE streaming; calls
  `/models` to decide whether the key field is required

### Step 8 — Docker *(verified: build + `docker compose up`)*

- `Dockerfile` — multi-stage (`uv` builder → slim non-root runtime); copies the **AWS Lambda
  Web Adapter** binary from `public.ecr.aws/awsguru/aws-lambda-adapter` (pinned)
- `docker-compose.yml` — `api` (`env_file: ../.env`; non-secret config in `environment:`;
  healthcheck `/health`; `depends_on` db healthy) + `db` (`postgres:16`, named volume,
  healthcheck)
- **Verify** locally: `docker compose up`, `curl /health`, run the `@integration` Postgres
  smoke test against the live Compose DB

### Step 9 — AWS deploy scripts + docs — **merged with Step 11 (authored + run live together)**

> Per the 2026-07-11 revision, this step is executed **jointly with Step 11**: each script is
> written and then run against the real account in the same session (write→run→fix), not authored
> blind for later execution.

- `deploy/deploy.sh` — build → ECR push → create Lambda from image → Function URL (streaming) →
  reserved concurrency → IAM role (S3 least-privilege)
- `deploy/frontend_deploy.sh` — sync `frontend/` → S3 static website; inject Function URL
- `deploy/README.md` — step-by-step: create AWS account (Free Plan), set a **1 $ AWS Budget**,
  IAM user + CLI config, run scripts, CORS, teardown

### Step 10 — Documentation

- `README.md` — architecture diagram, "notebook → prod" narrative, local run + deploy guide,
  screenshot placeholders
- Update root `README.md` (row `08` → ✅, merge old `08`/`09` rows); update `PROJECT_STATE.md`

### Step 11 — Guided live deployment & validation *(collaborative; live URL = done)* — **merged into Step 9**

> Per the 2026-07-11 revision, this is no longer a separate later session: it runs **together with
> Step 9** (author each script, then execute it live immediately). Sequence within the merged
> session: (1) **1 $ AWS Budget + email alert first**, (2) AWS account / IAM user / `aws configure`
> (user-driven), (3) run `deploy.sh` stage by stage, (4) run `frontend_deploy.sh`, (5) CORS.

- Walk the user through `deploy/README.md`: AWS account, Budget, IAM/CLI, `deploy.sh`,
  `frontend_deploy.sh`, CORS
- Validate end-to-end on the live public URL: `/health`, `/invoke`, SSE streaming, S3
  persistence, BYOK enforced (no server key)
- Capture screenshots for the README (feeds Step 10)

### Step 12 — CI workflow

- `.github/workflows/ci-08.yml` — **at the repo root** (GitHub only reads root `.github/`),
  with a path filter on `08_fastapi_docker_aws/**`
- Job: `ruff check` + `ruff format --check` + `uv run pytest -m "not integration"`
- Designed to be generalizable later into a repo-wide matrix workflow (see Future Work)

---

## Dependencies

```text
# runtime
fastapi · uvicorn[standard] · pydantic · pydantic-settings · python-dotenv
langgraph · deepagents
langchain-anthropic · langchain-openai · langchain-google-genai
sqlmodel · psycopg[binary] · boto3 · slowapi · sse-starlette
# dev
pytest · pytest-asyncio · httpx · moto[s3] · ruff
```

Versions resolved via `uv add` (latest stable at install time), pinned in `uv.lock`. Lambda
Web Adapter is a container binary, not a pip dep. Streamlit and mypy intentionally excluded.

---

## Full test matrix

**Unit — `test_storage.py`** (parametrized: memory / sqlite / moto-s3)

- `test_add_and_list`
- `test_list_filter_by_status`
- `test_add_note_appends`
- `test_update_status_valid`
- `test_update_status_invalid_transition`
- `test_update_status_lead_not_found`
- `test_pipeline_stats_counts`
- `test_get_store_selected_by_env`

**Unit — `test_config.py`**

- `test_key_header_takes_precedence`
- `test_key_falls_back_to_env`
- `test_key_missing_returns_401`
- `test_get_llm_injects_key`
- `test_get_llm_unknown_provider_raises`
- `test_provider_models_map`

**Unit — `test_tools.py`**

- `test_all_tools_have_schema`
- `test_add_lead_tool_persists`
- `test_email_draft_returned_not_written`
- `test_list_leads_tool_formats`

**Unit — `test_security.py`**

- `test_auth_disabled_by_default`
- `test_auth_rejects_bad_token_when_enabled`
- `test_rate_limit_enforced`
- `test_cors_headers_present`

**Integration — `test_agents.py`** (fake LLM)

- `test_langgraph_list_leads_intent`
- `test_langgraph_add_lead_intent`
- `test_langgraph_guardrail_out_of_scope`
- `test_runner_parses_email_draft`
- `test_deep_agents_smoke` (fake-or-`@integration`)

**Integration — `test_api.py`** (fake LLM)

- `test_health_ok`
- `test_models_lists_providers_and_key_flag`
- `test_invoke_returns_reply`
- `test_invoke_persists_lead`
- `test_invoke_stream_emits_sse`
- `test_leads_endpoint_lists`
- `test_missing_key_returns_401`
- `test_unsupported_engine_returns_422`
- `test_rate_limit_429`

**Integration — `@pytest.mark.integration` (Docker, Phase 4)**

- `test_postgres_store_parity` — parametrized store suite against the real Compose Postgres

---

## Risks & open questions

1. **Deep Agents + fake LLM (medium)** — `create_deep_agent` may only accept a
   `"provider:model"` string, making a fake hard to inject. *Mitigation:* verify at impl
   whether it accepts a `BaseChatModel`; if not, cover the deep-agents path via `@integration`
   (real key) and unit-test only engine routing with the fake.
2. **SSE over Lambda Function URL (low/medium)** — response streaming must be enabled on the
   Function URL; validated in Step 11. Local SSE fully tested earlier.
3. **SQLite ↔ Postgres parity (low)** — mitigated by avoiding PG-only features + the Postgres
   smoke test.
4. **Cold start with heavy deps (low)** — image-size optimization; measured at deploy time.
5. **Lambda Web Adapter version pin (low)** — pin the adapter image tag for reproducibility.

---

## Out of scope (this feature)

- **Custom domain** (Route 53 + ACM) — cosmetic; the auto-generated Function URL / S3 website
  URL is sufficient for the live demo.
- Auth beyond the optional bearer token.

## Future work (separate follow-up after merge)

- **Repo-wide CI** — generalize the lab-08 workflow into a matrix GitHub Actions workflow
  covering all labs (`chore(ci): repo-wide matrix workflow`); it would then supersede
  `ci-08.yml`.

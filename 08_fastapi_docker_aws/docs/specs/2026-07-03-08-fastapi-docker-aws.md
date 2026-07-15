# Feature: Production Deployment — Commercial Agent as a FastAPI Service on AWS

## Feature Brief

**Goal**: Build lab `08_fastapi_docker_aws/`, demonstrating the full "notebook → deployed
product" path. Take the commercial assistant agent from lab 06 (LangGraph & Deep Agents),
expose it as a **stateless FastAPI service**, containerize it (multi-stage Dockerfile +
local `docker-compose` with PostgreSQL), and deploy it to **AWS Lambda** (container image +
Function URL, via the AWS Lambda Web Adapter) at a durable **~0 € cost**. A minimal static
frontend (served by FastAPI locally, hosted on S3 in the cloud) lets anyone try the live app
with their own LLM key (BYOK).

**Users**:

- Anyone landing on the GitHub repo via LinkedIn, freelance platforms, or job applications —
  **recruiters, tech managers, developers** — who wants to try the live app or read
  production-grade code.
- HTTP callers (integrations, demos) consuming the API programmatically.
- Developers learning how to productionize an agentic app (pedagogical audience).

---

## Core Design Decisions

| Concern | Decision | Rationale |
| --- | --- | --- |
| Agent state | **Stateless** — client carries full conversation history per request | Fits Lambda's ephemeral model; no external session store |
| Agent engines | Both **LangGraph** & **Deep Agents**, selectable via `engine` param | Preserves lab 06's comparative narrative |
| LLM keys | Local: server-side keys from repo-root `.env` (shared by other labs). Cloud: **BYOK** via `X-LLM-API-Key`. Single resolution path, precedence **header > env > 401** | Owner LLM cost 0 € in cloud; zero-config locally |
| API auth | Optional bearer token, **env-gated, disabled by default** | Open live demo, but skill demonstrated |
| Abuse control | Lambda reserved concurrency + AWS Budget (1 $) + per-IP rate limit | Bounds compute without blocking testers |
| Lambda adapter | **AWS Lambda Web Adapter** | Same uvicorn container runs identically local ↔ Lambda |
| Lead persistence | **`LeadStore` abstraction**: PostgreSQL (local) / S3 (cloud) / in-memory (tests) | Teaches "local stack ≠ serverless stack"; both always-free |
| Frontend | Static HTML/JS bundle: served by FastAPI locally, hosted on S3 in cloud | Clickable live demo; serverless full-stack (S3 + Lambda) |

---

## Lead Data Model

Reused from lab 06 (persisted via `LeadStore`, not a raw file):

```json
{
  "id": "lead_001",
  "name": "Alice Dupont",
  "company": "Tech Solutions SAS",
  "email": "a.dupont@techsolutions.fr",
  "status": "prospect",
  "notes": ["First contact via website"],
  "created_at": "2026-04-10",
  "updated_at": "2026-05-01"
}
```

Valid statuses: `prospect → qualified → won / lost`

---

## Agent Tools (copied & adapted from lab 06)

Defined in `app/agents/tools.py`, backed by `LeadStore`, shared by both engines:

| Tool | Adaptation for stateless API |
| --- | --- |
| `list_leads(status_filter?)` | Reads via `LeadStore` |
| `add_lead(name, company, email)` | Writes via `LeadStore` |
| `add_note(lead_id, note)` | Writes via `LeadStore` |
| `update_lead_status(lead_id, new_status)` | Writes via `LeadStore` |
| `generate_email_draft(lead_id, intent)` | **Returned in the API response** (no `drafts/` file, no sending) |
| `get_pipeline_stats()` | Reads via `LeadStore` |

**HITL adaptation**: lab 06's interactive `interrupt()` gates (draft approval, confirm before
`→ lost`) are replaced by **autonomous execution**; sensitive actions and drafts are surfaced
in the structured response, and the client re-invokes with explicit instructions if changes
are wanted (state carried client-side).

---

## API Contract

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness (Docker healthcheck + Lambda) → `{"status":"ok"}` |
| `GET` | `/models` | Supported `provider → [models]` map + whether server-side keys exist (frontend uses it to know if the BYOK field is required) |
| `POST` | `/invoke` | Run agent, return full JSON (`reply`, `tool_calls`, `leads_touched`, `email_draft?`, `usage`) |
| `POST` | `/invoke/stream` | Same input, **SSE** token stream + final structured event |
| `GET` | `/leads` | List persisted leads (optional `?status=` filter) |
| `GET` | `/` | Serve static frontend (local only; S3 serves it in cloud) |

Request body (`/invoke*`): `{ engine, provider, model, messages[] }` + optional header
`X-LLM-API-Key`. Validated by Pydantic. Provider↔model coupling validated server-side;
conversation history supplied by the client.

---

## Security & BYOK

- Key resolution is precedence-based (`X-LLM-API-Key` header, else root-`.env` server key,
  else `401`). BYOK is the enforced path in the cloud (no server keys deployed); local runs
  use the root `.env` keys with no header required.
- `X-LLM-API-Key` used to instantiate the LLM client in-memory only; **never logged, never
  persisted**. HTTPS in transit (Function URL).
- Provider mismatch / missing key → `401`; unsupported engine/provider/model → `422`.
- Optional `Authorization: Bearer` check (env `API_AUTH_TOKEN`; off if unset).
- Per-IP rate limit (e.g. `slowapi`). CORS restricted to the frontend origin(s).

---

## Docker & Compose

- **`Dockerfile`** — multi-stage: `uv`-based builder → slim non-root runtime; bundles the
  AWS Lambda Web Adapter; single image for local **and** Lambda.
- **`docker-compose.yml`** — `api` (build, port 8000, `env_file: ../.env` for repo-root LLM
  keys [local only], non-secret config such as `LEAD_STORE=postgres` / `DATABASE_URL` set with
  defaults in `environment:`, `depends_on` db healthy, healthcheck on `/health`) + `db`
  (`postgres:16`, named volume, healthcheck). Demonstrates multi-service networking, health
  checks, volumes.
- `.dockerignore` provided. **No lab-level `.env`** — LLM keys come from the repo-root `.env`.

---

## AWS Deployment (~0 €, always-free only)

- Build image → push to **ECR** → create **Lambda** from container image → **Function URL**
  (HTTPS, response streaming enabled).
- **Reserved concurrency** (e.g. 5); env `LEAD_STORE=s3`, `LEADS_BUCKET=…`; **no LLM keys** on
  server (BYOK).
- **IAM role**: Lambda → S3 read/write on the leads bucket (least privilege).
- **S3**: one bucket for leads data, one **static-website** bucket for the frontend
  (configured with the Function URL; CORS on the API).
- **AWS Budgets** monthly budget (1 $) with email alert.
- Reproducible **`deploy/deploy.sh`** + **`deploy/frontend_deploy.sh`** (AWS CLI), documented
  step-by-step in the README with screenshots.

**Free-tier note (July 2026)**: new AWS accounts are on the "Free Plan" (200 $ credits /
6 months, then account closes unless upgraded to Paid Plan). The **always-free** allowances
(Lambda 1M req + 400k GB-s, S3 5 GB, API Gateway 1M calls) persist indefinitely even on Paid
Plan as long as usage stays within caps → 0 € achievable with the guardrails above.

---

## Acceptance Criteria

- Stateless FastAPI service with `/health`, `/models`, `/invoke`, `/invoke/stream` (SSE),
  `/leads`; auto-generated OpenAPI at `/docs`.
- `engine=langgraph|deep_agents` + `provider`/`model` selection, Pydantic-validated; agent
  core copied & adapted from lab 06 (lab 06 untouched, no cross-lab import).
- BYOK via `X-LLM-API-Key` (never logged/persisted) with header > env > 401 precedence;
  optional env-gated bearer auth; per-IP rate limit; CORS.
- `LeadStore` abstraction with PostgreSQL, S3, and in-memory backends, selected by env var.
- Multi-stage Dockerfile + `docker-compose` (`api` + `db`) with health checks and a volume;
  both run locally with one command.
- Same image deployed to Lambda (container + Function URL) via the Lambda Web Adapter; **a
  live public HTTPS URL works end-to-end**.
- Static frontend (chat + provider/model/BYOK-key panel, SSE streaming) served by FastAPI
  locally and hosted on S3 in the cloud.
- Cost guardrails in place (reserved concurrency + AWS Budget) and documented.
- `pytest` suite green: unit + integration, using a **fake LLM** (no real API calls/keys).
- `README.md` with architecture diagram, "notebook → prod" narrative, local-run + deployment
  instructions; root `README.md` updated (`08` → ✅, merging old `08`/`09` rows).

---

## Edge Cases

- Missing/invalid LLM key → `401`, no leakage.
- Unsupported `engine`/`provider`/`model` → `422` with clear message.
- LLM provider error (timeout, quota) → `502/504`, actionable message.
- Cold start with heavy deps (langchain/langgraph) → optimized image, warm-up documented.
- Storage backend unavailable (Postgres/S3) → explicit error; reads degrade gracefully where
  possible.
- Concurrent users → naturally isolated (separate Lambda envs); no shared server state.
- Quota overrun → bounded by reserved concurrency + Budget alert.
- Invalid status transition (e.g. `prospect → won`) → agent warns (lab 06 behavior preserved).

---

## Dependencies

```text
fastapi
uvicorn[standard]
pydantic / pydantic-settings
langgraph
deepagents
langchain-anthropic / langchain-openai / langchain-google-genai
sqlmodel + psycopg[binary]      # Postgres (local)
boto3                            # S3 (cloud)
slowapi                          # rate limiting
pytest / httpx                   # tests
```

Infra: AWS account (Lambda, ECR, S3, IAM, Budgets), AWS CLI, Docker. LLM keys supplied by
callers (BYOK) in the cloud; repo-root `.env` locally. Reuses lab 06 agent logic
(copied & adapted).

---

## Constraints

- All code, comments, docstrings, README in **English**.
- **Stateless** — no server-side conversation; history carried by the client.
- **Self-contained** — no cross-lab imports; lab 06 left intact.
- **~0 € durable** — always-free services only (Lambda, S3) + guardrails.
- No real email sending, no external CRM (as lab 06).
- Secrets never committed; `.env` gitignored; BYOK key never logged.
- Local mode reuses the repo-root `.env` (shared across labs); **no lab-level `.env`**.
  Cloud deploys **no** LLM keys (BYOK only).
- Python via `uv`; Ruff (line length 88); pytest.

---

## Project Structure

```text
08_fastapi_docker_aws/
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml        # api (env_file: ../.env) + db (postgres)
│                             # NOTE: no lab-level .env — LLM keys come from repo-root .env
├── .dockerignore
├── docs/
│   ├── specs/2026-07-03-08-fastapi-docker-aws.md
│   └── decisions/                 # optional ADR(s)
├── app/
│   ├── main.py                    # FastAPI app, routes, middleware, StaticFiles
│   ├── config.py                  # settings + get_llm(provider, model, api_key)
│   ├── schemas.py                 # Pydantic request/response models
│   ├── security.py                # BYOK, optional auth, rate limit, CORS
│   ├── agents/
│   │   ├── langgraph_agent.py      # copied & adapted from lab 06 (stateless)
│   │   ├── deep_agent.py           # copied & adapted from lab 06
│   │   └── tools.py                # tools backed by LeadStore
│   └── storage/
│       ├── base.py                 # LeadStore ABC
│       ├── postgres_store.py
│       ├── s3_store.py
│       └── memory_store.py
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── tests/
│   ├── conftest.py                 # fake LLM, TestClient fixtures
│   ├── test_health.py
│   ├── test_invoke.py
│   └── test_storage.py
├── deploy/
│   ├── deploy.sh                   # build → ECR → Lambda → Function URL
│   └── frontend_deploy.sh          # sync frontend → S3 static website
└── data/
    └── seed_leads.json             # ~8 fictitious demo leads
```

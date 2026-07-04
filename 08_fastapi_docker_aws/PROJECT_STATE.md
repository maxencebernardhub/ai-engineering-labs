# Project State — Lab 08: FastAPI + Docker + AWS

## Status

🟢 Phase 4 (TDD implementation) — in progress. **Step 0 (scaffolding) done.** Next: Step 1
(domain + storage layer, TDD).

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

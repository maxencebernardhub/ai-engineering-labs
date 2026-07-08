"""FastAPI application: the stateless commercial-agent service.

`create_app(settings)` is the app factory. At boot it wires the security stack
(per-IP rate limiting + CORS), selects a `LeadStore` from configuration and seeds
the demo leads once (idempotent), and registers the routes. The same image runs
locally and on AWS Lambda (via the Lambda Web Adapter), so there is a single code
path — no serverless-specific branching here.

Routes:

* `GET  /health`        — liveness (never rate-limited; used by Docker/Lambda).
* `GET  /models`        — provider->model map + per-provider server-key flag.
* `POST /invoke`        — run the agent, return the structured result.
* `POST /invoke/stream` — same input, Server-Sent Events (token deltas + final).
* `GET  /leads`         — list persisted leads (optional `?status=`).
* `GET  /`              — the static frontend (mounted only if `frontend/` exists).

Two dependencies (`get_lead_store`, `get_llm_factory`) are thin seams the tests
override to inject a known store and a fake LLM.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sse_starlette.sse import EventSourceResponse

from app.agents import runner
from app.config import (
    DEFAULT_MODELS,
    PROVIDER_MODELS,
    SUPPORTED_PROVIDERS,
    Settings,
    get_llm,
    get_settings,
    has_server_key,
    resolve_api_key,
)
from app.schemas import InvokeRequest, InvokeResponse, Message, ModelsResponse
from app.security import (
    RATE_LIMIT,
    configure_security,
    get_api_key,
    limiter,
    require_auth,
)
from app.storage import LeadStore, get_store

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_SEED_PATH = _DATA_DIR / "seed_leads.json"
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Map the client-carried conversation roles to LangChain message classes.
_ROLE_TO_MESSAGE = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}

# Type of the LLM factory the routes call: (provider, model, api_key) -> model.
LLMFactory = Callable[[str, str | None, str], BaseChatModel]


# --------------------------------------------------------------------------- #
# Dependencies (test seams)
# --------------------------------------------------------------------------- #


def get_lead_store(request: Request) -> LeadStore:
    """The process-wide store built at boot (overridden in tests)."""
    return request.app.state.store


def get_llm_factory() -> LLMFactory:
    """Return the BYOK LLM factory. Overridden in tests to inject a fake model."""
    return get_llm


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_seed() -> list[dict]:
    """Load the demo leads shipped in `data/seed_leads.json` (or `[]` if absent)."""
    if _SEED_PATH.is_file():
        return json.loads(_SEED_PATH.read_text(encoding="utf-8"))
    return []


def _to_lc_messages(messages: list[Message]) -> list[BaseMessage]:
    """Convert the request's typed history into LangChain messages."""
    return [_ROLE_TO_MESSAGE[m.role](content=m.content) for m in messages]


def _provider_error(exc: Exception) -> HTTPException:
    """Map an LLM/provider failure to an actionable 5xx (504 timeout, else 502)."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if "timeout" in text or "timed out" in text:
        return HTTPException(
            status_code=504,
            detail="The LLM provider timed out. Please retry in a moment.",
        )
    return HTTPException(
        status_code=502,
        detail=f"The LLM provider returned an error: {exc}",
    )


def _resolve_llm(
    body: InvokeRequest,
    header_key: str | None,
    settings: Settings,
    factory: LLMFactory,
) -> BaseChatModel:
    """Resolve the BYOK key (header > env > 401) and build the chat model."""
    api_key = resolve_api_key(header_key, body.provider, settings)
    return factory(body.provider, body.model, api_key)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
#
# The router is built once, at import, so the `@limiter.limit` decorators run
# exactly once. The limiter is a module-level singleton that records a limit per
# decorated endpoint the first time it is seen; re-decorating (e.g. defining the
# routes inside the app factory, which the tests call repeatedly) would stack
# duplicate limits on the same endpoint and multiply the per-request cost. A
# module-level router that `create_app` merely `include_router`s avoids that.

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe (never rate-limited)."""
    return {"status": "ok"}


@router.get("/models", response_model=ModelsResponse)
def models(settings: Settings = Depends(get_settings)) -> ModelsResponse:
    """Providers, their models, defaults, and which have a server-side key."""
    return ModelsResponse(
        providers=PROVIDER_MODELS,
        default_models=DEFAULT_MODELS,
        server_keys={p: has_server_key(p, settings) for p in SUPPORTED_PROVIDERS},
    )


@router.post(
    "/invoke",
    response_model=InvokeResponse,
    dependencies=[Depends(require_auth)],
)
@limiter.limit(RATE_LIMIT)
def invoke(
    request: Request,
    body: InvokeRequest,
    header_key: str | None = Depends(get_api_key),
    settings: Settings = Depends(get_settings),
    store: LeadStore = Depends(get_lead_store),
    llm_factory: LLMFactory = Depends(get_llm_factory),
) -> dict:
    """Run the agent to completion and return the structured result."""
    llm = _resolve_llm(body, header_key, settings, llm_factory)
    try:
        return runner.run(body.engine, llm, store, _to_lc_messages(body.messages))
    except HTTPException:
        raise
    except Exception as exc:  # provider timeout/quota/etc.
        raise _provider_error(exc) from exc


@router.post("/invoke/stream", dependencies=[Depends(require_auth)])
@limiter.limit(RATE_LIMIT)
async def invoke_stream(
    request: Request,
    body: InvokeRequest,
    header_key: str | None = Depends(get_api_key),
    settings: Settings = Depends(get_settings),
    store: LeadStore = Depends(get_lead_store),
    llm_factory: LLMFactory = Depends(get_llm_factory),
) -> EventSourceResponse:
    """Stream the run as SSE: token deltas, then a final structured event."""
    llm = _resolve_llm(body, header_key, settings, llm_factory)
    messages = _to_lc_messages(body.messages)

    async def event_stream():
        try:
            async for event in runner.astream(body.engine, llm, store, messages):
                yield {"data": json.dumps(event)}
        except Exception as exc:  # can't change status mid-stream: emit an event
            err = _provider_error(exc)
            yield {"event": "error", "data": json.dumps({"detail": err.detail})}

    return EventSourceResponse(event_stream())


@router.get("/leads")
def list_leads(
    status: str | None = Query(default=None, description="Filter by status."),
    store: LeadStore = Depends(get_lead_store),
) -> list[dict]:
    """List persisted leads, optionally filtered by status."""
    return store.list_leads(status_filter=status)


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = settings or get_settings()

    app = FastAPI(
        title="Commercial Assistant API",
        version="1.0.0",
        description=(
            "Stateless FastAPI service exposing the lab-06 commercial agent "
            "(LangGraph & Deep Agents) with BYOK LLM keys."
        ),
    )

    configure_security(app, settings)

    store = get_store(settings)
    store.seed_if_empty(_load_seed())
    app.state.store = store
    app.state.settings = settings

    app.include_router(router)
    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the static frontend at `/` when present (local dev).

    Mounted last so the API routes take precedence; in the cloud S3 serves the
    frontend instead, so the directory may be absent — mount defensively.
    """
    if _FRONTEND_DIR.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=_FRONTEND_DIR, html=True),
            name="frontend",
        )


app = create_app()

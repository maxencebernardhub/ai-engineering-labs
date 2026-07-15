"""Security wiring for the API: BYOK key extraction, optional bearer auth,
per-IP rate limiting, and CORS.

Design notes:

* **BYOK** — `get_api_key` is a FastAPI dependency that reads the caller's LLM
  key from the `X-LLM-API-Key` header (or `None`). The key is passed to
  `app.config.resolve_api_key`, which owns the header > env > `401` precedence;
  it is used in-memory only and never logged or persisted.
* **Bearer auth** — `require_auth` is env-gated: with no `API_AUTH_TOKEN` set it
  is a no-op (the open demo), otherwise it requires a matching bearer token
  (compared in constant time).
* **Rate limiting** — a single `slowapi` limiter, applied per route via the
  `@limiter.limit(RATE_LIMIT)` decorator (the app decorates only `/invoke*`, so
  the Docker `/health` healthcheck is never throttled). `init_rate_limiter`
  registers the limiter and its `429` handler on the app.
* **CORS** — `configure_cors` restricts origins to `settings.cors_origins`
  (default `"*"`); no credentials are used, so the wildcard is safe.
"""

from __future__ import annotations

import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import Settings, get_settings

# Header carrying the caller's own LLM key (BYOK).
API_KEY_HEADER = "X-LLM-API-Key"

# Per-IP limit applied to the agent endpoints. Generous enough for a demo, low
# enough to bound abuse; combined with Lambda reserved concurrency in the cloud.
RATE_LIMIT = "20/minute"

# One process-wide limiter keyed by client IP. Routes opt in via the decorator;
# `init_rate_limiter` wires it (and the 429 handler) onto the app.
limiter = Limiter(key_func=get_remote_address)


def get_api_key(
    x_llm_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> str | None:
    """FastAPI dependency: the caller's BYOK key from the header, or `None`.

    Resolution/`401` is deferred to `app.config.resolve_api_key`, which also
    considers the server-side env key.
    """
    return x_llm_api_key


def require_auth(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """Optional bearer-token gate.

    Disabled (no-op) when `API_AUTH_TOKEN` is unset — the default open demo.
    When set, requires `Authorization: Bearer <token>` matching it exactly.
    """
    expected = settings.api_auth_token
    if not expected:
        return  # auth disabled

    header = request.headers.get("Authorization", "")
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(presented, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def init_rate_limiter(app: FastAPI) -> None:
    """Register the shared limiter and its `429` handler on the app."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def configure_cors(app: FastAPI, settings: Settings) -> None:
    """Add CORS middleware restricting origins to `settings.cors_origins`."""
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()] or [
        "*"
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,  # no cookies/sessions; keys travel in headers
        allow_methods=["*"],
        allow_headers=["*"],
    )


def configure_security(app: FastAPI, settings: Settings) -> None:
    """Convenience for Step 6: wire rate limiting and CORS in one call."""
    init_rate_limiter(app)
    configure_cors(app, settings)

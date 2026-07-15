"""Security wiring tests: optional bearer auth, per-IP rate limit, CORS.

Each test wires the relevant `app.security` helper into a throwaway FastAPI app
(the real `app.main` is built in Step 6) and drives it with `TestClient`, so the
tests exercise exactly the wiring the app will use.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.security import (
    RATE_LIMIT,
    configure_cors,
    init_rate_limiter,
    limiter,
    require_auth,
)


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


# --------------------------------------------------------------------------- #
# Optional bearer auth (env-gated, off when API_AUTH_TOKEN is unset)
# --------------------------------------------------------------------------- #


def _auth_app(token: str | None) -> FastAPI:
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_auth)])
    def protected():
        return {"ok": True}

    app.dependency_overrides[get_settings] = lambda: _settings(api_auth_token=token)
    return app


def test_auth_disabled_by_default():
    # No API_AUTH_TOKEN configured -> the dependency is a no-op, no header needed.
    client = TestClient(_auth_app(None))
    assert client.get("/protected").status_code == 200


def test_auth_rejects_bad_token_when_enabled():
    client = TestClient(_auth_app("s3cret"))
    assert client.get("/protected").status_code == 401  # missing header
    bad = client.get("/protected", headers={"Authorization": "Bearer wrong"})
    assert bad.status_code == 401
    good = client.get("/protected", headers={"Authorization": "Bearer s3cret"})
    assert good.status_code == 200


# --------------------------------------------------------------------------- #
# Per-IP rate limit
# --------------------------------------------------------------------------- #


def test_rate_limit_enforced():
    app = FastAPI()
    init_rate_limiter(app)

    @app.get("/ping")
    @limiter.limit("2/minute")
    def ping(request: Request):
        return {"pong": True}

    client = TestClient(app)
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 429  # third call over the limit


def test_default_rate_limit_is_a_valid_limit_string():
    # Sanity: the constant the app decorates /invoke* with parses as a limit.
    assert "/" in RATE_LIMIT


# --------------------------------------------------------------------------- #
# CORS
# --------------------------------------------------------------------------- #


def test_cors_headers_present():
    app = FastAPI()
    configure_cors(app, _settings())  # default origins -> "*"

    @app.get("/ping")
    def ping():
        return {"pong": True}

    client = TestClient(app)
    resp = client.get("/ping", headers={"Origin": "http://example.com"})
    assert resp.headers.get("access-control-allow-origin") == "*"


def test_cors_restricts_to_configured_origins():
    app = FastAPI()
    configure_cors(app, _settings(cors_origins="https://frontend.example"))

    @app.get("/ping")
    def ping():
        return {"pong": True}

    client = TestClient(app)
    allowed = client.get("/ping", headers={"Origin": "https://frontend.example"})
    assert (
        allowed.headers.get("access-control-allow-origin") == "https://frontend.example"
    )
    # A non-listed origin is simply not echoed back as allowed.
    other = client.get("/ping", headers={"Origin": "https://evil.example"})
    assert other.headers.get("access-control-allow-origin") != "https://evil.example"

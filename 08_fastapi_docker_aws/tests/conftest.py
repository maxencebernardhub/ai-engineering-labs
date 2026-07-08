"""Shared fixtures for the API integration tests.

The app is exercised through FastAPI's `TestClient` with the real security stack
wired in, but two seams are swapped out so no real LLM or backing store is
touched:

* `get_llm_factory` is overridden to return the scripted `FakeToolCallingModel`
  from `test_agents` (no network, deterministic tool calls);
* the lead store can be overridden with a fresh `InMemoryStore` so a test can
  assert on what the agent persisted.

The per-IP rate limiter is process-wide (a module-level `slowapi.Limiter`), so an
autouse fixture disables it and clears its counters around every test; the single
rate-limit test re-enables it explicitly.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app, get_lead_store, get_llm_factory
from app.security import limiter
from app.storage import InMemoryStore
from tests.test_agents import fake_llm  # reuse the scripted fake LLM


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Clear the shared limiter's counters around every test.

    The limiter is a single process-wide object, so without this its per-IP
    counters would leak between tests (one test's requests eating into another's
    budget). Resetting lets each test start from a clean allowance; the limiter
    stays enabled so the rate-limit test can observe a real 429.
    """
    limiter.reset()
    yield
    limiter.reset()


def make_settings(**overrides) -> Settings:
    """Build isolated settings: no `.env`, no server keys, memory store.

    Keys are forced to `None` via init kwargs (highest priority in
    pydantic-settings) so the developer's ambient environment can't leak a real
    key into the tests. Individual tests override what they need.
    """
    base: dict = {
        "anthropic_api_key": None,
        "openai_api_key": None,
        "google_api_key": None,
        "lead_store": "memory",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def build_client(
    *,
    responses: list | None = None,
    settings: Settings | None = None,
    store: InMemoryStore | None = None,
) -> TestClient:
    """Construct a `TestClient` over a freshly built app with the seams swapped.

    * `responses` — scripted `AIMessage`s for the fake LLM (omit to leave the
      real factory in place, e.g. for `/health` or `/models`).
    * `settings` — request-time settings (also used to build the boot store).
    * `store` — inject a known store to assert persistence; when omitted the
      app's own seeded store (8 demo leads) is used.
    """
    settings = settings or make_settings()
    app = create_app(settings)

    the_store = store if store is not None else app.state.store
    if store is not None:
        app.dependency_overrides[get_lead_store] = lambda: the_store
    app.dependency_overrides[get_settings] = lambda: settings
    if responses is not None:
        app.dependency_overrides[get_llm_factory] = lambda: (
            lambda provider, model, api_key: fake_llm(*responses)
        )

    client = TestClient(app)
    client.store = the_store  # convenience handle for assertions
    return client


@pytest.fixture
def client_factory():
    """Yield a builder for `TestClient`s, closing each one at teardown."""
    created: list[TestClient] = []

    def _factory(**kwargs) -> TestClient:
        client = build_client(**kwargs)
        created.append(client)
        return client

    yield _factory
    for client in created:
        client.close()

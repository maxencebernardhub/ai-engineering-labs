"""End-to-end API tests for the FastAPI app (Step 6).

Every endpoint and its error paths are driven through `TestClient` with a fake
LLM (see `conftest`), so nothing hits a real provider. These assert the wiring
the plan's test matrix calls for: health, models + key flag, invoke (reply and
persistence), SSE streaming, leads listing, and the 401/422/429 error paths.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.security import RATE_LIMIT, limiter
from app.storage import InMemoryStore
from tests.conftest import make_settings
from tests.test_agents import _tool_call


def _body(**overrides) -> dict:
    body = {
        "engine": "langgraph",
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(overrides)
    return body


_KEY = {"X-LLM-API-Key": "sk-byok-test"}


def test_health_ok(client_factory):
    client = client_factory()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_models_lists_providers_and_key_flag(client_factory):
    # A server key is configured for anthropic only.
    settings = make_settings(anthropic_api_key="sk-server")
    client = client_factory(settings=settings)

    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()

    assert set(data["providers"]) == {"anthropic", "openai", "google"}
    assert data["default_models"]["anthropic"] == "claude-sonnet-5"
    assert data["server_keys"] == {
        "anthropic": True,
        "openai": False,
        "google": False,
    }


def test_invoke_returns_reply(client_factory):
    responses = [AIMessage(content="I can help with your sales pipeline.")]
    client = client_factory(responses=responses)

    resp = client.post(
        "/invoke",
        json=_body(messages=[{"role": "user", "content": "hello"}]),
        headers=_KEY,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "pipeline" in data["reply"].lower()
    assert data["tool_calls"] == []
    assert data["email_draft"] is None
    assert data["usage"]["total_tokens"] >= 0


def test_invoke_persists_lead(client_factory):
    store = InMemoryStore()
    responses = [
        _tool_call(
            "add_lead",
            {"name": "Bob Martin", "company": "Acme", "email": "bob@acme.fr"},
        ),
        AIMessage(content="Added Bob Martin to the pipeline."),
    ]
    client = client_factory(responses=responses, store=store)

    resp = client.post(
        "/invoke",
        json=_body(messages=[{"role": "user", "content": "add Bob Martin at Acme"}]),
        headers=_KEY,
    )
    assert resp.status_code == 200

    leads = store.list_leads()
    assert len(leads) == 1
    assert leads[0]["name"] == "Bob Martin"
    assert resp.json()["leads_touched"] == [leads[0]["id"]]


def test_invoke_stream_emits_sse(client_factory):
    store = InMemoryStore()
    store.add_lead("Alice Moreau", "Tech Solutions SAS", "a@ts.fr")
    responses = [
        _tool_call("list_leads", {}),
        AIMessage(content="You have 1 lead."),
    ]
    client = client_factory(responses=responses, store=store)

    resp = client.post(
        "/invoke/stream",
        json=_body(messages=[{"role": "user", "content": "list my leads"}]),
        headers=_KEY,
    )
    assert resp.status_code == 200
    body = resp.text
    # Token deltas stream first, then a final structured event.
    assert '"type": "token"' in body
    assert '"type": "final"' in body
    assert "1 lead" in body


def test_leads_endpoint_lists(client_factory):
    client = client_factory()  # app's own seeded store (8 demo leads)

    resp = client.get("/leads")
    assert resp.status_code == 200
    assert len(resp.json()) == 8

    won = client.get("/leads", params={"status": "won"})
    assert won.status_code == 200
    leads = won.json()
    assert leads and all(lead["status"] == "won" for lead in leads)


def test_missing_key_returns_401(client_factory):
    # Keyless settings + no BYOK header -> resolve_api_key raises 401.
    client = client_factory(responses=[AIMessage(content="unused")])

    resp = client.post("/invoke", json=_body())
    assert resp.status_code == 401


def test_unsupported_engine_returns_422(client_factory):
    client = client_factory(responses=[AIMessage(content="unused")])

    resp = client.post("/invoke", json=_body(engine="nope"), headers=_KEY)
    assert resp.status_code == 422


def test_rate_limit_429(client_factory):
    client = client_factory(responses=[AIMessage(content="ok")])

    limiter.reset()
    limiter.enabled = True
    limit = int(RATE_LIMIT.split("/")[0])

    codes = [
        client.post("/invoke", json=_body(), headers=_KEY).status_code
        for _ in range(limit + 1)
    ]

    assert codes[:limit] == [200] * limit
    assert codes[-1] == 429

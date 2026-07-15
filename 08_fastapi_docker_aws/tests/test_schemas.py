"""Schema validation tests.

The request schemas are the API's first line of defence: bad `engine` /
`provider` / `model` values must fail validation (surfaced by FastAPI as `422`),
and the provider->model coupling is checked server-side against the canonical
maps in `app.config`. The response schemas mirror the runner's structured output
(the Step 4 cross-step contract), so a runner dict round-trips through them.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.runner import ENGINES as RUNNER_ENGINES
from app.config import DEFAULT_MODELS, PROVIDER_MODELS
from app.schemas import (
    ENGINES,
    InvokeRequest,
    InvokeResponse,
    ModelsResponse,
)


def _request(**overrides) -> dict:
    body = {
        "engine": "langgraph",
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(overrides)
    return body


# --------------------------------------------------------------------------- #
# InvokeRequest validation
# --------------------------------------------------------------------------- #


def test_valid_request_parses():
    req = InvokeRequest.model_validate(_request())
    assert req.engine == "langgraph"
    assert req.provider == "anthropic"
    assert req.messages[0].role == "user"


def test_model_is_optional_and_defaults_to_none():
    req = InvokeRequest.model_validate(_request(model=None))
    assert req.model is None  # the runner/get_llm applies the provider default


def test_unknown_engine_rejected():
    with pytest.raises(ValidationError):
        InvokeRequest.model_validate(_request(engine="crewai"))


def test_unknown_provider_rejected():
    with pytest.raises(ValidationError):
        InvokeRequest.model_validate(_request(provider="mistral"))


def test_model_provider_mismatch_rejected():
    # `gpt-5.4` is an OpenAI model, not valid under the anthropic provider.
    with pytest.raises(ValidationError):
        InvokeRequest.model_validate(_request(provider="anthropic", model="gpt-5.4"))


def test_empty_messages_rejected():
    with pytest.raises(ValidationError):
        InvokeRequest.model_validate(_request(messages=[]))


def test_engines_mirror_runner():
    # schemas keeps a local copy so it need not import the heavy agent graph;
    # this guards the two from drifting apart.
    assert ENGINES == RUNNER_ENGINES


# --------------------------------------------------------------------------- #
# InvokeResponse round-trips the runner's structured dict
# --------------------------------------------------------------------------- #


def test_response_accepts_runner_shape():
    runner_output = {
        "reply": "Done.",
        "tool_calls": [
            {"name": "add_lead", "args": {"name": "Acme"}, "result": "[lead_001] ..."}
        ],
        "leads_touched": ["lead_001"],
        "email_draft": {
            "lead_id": "lead_001",
            "to": "a@b.com",
            "subject": "Hi",
            "body": "Hello",
            "generated_at": "2026-07-08T10:00:00",
        },
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    resp = InvokeResponse.model_validate(runner_output)
    assert resp.reply == "Done."
    assert resp.tool_calls[0].name == "add_lead"
    assert resp.email_draft.to == "a@b.com"
    assert resp.usage.total_tokens == 15


def test_response_email_draft_optional():
    resp = InvokeResponse.model_validate(
        {
            "reply": "Nothing to do.",
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
    )
    assert resp.email_draft is None
    assert resp.tool_calls == []
    assert resp.leads_touched == []


# --------------------------------------------------------------------------- #
# ModelsResponse
# --------------------------------------------------------------------------- #


def test_models_response_shape():
    resp = ModelsResponse(
        providers=PROVIDER_MODELS,
        default_models=DEFAULT_MODELS,
        server_keys={"anthropic": True, "openai": False, "google": False},
    )
    assert resp.providers["anthropic"]
    assert resp.server_keys["anthropic"] is True

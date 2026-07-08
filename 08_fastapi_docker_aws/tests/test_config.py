"""Config + BYOK key-resolution tests.

Covers the single key-resolution path (header > server env key > 401), the
provider/model maps, and the BYOK-injecting `get_llm` factory. Settings are
built with `_env_file=None` so tests never depend on the real repo-root `.env`.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.config import (
    DEFAULT_MODELS,
    PROVIDER_MODELS,
    SUPPORTED_PROVIDERS,
    Settings,
    get_llm,
    resolve_api_key,
)


def _settings(**overrides) -> Settings:
    """Build isolated Settings (no `.env` file, no ambient env vars leaking in)."""
    return Settings(_env_file=None, **overrides)


# --------------------------------------------------------------------------- #
# Key resolution: header > server env key > 401
# --------------------------------------------------------------------------- #


def test_key_header_takes_precedence():
    settings = _settings(anthropic_api_key="sk-env")
    assert resolve_api_key("sk-header", "anthropic", settings) == "sk-header"


def test_key_falls_back_to_env():
    settings = _settings(anthropic_api_key="sk-env")
    assert resolve_api_key(None, "anthropic", settings) == "sk-env"


def test_key_google_env_accepts_gemini_alias():
    # The shared repo-root `.env` names the Google key `GEMINI_API_KEY`.
    settings = _settings(GEMINI_API_KEY="sk-gemini")
    assert resolve_api_key(None, "google", settings) == "sk-gemini"


def test_key_missing_returns_401():
    settings = _settings()  # no server keys configured
    with pytest.raises(HTTPException) as exc:
        resolve_api_key(None, "anthropic", settings)
    assert exc.value.status_code == 401


def test_key_empty_env_value_is_treated_as_missing():
    # An empty `ANTHROPIC_API_KEY=` line must not count as a usable key.
    settings = _settings(anthropic_api_key="")
    with pytest.raises(HTTPException) as exc:
        resolve_api_key(None, "anthropic", settings)
    assert exc.value.status_code == 401


# --------------------------------------------------------------------------- #
# get_llm: BYOK injection + validation
# --------------------------------------------------------------------------- #


def test_get_llm_injects_key():
    from langchain_anthropic import ChatAnthropic

    llm = get_llm("anthropic", "claude-sonnet-5", "sk-injected")
    assert isinstance(llm, ChatAnthropic)
    assert llm.anthropic_api_key.get_secret_value() == "sk-injected"
    assert llm.model == "claude-sonnet-5"


def test_get_llm_defaults_model_when_omitted():
    from langchain_openai import ChatOpenAI

    llm = get_llm("openai", None, "sk-injected")
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == DEFAULT_MODELS["openai"]


def test_get_llm_google_injects_key():
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = get_llm("google", None, "sk-injected")
    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.google_api_key.get_secret_value() == "sk-injected"


def test_get_llm_unknown_provider_raises():
    with pytest.raises(ValueError, match="[Uu]nknown provider"):
        get_llm("mistral", None, "sk-injected")


def test_anthropic_backfills_omitted_thinking_block():
    """A signature-only `thinking` block must get an empty `thinking` field.

    Streamed adaptive/`omitted` thinking yields blocks with only a signature;
    replaying them on a tool-loop turn is otherwise rejected by Anthropic with
    `thinking.thinking: Field required`. `get_llm` returns the shim that repairs
    the outgoing payload (checked offline — `_get_request_payload` makes no call).
    """
    from langchain_core.messages import AIMessage, HumanMessage

    from app.anthropic_compat import ThinkingSafeChatAnthropic

    llm = get_llm("anthropic", "claude-sonnet-5", "sk-injected")
    assert isinstance(llm, ThinkingSafeChatAnthropic)

    messages = [
        HumanMessage(content="list my leads"),
        AIMessage(  # a prior streamed assistant turn: signature-only thinking
            content=[
                {"type": "thinking", "signature": "sig-abc"},
                {"type": "text", "text": "Here you go."},
            ]
        ),
        HumanMessage(content="thanks"),
    ]
    payload = llm._get_request_payload(messages)

    thinking_blocks = [
        block
        for message in payload["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "thinking"
    ]
    assert thinking_blocks, "expected the thinking block to survive formatting"
    assert all(block.get("thinking") == "" for block in thinking_blocks)


# --------------------------------------------------------------------------- #
# Provider / model maps
# --------------------------------------------------------------------------- #


def test_provider_models_map():
    assert set(PROVIDER_MODELS) == set(SUPPORTED_PROVIDERS)
    assert set(DEFAULT_MODELS) == set(SUPPORTED_PROVIDERS)
    for provider, models in PROVIDER_MODELS.items():
        assert models, f"{provider} must list at least one model"
        assert DEFAULT_MODELS[provider] in models


# --------------------------------------------------------------------------- #
# Settings ⇄ storage factory contract (Step 1 cross-step contract)
# --------------------------------------------------------------------------- #


def test_settings_exposes_storage_attributes():
    settings = _settings()
    assert settings.lead_store == "memory"
    assert settings.database_url is None
    assert settings.leads_bucket is None

"""Application settings and the BYOK-injecting LLM factory.

Two responsibilities:

1. `Settings` (pydantic-settings) — configuration loaded from the environment,
   or, for local development, from the shared repo-root `.env` (no lab-level
   `.env`; see the feature brief). It exposes the three storage attributes the
   `app.storage.get_store` factory reads (`lead_store`, `database_url`,
   `leads_bucket`), the optional bearer-auth token, and the server-side LLM keys
   used locally.

2. Key resolution + LLM construction — `resolve_api_key` implements the single
   precedence path (request header > server-side env key > `401`), and `get_llm`
   builds a LangChain chat model with the resolved key injected explicitly so the
   cloud (BYOK) and local paths are identical.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException
from langchain_core.language_models import BaseChatModel
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# The repo-root `.env`, shared across labs. Used only for local development;
# in Docker and the cloud, configuration arrives as real environment variables.
_REPO_ROOT_ENV = Path(__file__).resolve().parent.parent.parent / ".env"

SUPPORTED_PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "google")

# Default model per provider (used when the request omits `model`). Kept in step
# with lab 06 where it still points at a current model, refreshed to the latest
# stable IDs (validated July 2026).
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-5.4",
    "google": "gemini-3.1-flash-lite",
}

# Models offered to the frontend dropdown, per provider. Deliberately small; the
# default above must appear in each list (guarded by `test_provider_models_map`).
PROVIDER_MODELS: dict[str, list[str]] = {
    "anthropic": ["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"],
    "openai": ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini"],
    "google": ["gemini-3.5-flash", "gemini-3-flash", "gemini-3.1-flash-lite"],
}

# provider -> the `Settings` attribute holding its server-side key.
_ENV_KEY_ATTR: dict[str, str] = {
    "anthropic": "anthropic_api_key",
    "openai": "openai_api_key",
    "google": "google_api_key",
}


class Settings(BaseSettings):
    """Runtime configuration, from the environment or the repo-root `.env`."""

    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT_ENV,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage selection — consumed by `app.storage.get_store` (Step 1 contract).
    lead_store: str = "memory"
    database_url: str | None = None
    leads_bucket: str | None = None

    # Optional bearer auth (Step 5); disabled while unset.
    api_auth_token: str | None = None

    # CORS allowed origins for the browser frontend, comma-separated. Defaults to
    # "*" (open demo; the API uses no cookies/credentials, so `*` is safe). In the
    # cloud, set this to the S3 static-website origin(s).
    cors_origins: str = "*"

    # Server-side LLM keys: present locally (BYOK optional), absent in the cloud
    # (BYOK enforced). The Google key is named `GEMINI_API_KEY` in the shared
    # repo-root `.env`; accept `GOOGLE_API_KEY` too for portability.
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached `Settings` instance (the app's dependency)."""
    return Settings()


def resolve_api_key(header_key: str | None, provider: str, settings: Settings) -> str:
    """Resolve the LLM key: request header > server-side env key > `401`.

    A blank header or an empty server-side key value counts as absent — only a
    non-empty string is a usable key.
    """
    if header_key and header_key.strip():
        return header_key

    attr = _ENV_KEY_ATTR.get(provider)
    secret = getattr(settings, attr) if attr else None
    value = secret.get_secret_value() if secret else None
    if value:
        return value

    raise HTTPException(
        status_code=401,
        detail=(
            "No LLM API key provided. Send one in the 'X-LLM-API-Key' header "
            f"or configure a server-side key for provider '{provider}'."
        ),
    )


def get_llm(provider: str, model: str | None, api_key: str) -> BaseChatModel:
    """Build a LangChain chat model with `api_key` injected explicitly (BYOK).

    Imports are local so tests and startup don't pay for every provider's SDK.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider: '{provider}'. Supported: {list(SUPPORTED_PROVIDERS)}"
        )
    model = model or DEFAULT_MODELS[provider]

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=api_key)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=api_key)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, api_key=api_key)

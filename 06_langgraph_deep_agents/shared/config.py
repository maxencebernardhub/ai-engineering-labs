import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv(Path(__file__).parent.parent.parent / ".env")

SUPPORTED_PROVIDERS = ["anthropic", "openai", "google"]

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-5.4",
    "google": "gemini-3-flash-preview",
}


def get_llm(provider: str | None = None, model: str | None = None) -> BaseChatModel:
    """Return a chat model for the given provider and model name.

    Falls back to env vars LLM_PROVIDER / LLM_MODEL, then defaults.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider: '{provider}'. Supported: {SUPPORTED_PROVIDERS}"
        )
    model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model)

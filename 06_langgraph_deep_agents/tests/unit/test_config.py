from unittest.mock import patch

import pytest


def test_get_llm_anthropic():
    from langchain_anthropic import ChatAnthropic

    from shared.config import get_llm

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        llm = get_llm("anthropic")
    assert isinstance(llm, ChatAnthropic)


def test_get_llm_openai():
    from langchain_openai import ChatOpenAI

    from shared.config import get_llm

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        llm = get_llm("openai")
    assert isinstance(llm, ChatOpenAI)


def test_get_llm_google():
    from langchain_google_genai import ChatGoogleGenerativeAI

    from shared.config import get_llm

    with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
        llm = get_llm("google")
    assert isinstance(llm, ChatGoogleGenerativeAI)


def test_get_llm_default_model_per_provider():
    from shared.config import DEFAULT_MODELS, SUPPORTED_PROVIDERS

    assert set(DEFAULT_MODELS.keys()) == set(SUPPORTED_PROVIDERS)
    assert DEFAULT_MODELS["anthropic"] == "claude-sonnet-4-6"
    assert DEFAULT_MODELS["openai"] == "gpt-5.4"
    assert DEFAULT_MODELS["google"] == "gemini-3-flash-preview"


def test_get_llm_unknown_provider_raises():
    from shared.config import get_llm

    with pytest.raises(ValueError, match="Unknown provider"):
        get_llm("mistral")

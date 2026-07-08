"""Pydantic request/response models — the API's typed boundary.

Requests are validated here before any agent runs: an unsupported `engine`,
`provider`, or a `model` that does not belong to the chosen `provider` fails
validation, which FastAPI turns into a `422`. Providers and their model lists
come from `app.config` (the single source of truth); the engine list is mirrored
locally so this module need not import the heavy agent-graph packages (a test
guards the two against drifting).

The response models mirror the runner's structured output (the Step 4 contract):
`reply`, `tool_calls`, `leads_touched`, `email_draft`, `usage`. A runner dict
validates straight into `InvokeResponse` unchanged.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import PROVIDER_MODELS, SUPPORTED_PROVIDERS

# Mirror of `app.agents.runner.ENGINES`, kept here so schema validation does not
# import langgraph/deepagents. `test_engines_mirror_runner` guards the copy.
ENGINES: tuple[str, ...] = ("langgraph", "deep_agents")


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


class Message(BaseModel):
    """One turn of the client-carried conversation history (stateless service)."""

    role: Literal["system", "user", "assistant"]
    content: str


class InvokeRequest(BaseModel):
    """Body of `POST /invoke` and `POST /invoke/stream`."""

    engine: str = Field(description=f"Agent engine, one of {list(ENGINES)}.")
    provider: str = Field(
        description=f"LLM provider, one of {list(SUPPORTED_PROVIDERS)}."
    )
    model: str | None = Field(
        default=None,
        description="Model id; omit to use the provider's default.",
    )
    messages: list[Message] = Field(
        min_length=1, description="Full conversation history, oldest first."
    )

    @field_validator("engine")
    @classmethod
    def _known_engine(cls, value: str) -> str:
        if value not in ENGINES:
            raise ValueError(f"Unknown engine '{value}'. Supported: {list(ENGINES)}")
        return value

    @field_validator("provider")
    @classmethod
    def _known_provider(cls, value: str) -> str:
        if value not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown provider '{value}'. Supported: {list(SUPPORTED_PROVIDERS)}"
            )
        return value

    @model_validator(mode="after")
    def _model_matches_provider(self) -> InvokeRequest:
        # `provider` is validated above; guard the lookup in case it failed.
        allowed = PROVIDER_MODELS.get(self.provider)
        if self.model is not None and allowed is not None and self.model not in allowed:
            raise ValueError(
                f"Model '{self.model}' is not available for provider "
                f"'{self.provider}'. Choose one of {allowed}."
            )
        return self


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #


class ToolCall(BaseModel):
    """A single tool invocation and its textual result (for the actions trace)."""

    name: str
    args: dict[str, Any]
    result: str | None = None


class EmailDraft(BaseModel):
    """A generated email draft, surfaced for review (never sent nor persisted)."""

    lead_id: str
    to: str
    subject: str
    body: str
    generated_at: str


class Usage(BaseModel):
    """Token accounting summed across the agent's LLM turns."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class InvokeResponse(BaseModel):
    """Structured result of a `POST /invoke` run (mirrors the runner dict)."""

    reply: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    leads_touched: list[str] = Field(default_factory=list)
    email_draft: EmailDraft | None = None
    usage: Usage


class ModelsResponse(BaseModel):
    """Payload of `GET /models`: what the frontend needs to build its selectors.

    `server_keys` tells the frontend, per provider, whether a server-side key is
    configured — so it knows whether the BYOK field is required for that choice.
    """

    providers: dict[str, list[str]]
    default_models: dict[str, str]
    server_keys: dict[str, bool]

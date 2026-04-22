# Feature: Multi-Provider LLM Client

## Feature Brief

**Goal**: Build a reusable Python module `llm_client/` that abstracts OpenAI, Anthropic, and
Gemini behind a common interface, with intelligent routing, automatic fallback, and persistent
cost tracking. This module will be reused in future labs.

**Users**: The lab developer, who will import `llm_client` in subsequent labs without caring
about which provider is called underneath.

**Acceptance criteria**:

- `generate(prompt, model, temperature, max_tokens, stream, schema)` always returns an
  `LLMResponse` object regardless of the provider
- `LLMResponse` always contains: `text`, `parsed`, `input_tokens`, `output_tokens`, `model`,
  `provider`, `cost_usd`, `latency_ms`
- `schema=PydanticModel` (optional) triggers native structured output for all 3 providers;
  `response.parsed` contains the validated Pydantic object
- The router applies hard filters first (model-specific context window, modality support,
  availability), then sorts by strategy: `"cheapest" | "fastest" | "most_capable"`
- Fallback logic:
  - `429 Rate limit` → immediate fallback to next provider
  - Timeout → 1 retry on same provider, then fallback
  - Any other exception → immediate fallback
  - Fallback order = re-run router excluding the failed provider
- `NoProviderAvailableError` raised when all providers are filtered out before any call
- `AllProvidersFailedError` raised when all providers have been tried and failed
- Cost tracker persists each call as a JSON line in a `.jsonl` file (in-memory + auto-flush)
- Unit tests pass with no API keys (fully mocked)
- Integration tests are marked `@pytest.mark.integration` and run against real APIs
- CI (GitHub Actions) runs unit tests only; README documents how to enable integration tests
  in CI via GitHub Secrets
- Demo notebook sends the same prompt to all 3 providers and displays a comparative table
  of results and costs

**Edge cases**:

- All providers eliminated by hard filters → `NoProviderAvailableError` with reason
- All providers fail during fallback chain → `AllProvidersFailedError` listing all failures
- Provider returns malformed JSON despite structured output request → `StructuredOutputParseError`
- Context window check is model-specific (not provider-level):
  - Claude Opus 4.6 / Sonnet 4.6: 1M tokens
  - GPT-5.4: 1.05M tokens
  - Gemini 3.1 Pro: 1M tokens
- `.jsonl` cost log file does not exist on first call → created automatically

**Dependencies**:

- `anthropic` (latest stable) — `client.messages.parse()` + `output_format=PydanticModel`
- `openai` (latest stable) — `client.responses.parse()` + `text_format=PydanticModel`
- `google-genai` (latest stable) — `client.models.generate_content()` +
  `config={"response_json_schema": PydanticModel.model_json_schema()}`
- `pydantic` — shared schema definition across all providers
- `python-dotenv` — API key loading from root `.env`
- `pytest` — unit + integration tests
- Optional (notebook): `ipykernel`, `jupyterlab`

**Constraints**:

- No LiteLLM — built from scratch for learning purposes
- Full type hints, compatible with mypy
- Python >= 3.13, managed with `uv`
- Two test levels: `tests/unit/` (mocked, always run) and `tests/integration/` (real API, opt-in)
- Ruff for linting/formatting (line length 88, rules: E, W, F, I, B, UP)
- Module must remain importable as a standalone package in future labs

**Structure**:

```text
04_multi_provider/
├── pyproject.toml
├── README.md
├── docs/
│   └── specs/
│       └── 2026-04-14-multi-provider-llm-client.md
├── llm_client/
│   ├── __init__.py
│   ├── base.py               # LLMResponse dataclass + BaseProvider abstract class
│   ├── openai_provider.py    # OpenAI implementation
│   ├── anthropic_provider.py # Anthropic implementation
│   ├── gemini_provider.py    # Gemini implementation
│   ├── router.py             # Hard filters + soft sort + fallback logic
│   └── cost_tracker.py       # In-memory log + .jsonl persistence
├── tests/
│   ├── conftest.py           # pytest mark registration
│   ├── unit/
│   │   ├── test_router.py
│   │   ├── test_fallback.py
│   │   └── test_cost_tracker.py
│   └── integration/
│       └── test_providers.py
└── demo.ipynb

# At repo root:
.github/workflows/ci.yml     # Runs unit tests only; documents secrets for integration tests
```

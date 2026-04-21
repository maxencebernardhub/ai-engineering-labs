# Project State — Multi-Provider LLM Client

## Current Status

**Phase**: 3 — Implementation COMPLETE. All 20 unit tests passing.
**Branch**: `feat/04-multi-provider-lab`
**Last updated**: 2026-04-21

---

## Feature Brief

See full spec: [`docs/specs/2026-04-14-multi-provider-llm-client.md`](docs/specs/2026-04-14-multi-provider-llm-client.md)

**Goal**: Build a reusable async Python module `llm_client/` abstracting OpenAI, Anthropic,
and Gemini behind a common interface, with intelligent routing, automatic fallback, and
persistent cost tracking.

---

## Implementation Plan

### Files to Create or Modify

| File | Description |
|---|---|
| `pyproject.toml` | Project config, dependencies, Ruff, pytest marks |
| `llm_client/base.py` | `LLMResponse`, `StreamingResponse`, `BaseProvider` ABC, custom exceptions |
| `llm_client/cost_tracker.py` | `CostTracker`: in-memory buffer + `.jsonl` auto-flush |
| `llm_client/openai_provider.py` | `OpenAIProvider`: `client.responses.parse()` + `text_format` |
| `llm_client/anthropic_provider.py` | `AnthropicProvider`: `client.messages.parse()` + `output_format` |
| `llm_client/gemini_provider.py` | `GeminiProvider`: `client.models.generate_content()` + `config` |
| `llm_client/router.py` | Hard filters + soft sort + async `generate_with_fallback()` |
| `llm_client/__init__.py` | Public API surface |
| `tests/conftest.py` | `integration` mark registration + async fixtures |
| `tests/unit/test_cost_tracker.py` | Unit tests — cost tracker |
| `tests/unit/test_router.py` | Unit tests — router |
| `tests/unit/test_fallback.py` | Unit tests — fallback logic |
| `tests/integration/test_providers.py` | Integration tests against real APIs |
| `(repo root) .github/workflows/ci.yml` | CI: unit tests only |
| `demo.ipynb` | Comparative notebook — 3 providers, same prompt |

---

### Test Cases

**`test_cost_tracker.py`**
- `test_log_entry_has_all_fields`
- `test_multiple_calls_accumulate_in_memory`
- `test_flush_writes_jsonl_file`
- `test_jsonl_created_if_not_exists`
- `test_total_cost_sums_correctly`

**`test_router.py`**
- `test_hard_filter_excludes_model_with_insufficient_context`
- `test_all_providers_filtered_raises_no_provider_available`
- `test_strategy_cheapest_selects_lowest_cost_provider`
- `test_strategy_fastest_selects_lowest_latency_provider`
- `test_strategy_most_capable_selects_highest_tier_provider`
- `test_exclude_parameter_removes_provider_from_selection`

**`test_fallback.py`**
- `test_rate_limit_error_triggers_immediate_fallback`
- `test_timeout_retries_once_before_fallback`
- `test_generic_exception_triggers_immediate_fallback`
- `test_all_providers_fail_raises_all_providers_failed`
- `test_successful_fallback_returns_llm_response`
- `test_streaming_fallback_on_rate_limit`
- `test_streaming_fallback_on_timeout`

**`test_providers.py`** (integration, `@pytest.mark.integration`)
- `test_openai_generate_returns_llm_response`
- `test_anthropic_generate_returns_llm_response`
- `test_gemini_generate_returns_llm_response`
- `test_openai_structured_output_returns_parsed_model`
- `test_anthropic_structured_output_returns_parsed_model`
- `test_gemini_structured_output_returns_parsed_model`
- `test_openai_stream_yields_chunks`
- `test_anthropic_stream_yields_chunks`
- `test_gemini_stream_yields_chunks`
- `test_openai_stream_with_schema_returns_parsed_final_response`
- `test_anthropic_stream_with_schema_returns_parsed_final_response`
- `test_gemini_stream_with_schema_returns_parsed_final_response`

---

### Steps

| # | Step | Status |
|---|---|---|
| 1 | Project setup: `pyproject.toml` + directory structure + `uv sync` | `done` |
| 2 | `base.py`: `LLMResponse`, `StreamingResponse`, `BaseProvider`, exceptions | `done` |
| 3 | `cost_tracker.py` + `test_cost_tracker.py` (TDD) | `done` |
| 4 | `openai_provider.py` | `done` |
| 5 | `anthropic_provider.py` | `done` |
| 6 | `gemini_provider.py` | `done` |
| 7 | `router.py` + `test_router.py` + `test_fallback.py` (TDD) | `done` |
| 8 | `__init__.py` | `done` |
| 9 | `conftest.py` + `test_providers.py` (integration) | `done` |
| 10 | `.github/workflows/ci.yml` | `done` |
| 11 | `demo.ipynb` | `done` |

---

## Key Design Decisions

- **Async-first**: all `generate()` methods are `async def`
- **`stream=True`** returns a `StreamingResponse` (custom `AsyncIterator[str]`); after
  exhausting the iterator, `.final_response` exposes the full `LLMResponse` with `.parsed`
- **Router**: cascade — hard filters first (model-specific context window, modality,
  availability), then soft sort by strategy (`cheapest` | `fastest` | `most_capable`)
- **Fallback**: re-runs router excluding failed provider; `429` → immediate fallback;
  timeout → 1 retry then fallback; all others → immediate fallback
- **Cost tracker**: in-memory accumulation + auto-flush to `.jsonl` after each call
- **Token estimation**: `len(prompt) / 4` for context window hard filter (known limitation,
  documented in README)
- **No LiteLLM**: built from scratch for learning purposes

## Known Context Window Limits (model-specific)

| Model | Context window |
|---|---|
| Claude Opus 4.6 / Sonnet 4.6 | 1M tokens |
| GPT-5.4 | 1.05M tokens |
| Gemini 3.1 Pro | 1M tokens |

---

## Risks & Open Questions

- Exact syntax for OpenAI `client.responses.parse()` and Gemini
  `config={"response_json_schema": ...}` to be verified against live docs at implementation time
- `pytest-asyncio` required for all async tests

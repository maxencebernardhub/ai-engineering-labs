# Project State — Multi-Provider LLM Client

## Current Status

**Phase**: Implementation complete — 20 unit tests + 12 integration tests passing.
**Branch**: `feat/04-multi-provider-lab`
**Last updated**: 2026-04-22

---

## Feature Brief

See full spec:
[`docs/specs/2026-04-14-multi-provider-llm-client.md`](docs/specs/2026-04-14-multi-provider-llm-client.md)

**Goal**: Build a reusable async Python module `llm_client/` abstracting OpenAI, Anthropic,
and Gemini behind a common interface, with intelligent routing, automatic fallback, and
persistent cost tracking.

---

## Implementation Plan

### Files Created or Modified

| File | Description |
| --- | --- |
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
| --- | --- | --- |
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
- **Gemini tiered pricing**: `_cost()` applies ≤200k / >200k tier based on `input_tokens`
- **Gemini model aliases**: friendly names (e.g. `gemini-3.1-pro`) resolve to API preview
  identifiers via `_MODEL_ALIASES` in `gemini_provider.py`
- **Token estimation**: `len(prompt) // 4` for context window hard filter (known limitation,
  documented in README)
- **No LiteLLM**: built from scratch for learning purposes

---

## Supported Models

| Model alias | Provider | Context window | Tier | Input $/MTok | Output $/MTok |
| --- | --- | --- | --- | --- | --- |
| `gpt-5.4` | openai | 1 050 000 | 3 | 2.50 | 15.00 |
| `gpt-5.4-mini` | openai | 400 000 | 2 | 0.75 | 4.50 |
| `gpt-5.4-nano` | openai | 400 000 | 1 | 0.20 | 1.25 |
| `claude-opus-4-7` | anthropic | 1 000 000 | 3 | 5.00 | 25.00 |
| `claude-opus-4-6` | anthropic | 1 000 000 | 3 | 5.00 | 25.00 |
| `claude-sonnet-4-6` | anthropic | 1 000 000 | 2 | 3.00 | 15.00 |
| `claude-sonnet-4-5` | anthropic | 200 000 | 2 | 3.00 | 15.00 |
| `claude-haiku-4-5` | anthropic | 200 000 | 1 | 1.00 | 5.00 |
| `gemini-3.1-pro` | gemini | 1 000 000 | 3 | 2.00 | 12.00 |
| `gemini-2.5-pro` | gemini | 1 000 000 | 3 | 1.25 | 10.00 |
| `gemini-3-flash` | gemini | 1 000 000 | 2 | 0.50 | 3.00 |
| `gemini-2.5-flash` | gemini | 1 000 000 | 2 | 0.30 | 2.50 |
| `gemini-3.1-flash-lite` | gemini | 1 000 000 | 1 | 0.25 | 1.50 |

> Gemini input/output prices shown are for prompts ≤ 200k tokens.
> `gemini-3.1-pro` and `gemini-2.5-pro` apply higher rates above that threshold.

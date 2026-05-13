# Project State — Local Models & Privacy-First AI

## Current Status

**Phase**: Implementation — Step 8 in progress
**Branch**: `feat/07-local-models-privacy-first`
**Last updated**: 2026-05-13

---

## Feature Brief

See full spec:
[`docs/specs/2026-05-12-07-local-models-privacy-first.md`](docs/specs/2026-05-12-07-local-models-privacy-first.md)

**Goal**: Build Lab 07 — a standalone, production-aware lab demonstrating privacy-first AI
with local models via Ollama. The lab proves that an AI engineer can build capable systems
(structured outputs, tool calling, vision, RAG) without sending any data to the cloud, and
quantifies the tradeoffs vs cloud providers through a rigorous 4-dimension benchmark.

---

## Planned Structure

```text
07_local_models_privacy_first/
├── README.md
├── pyproject.toml
├── .python-version
├── PROJECT_STATE.md
├── docs/specs/2026-05-12-07-local-models-privacy-first.md
├── data/
│   ├── contoso_report_q4_2024.md        # Fake confidential HR/financial report
│   ├── generate_vision_assets.py         # Script to generate vision images with matplotlib
│   └── vision/
│       ├── invoice_sample.jpg            # Fake vendor invoice
│       ├── org_chart.png                 # Fake org chart
│       └── dashboard_screenshot.png      # Fake HR dashboard screenshot
├── utils/
│   └── helpers.py                        # check_ollama_running(), check_model_available(), chunk_text()
├── app/
│   ├── app.py                            # Streamlit entry point
│   └── chat.py                           # stream_response() and get_stats()
├── tests/
│   ├── unit/
│   │   ├── test_helpers.py               # Unit tests for chunk_text()
│   │   └── test_chat.py                  # Unit tests for get_stats()
│   └── integration/
│       └── test_ollama.py                # Ollama connectivity + model availability
├── 01_ollama_setup.ipynb
├── 02_advanced_capabilities.ipynb
├── 03_tool_calling.ipynb
├── 04_vision.ipynb
├── 05_local_rag.ipynb
└── 06_benchmark.ipynb
```

---

## Models

### Local (via Ollama)

| Model | Size | Role |
| --- | --- | --- |
| `gemma4:e4b` | 9.6 GB | Primary model — vision support (notebook 04) |
| `qwen3.5:9b` | ~6 GB Q4_K_M | Reasoning and tool calling |
| `mistral:7b` | ~4 GB | Lightweight reference baseline |
| `qwen3-embedding:0.6b` | 639 MB | Embeddings for notebook 05 (RAG) |

### Cloud (notebook 06 only, via LiteLLM)

| Model | Provider |
| --- | --- |
| `claude-sonnet-4-6` | Anthropic |
| `gpt-5.4` | OpenAI |
| `gemini/gemini-3.1-flash-lite` | Google |

---

## Key Design Decisions

- **LiteLLM**: single calling convention for all 6 models (local + cloud) in notebook 06 — avoids 3 native SDKs
- **FAISS**: lower-level alternative to ChromaDB (already used in lab 05) — no server, pure Python, full control
- **qwen3-embedding:0.6b**: MTEB #1 multilingual, 32K context (vs 2K for nomic-embed-text), same Qwen family as qwen3.5:9b
- **No LangChain**: lab 05 covers it in depth — lab 07 targets minimal deps, full offline, maximum control
- **matplotlib for vision assets**: programmatic generation ensures known ground-truth values for notebook 04 test cases
- **utils/helpers.py**: extracts `chunk_text()`, `check_ollama_running()`, `check_model_available()` for testability

---

## Implementation Steps

| # | Step | Status |
| --- | --- | --- |
| 1 | Project scaffolding: `pyproject.toml`, `.python-version`, `uv sync` | `done` |
| 2 | Synthetic data: `contoso_report_q4_2024.md` + generate 3 vision assets | `done` |
| 3 | TDD — `utils/helpers.py` + `app/chat.py` + all unit and integration tests | `done` |
| 4 | `01_ollama_setup.ipynb` — Ollama API, model management, streaming | `done` |
| 5 | `02_advanced_capabilities.ipynb` — Pydantic structured outputs + streaming | `done` |
| 6 | `03_tool_calling.ipynb` — 3 tools × 3 scenarios × 3 local models | `done` |
| 7 | `04_vision.ipynb` — multimodal with gemma4:e4b, all 3 vision assets | `done` |
| 8 | `05_local_rag.ipynb` — full offline RAG: FAISS + qwen3-embedding + Ollama LLM | `in progress` |
| 9 | `06_benchmark.ipynb` — 6-model benchmark via LiteLLM, DataFrame + visualizations | `pending` |
| 10 | Streamlit app: `app/app.py` + `app/chat.py` | `pending` |
| 11 | `README.md` — architecture diagram, setup instructions, screenshots | `pending` |
| 12 | Root `README.md` — update lab 07 row to `✅ Available` | `pending` |

---

## Risks

| Risk | Mitigation |
| --- | --- |
| LiteLLM model name format for local Ollama | Verify `ollama/gemma4:e4b` format before step 9 |
| Ollama embeddings API call signature for `qwen3-embedding:0.6b` | Test `ollama.embeddings()` vs raw REST at step 3 |
| Passing images to gemma4:e4b via Ollama SDK | Verify base64 format expected before notebook 04 |

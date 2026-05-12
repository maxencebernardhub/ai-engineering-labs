# Feature: Lab 07 — Local Models & Privacy-First AI

## Feature Brief

**Goal**: Build Lab 07 — a standalone, production-aware lab demonstrating privacy-first AI
with local models via Ollama. The lab proves that an AI engineer can build capable systems
(structured outputs, tool calling, vision, RAG) without sending any data to the cloud, and
quantifies the tradeoffs vs cloud providers through a rigorous 4-dimension benchmark.

**Users**:

- Technical recruiters, CTOs, and prospects evaluating the author's AI engineering profile
- Developers learning local AI deployment
- Consultants advising clients on privacy-preserving AI architecture

---

## Synthetic Data

### Confidential Company Report — `data/contoso_report_q4_2024.md`

A fictitious English-language HR/financial report for **Contoso Corp**, a fictional
UK-based B2B SaaS company (~200 employees). The document contains:

- Headcount by department and seniority level
- Salary bands and compensation ranges
- Attrition rate and reasons for departure (Q4 2024)
- Performance review results (rating distribution)
- Hiring plan for Q1 2025

This document simulates exactly the kind of data a company would never send to a
third-party cloud API — making it the ideal anchor for the "privacy-first" narrative.

### Vision Assets — `data/vision/`

Three images generated to support notebook 04:

| File | Type | Content |
| --- | --- | --- |
| `invoice_sample.jpg` | Fake invoice | Contoso Corp vendor invoice with line items, VAT, totals |
| `org_chart.png` | Org chart | Contoso Corp engineering department org chart |
| `dashboard_screenshot.png` | UI screenshot | Fictional HR dashboard with KPIs (headcount, attrition) |

---

## Models

### Local models (via Ollama)

| Model | Size | Role |
| --- | --- | --- |
| `gemma4:e4b` | 9.6 GB | Primary model — supports vision (notebook 04) |
| `qwen3.5:9b` | ~6 GB (Q4_K_M) | Reasoning and tool calling |
| `mistral:7b` | ~4 GB | Lightweight reference, widely used baseline |
| `qwen3-embedding:0.6b` | 639 MB | Embedding model for notebook 05 (RAG) |

All models must fit within 24 GB of unified/GPU memory. Users with less memory should
select smaller variants from the Ollama library. Model names are configurable at the
top of each notebook — no hardcoded IDs in business logic.

### Cloud models (notebook 06 only, via LiteLLM)

| Model | Provider |
| --- | --- |
| `claude-sonnet-4-6` | Anthropic |
| `gpt-5.4` | OpenAI |
| `gemini/gemini-3.1-flash-lite` | Google |

---

## Notebooks

### `01_ollama_setup.ipynb`

- Install check and Ollama version
- Pull and list models programmatically
- Basic inference with the `ollama` Python SDK
- Ollama's OpenAI-compatible REST endpoint (`http://localhost:11434/v1`)
- Model parameters: temperature, context length, system prompt
- Streaming output

### `02_advanced_capabilities.ipynb`

- Structured outputs with Pydantic: define a schema, extract structured data from an
  unstructured document (the Contoso report)
- Validation: what happens when the model returns malformed JSON
- Streaming: token-by-token output with `ollama.chat(stream=True)`
- Comparison: which local models comply with structured output constraints reliably

### `03_tool_calling.ipynb`

Tools defined:

| Tool | Description |
| --- | --- |
| `get_headcount(department)` | Returns headcount from the Contoso report for a given department |
| `get_attrition_rate(quarter)` | Returns attrition % for a given quarter |
| `calculate_salary_budget(department, raise_pct)` | Computes new budget given a raise % |

Scenarios tested on all 3 local models:

1. Single tool call
2. Sequential tool calls (two tools, one after the other)
3. Model refuses tool call (out-of-scope question)

Output: comparison table — tool call success rate per model per scenario.

### `04_vision.ipynb`

- Load each image from `data/vision/`
- Ask structured questions to `gemma4:e4b`:
  - Invoice: extract vendor name, total amount, VAT
  - Org chart: list direct reports of the CTO
  - Dashboard screenshot: read the attrition KPI value
- Demonstrate graceful error handling when a non-vision model is passed

### `05_local_rag.ipynb`

Full offline RAG pipeline — zero cloud calls, no LangChain:

1. **Document loading**: read `data/contoso_report_q4_2024.md`, split into chunks
   (chunk size: 512 tokens, overlap: 64 tokens)
2. **Embedding**: `ollama.embeddings(model="qwen3-embedding:0.6b", prompt=chunk)`
3. **Indexing**: store vectors in a FAISS `IndexFlatL2` index (no server, pure Python)
4. **Retrieval**: cosine similarity search, top-k=5
5. **Generation**: `ollama.chat()` with retrieved context injected in system prompt
6. **Test questions** (3–5 questions on the Contoso report, answers verifiable from the doc)

### `06_benchmark.ipynb`

Same 5 questions from notebook 05 sent to all 6 models via LiteLLM.
Results collected in a `pandas` DataFrame and visualized with `matplotlib`.

**Benchmark dimensions:**

| Dimension | Metrics |
| --- | --- |
| **Performance** | TTFT (time-to-first-token), tokens/sec, max context window |
| **Quality** | Factual accuracy (manual 1–5), instruction following, structured output compliance (pass/fail), tool calling success rate |
| **Economics** | Cost per 1M tokens (cloud: official pricing; local: $0 + ~0.01 kWh estimate), rate limits |
| **Data Sovereignty** | Data leaves the machine (yes/no), data residency guarantees, applicability to regulated industries (healthcare, finance, legal), model license (open vs proprietary), fine-tuning capability |

---

## Streamlit App — `app/`

A local chat interface demonstrating that the local setup is production-viable.

**`app/app.py`** — entry point:

- Sidebar: model selector (gemma4:e4b / qwen3.5:9b / mistral:7b)
- Main area: chat interface (message history + input box)
- Stats panel (updated per response):
  - Tokens/sec
  - Response latency (ms)
  - Data residency indicator: `✅ Data stays on your machine`

**`app/chat.py`** — chat logic:

- `stream_response(model, messages)` → generator yielding tokens
- `get_stats(start_time, token_count)` → returns latency and throughput

---

## Acceptance Criteria

- 6 notebooks each executable end-to-end without errors
- Notebook 01 passes an Ollama connectivity check before any other cell runs
- Notebook 03 produces a comparison table for all 3 local models × 3 scenarios
- Notebook 04 successfully extracts structured data from all 3 vision assets
- Notebook 05 answers all 5 test questions with correct answers verifiable from the document
- Notebook 06 produces a complete results DataFrame and at least 2 visualizations
- Streamlit app runs with `uv run streamlit run app/app.py` and displays live token stats
- README includes architecture diagram, setup instructions (including `ollama pull` commands),
  and screenshots of the app and benchmark results

---

## Edge Cases

- Model not pulled: notebooks detect missing model via `ollama.list()` and print a clear
  error with the exact `ollama pull <model>` command to run
- Vision not supported: notebook 04 wraps the call in try/except and prints which models
  support vision
- Cloud API key missing: notebook 06 skips that provider's models gracefully, logs a
  warning, and continues with available models
- Ollama not running: all notebooks catch `ConnectionRefusedError` and print
  `ollama serve` instructions
- FAISS index empty: notebook 05 guards against querying before indexing

---

## Key Design Decisions

### LiteLLM over native provider SDKs (notebook 06)

**Chosen**: `litellm.completion()` for all 6 models

**Rationale**: A single calling convention for all models — local (via Ollama) and cloud —
keeps the benchmark code uniform and the comparison honest. Installing three native SDKs
(`anthropic`, `openai`, `google-generativeai`) would add dependencies without pedagogical
benefit. LiteLLM is a production-grade library used for exactly this purpose.

**Trade-off**: One additional abstraction layer. Debuggability is slightly lower than
native SDKs, but acceptable for a benchmark context.

### FAISS over ChromaDB (notebook 05)

**Chosen**: `faiss-cpu` with `IndexFlatL2`, no server

**Rationale**: ChromaDB is already covered in lab 05. FAISS is the lower-level alternative —
no daemon, no HTTP server, pure Python library. Using it here demonstrates understanding of
the vector search layer beneath the framework abstractions. Coherent with the "minimal
dependencies, full control" philosophy of a privacy-first lab.

**Trade-off**: FAISS requires more boilerplate than ChromaDB (manual serialization,
no built-in metadata filtering). This is intentional: the extra code is pedagogically valuable.

### qwen3-embedding:0.6b over nomic-embed-text

**Chosen**: `qwen3-embedding:0.6b` via Ollama

**Rationale**: nomic-embed-text (2 years old, 2K context) is no longer state-of-the-art.
qwen3-embedding:0.6b ranks #1 on the MTEB multilingual leaderboard, supports 32K context
(vs 2K), and is from the same model family as qwen3.5:9b — creating internal coherence.
At 639MB it adds negligible memory overhead alongside the LLM.

### No LangChain

**Rationale**: Lab 05 already demonstrates LangChain at depth. Lab 07 targets a different
point in the design space: minimal dependencies, full offline, maximum control. Using raw
`faiss-cpu` + `ollama` SDK makes the pipeline more transparent and differentiates the two labs.

---

## Dependencies

```toml
[project]
requires-python = ">=3.13"

[project.dependencies]
ollama = "*"
faiss-cpu = "*"
pydantic = "*"
streamlit = "*"
litellm = "*"
pandas = "*"
matplotlib = "*"
python-dotenv = "*"

[project.optional-dependencies]
notebook = ["jupyterlab", "ipywidgets"]
```

Environment variables — consumed from root `.env`:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`

---

## Project Structure

```text
07_local_models_privacy_first/
├── README.md
├── pyproject.toml
├── .python-version
├── .gitignore
├── docs/
│   └── specs/
│       └── 2026-05-12-07-local-models-privacy-first.md
├── data/
│   ├── contoso_report_q4_2024.md        # Fake confidential HR/financial report
│   └── vision/
│       ├── invoice_sample.jpg            # Fake vendor invoice
│       ├── org_chart.png                 # Fake org chart
│       └── dashboard_screenshot.png      # Fake HR dashboard screenshot
├── app/
│   ├── app.py                            # Streamlit entry point
│   └── chat.py                           # Streaming chat logic + stats
├── 01_ollama_setup.ipynb
├── 02_advanced_capabilities.ipynb
├── 03_tool_calling.ipynb
├── 04_vision.ipynb
├── 05_local_rag.ipynb
└── 06_benchmark.ipynb
```

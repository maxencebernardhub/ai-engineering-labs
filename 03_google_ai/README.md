# Google Gemini API Labs

This directory contains a curated set of educational labs for learning the
Google Gemini API with Python using the latest `google-genai` SDK. The goal of the
collection is to stay practical, readable, and progressively structured: each
notebook focuses on one main idea, while the reusable helper module reduces
boilerplate across all of them.

The labs are intentionally hands-on. They cover the latest **Gemini 3.1**
features, including industry-leading long context (up to 2M tokens), native
multimodality (Images, Audio, Video), structured JSON outputs, function calling,
the Multimodal Live API, and advanced optimization techniques like context caching.

## What This Directory Contains

- `notebooks/`: 15 Jupyter notebooks organized by topic, from API basics to
  advanced production features.
- `scripts/`: reusable helper module (`gemini_client.py`) and its test suite.

## Prerequisites

Before running the Gemini labs, make sure you have:

- Python `3.13` or newer
- a Google AI (Gemini) API key
- a single `.env` file at the repository root

Example root `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## Python Dependencies

This directory contains its own `pyproject.toml` so users can install only the
dependencies needed for the Google AI labs instead of installing the entire
repository stack.

Base dependencies required for the scripts and core logic:

- `google-genai` (v1.68+)
- `pydantic`
- `python-dotenv`
- `numpy`
- `httpx`

Optional notebook dependencies:

- `jupyterlab`
- `ipykernel`

## Installation

### Option 1: Install With `uv` (Recommended)

If you want only the Python scripts and base SDK:

```bash
cd 03_google_ai
uv sync
```

If you also want to run the Jupyter notebooks:

```bash
cd 03_google_ai
uv sync --extra notebooks
```

To launch JupyterLab:

```bash
cd 03_google_ai
uv run --extra notebooks jupyter lab
```

### Option 2: Install Without `uv`

Create and activate a virtual environment first:

```bash
cd 03_google_ai
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies for the labs:

```bash
pip install .
```

Install the notebook dependencies as well:

```bash
pip install ".[notebooks]"
```

Then launch JupyterLab:

```bash
jupyter lab
```

## Running The Labs

### Running The Jupyter Notebooks

All main labs live in `notebooks/`. Start JupyterLab from inside `03_google_ai`:

```bash
uv run --extra notebooks jupyter lab
```

Begin with the quickstart notebook for a compact overview, then move through the
numbered notebooks in order.

### Suggested Starting Point

```bash
# Quick overview of all main SDK patterns
notebooks/00_quickstart_gemini.ipynb

# Then follow the numbered sequence:
notebooks/01_model_configuration_and_safety.ipynb
notebooks/02_chat_and_streaming.ipynb
...
```

## Directory Structure

```text
03_google_ai/
├── notebooks/
│   ├── 00_quickstart_gemini.ipynb
│   ├── 01_model_configuration_and_safety.ipynb
│   ├── 02_chat_and_streaming.ipynb
│   ├── 03_file_api_and_long_context.ipynb
│   ├── 04_multimodality_deep_dive.ipynb
│   ├── 05_structured_outputs_and_reliability.ipynb
│   ├── 06_function_calling_and_tools.ipynb
│   ├── 07_code_execution_tool.ipynb
│   ├── 08_prompt_evaluation_and_grading.ipynb
│   ├── 09_model_comparison_benchmarks.ipynb
│   ├── 10_multimodal_live_interactions.ipynb
│   ├── 11_advanced_optimization.ipynb
│   ├── 12_token_management_and_costs.ipynb
│   ├── 13_batch_api_processing.ipynb
│   └── 14_controlled_generation.ipynb
├── scripts/
│   ├── gemini_client.py
│   └── test_gemini_client.py
└── README.md
```

## Labs Overview

### Foundational API Usage

- `00_quickstart_gemini.ipynb`
  Introduces the `google-genai` SDK: client setup, model listing, basic
  generation, and asynchronous calls.

- `01_model_configuration_and_safety.ipynb`
  Covers parameters like `temperature`, `top_k`, `seed`, and the mandatory
  Safety Settings thresholds.

- `02_chat_and_streaming.ipynb`
  Stateful multi-turn conversations using `client.chats`, manual history
  manipulation, and robust streaming.

### Long Context and Multimodality

- `03_file_api_and_long_context.ipynb`
  Uploading and managing documents via the **File API** to leverage the 2M
  token window for multi-document reasoning.

- `04_multimodality_deep_dive.ipynb`
  Advanced Image Comparison, Audio summarization, and Video temporal reasoning
  (extracting timestamps).

### Agents and Automation

- `05_structured_outputs_and_reliability.ipynb`
  Strict JSON enforcement using **Pydantic schemas** and Enums for 100%
  reliable extraction.

- `06_function_calling_and_tools.ipynb`
  Comparing **Automatic Function Calling (AFC)** vs. **Manual Handshakes**
  for custom tools.

- `07_code_execution_tool.ipynb`
  Using Gemini's built-in Python interpreter to solve complex math and
  perform data analysis on raw text.

### Production and Optimization

- `08_prompt_evaluation_and_grading.ipynb`
  Building a **Model-as-a-Judge** pipeline to evaluate and rank model
  outputs based on custom rubrics.

- `09_model_comparison_benchmarks.ipynb`
  Interactive benchmarking of TTFT (latency), reasoning, and cost across
  Flash-Lite and Pro models.

- `10_multimodal_live_interactions.ipynb`
  Working with the **Multimodal Live API** (WebSocket) for low-latency,
  real-time interactions with audio and tool calling.

- `11_advanced_optimization.ipynb`
  Techniques for scale: **Context Caching** (Text/Media), Semantic Embeddings,
  and **Google Search Grounding**.

### Financial and High-Throughput Ops

- `12_token_management_and_costs.ipynb`
  Auditing requests with `count_tokens` and implementing dynamic cost
  estimation and retry logic.

- `13_batch_api_processing.ipynb`
  Handling massive datasets with the **Batch API** for a 50% cost reduction
  and higher throughput.

- `14_controlled_generation.ipynb`
  Expert techniques: Few-Shot Content pairs, **Thinking Mode** (internal
  reasoning), and **Response Hinting** (prefilling).

## Reusable Helper Module

The file `scripts/gemini_client.py` contains a small wrapper around recurring
Gemini SDK patterns. It is intentionally lightweight and educational rather
than abstract:

- `get_client()`: creation of the GenAI client.
- `generate_text()`: simple instruction-based generation.
- `generate_structured()`: Pydantic-validated parsing.
- `stream_response()`: clean iteration over text chunks.
- `generate_embeddings()`: single and batch vector generation.

## Suggested Learning Path

If you are new to Gemini, follow this progression:

1. `00_quickstart_gemini.ipynb` — The basics.
2. `02_chat_and_streaming.ipynb` — Building a chat UI.
3. `05_structured_outputs_and_reliability.ipynb` — Connecting to APIs.
4. `06_function_calling_and_tools.ipynb` — Building an agent.
5. `03_file_api_and_long_context.ipynb` — Handling large data.
6. `10_multimodal_live_interactions.ipynb` — Real-time audio agents.
7. `12_token_management_and_costs.ipynb` — Ready for production.

## Philosophy Of The Labs

These labs follow the principles of the entire repository:

- **Minimal abstraction**: We stay close to the official SDK to ensure
  knowledge is transferable.
- **Progressive complexity**: We move from "Hello World" to "Real-time
  Multimodal Audio Agent".
- **Production-ready**: We cover real-world concerns like error handling,
  cost management, and evaluation.

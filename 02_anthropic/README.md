# Anthropic API Labs

This directory contains a curated set of educational labs for learning the
Anthropic API with Python. The goal of the collection is to stay practical,
readable, and progressively structured: each notebook focuses on one main idea
while the quick-reference companion and the reusable helper module reduce
boilerplate across all of them.

The labs are intentionally hands-on. They cover core messaging patterns, prompt
engineering and evaluation, tool use and agentic loops, retrieval-augmented
generation, advanced features such as thinking mode, prompt caching, citations,
vision, PDF processing, and code execution, as well as a standalone CLI project
built on the MCP architecture.

## What This Directory Contains

- `notebooks/`: Jupyter notebooks organised by topic, from API basics to
  advanced features
- `scripts/`: reusable helper module (`anthropic_client.py`) shared across
  notebooks and quick experiments
- `cli_project/`: standalone MCP Chat CLI application built on the MCP
  architecture, demonstrating real-world tool use and document retrieval
- `inputs/`: sample files used by the labs (images, PDFs, etc.)
- `outputs/`: files produced by the labs at runtime

## Prerequisites

Before running the Anthropic labs, make sure you have:

- Python `3.13.12` or newer
- an Anthropic API key
- a VoyageAI API key (required for the embedding and RAG notebooks)
- a single `.env` file at the repository root

Example root `.env` file:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
VOYAGE_API_KEY=your_voyageai_api_key_here
```

## Python Dependencies

This directory contains its own `pyproject.toml` so users can install only the
dependencies needed for the Anthropic labs instead of installing the entire
repository stack.

Base dependencies required for the notebooks:

- `anthropic`
- `python-dotenv`
- `voyageai`
- `pydantic`

Optional notebook dependencies:

- `jupyterlab`
- `ipykernel`

## Installation

### Option 1: Install With `uv` (Recommended)

If you want only the Python scripts:

```bash
cd 02_anthropic
uv sync
```

If you also want to run the Jupyter notebooks:

```bash
cd 02_anthropic
uv sync --extra notebooks
```

To launch JupyterLab:

```bash
cd 02_anthropic
uv run --extra notebooks jupyter lab
```

### Option 2: Install Without `uv`

Create and activate a virtual environment first:

```bash
cd 02_anthropic
python3.13 -m venv .venv
source .venv/bin/activate
```

Install the dependencies for the scripts:

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

All labs live in `notebooks/`. Start JupyterLab from inside `02_anthropic`:

```bash
uv run --extra notebooks jupyter lab
```

Begin with the quick-reference notebook for a compact overview of the SDK,
then move through the numbered notebooks in order.

### Suggested Starting Point

```bash
# Quick overview of all main SDK patterns
notebooks/anthropic_quick_reference.ipynb

# Then follow the numbered sequence:
notebooks/00_notebook_basic_API.ipynb
notebooks/01_prompt_evals.ipynb
...
```

## Directory Structure

```text
02_anthropic/
├── notebooks/
│   ├── anthropic_quick_reference.ipynb
│   ├── 00_notebook_basic_API.ipynb
│   ├── 01_prompt_evals.ipynb
│   ├── 01_prompt_evals_fns.ipynb
│   ├── 01_prompt_evals_grader.ipynb
│   ├── 02_prompting.ipynb
│   ├── 03_text_editor_tool.ipynb
│   ├── 03_tool_streaming.ipynb
│   ├── 03_tools.ipynb
│   ├── 03_tools_multi_turns.ipynb
│   ├── 03_tools_using_multiple_tools.ipynb
│   ├── 03_web_search.ipynb
│   ├── 04_bm25.ipynb
│   ├── 04_chunking.ipynb
│   ├── 04_embeddings.ipynb
│   ├── 04_hybrid.ipynb
│   ├── 04_vectordb.ipynb
│   ├── 05_caching.ipynb
│   ├── 05_citations.ipynb
│   ├── 05_code_execution.ipynb
│   ├── 05_images.ipynb
│   ├── 05_pdf.ipynb
│   └── 05_thinking.ipynb
├── scripts/
│   └── anthropic_client.py
├── cli_project/
│   ├── core/
│   ├── main.py
│   ├── mcp_client.py
│   ├── mcp_server.py
│   ├── pyproject.toml
│   └── README.md
├── inputs/
├── outputs/
└── README.md
```

## Notebooks Overview

### Quick Reference

- `anthropic_quick_reference.ipynb`
  A compact notebook for everyday Anthropic SDK usage. Covers plain text
  generation, structured outputs via tool use, streaming, embeddings via
  VoyageAI, and multi-turn message inputs.

### API Basics

- `00_notebook_basic_API.ipynb`
  Introduces the SDK: client setup, the `messages.create()` call, model
  selection, and key response parameters.

### Prompt Engineering and Evaluation

- `01_prompt_evals.ipynb`
  Framework for evaluating prompts end-to-end: dataset generation, prompt
  execution, and LLM-based scoring with structured feedback.

- `01_prompt_evals_fns.ipynb`
  Extends the evaluation framework with specialised grading functions,
  including syntactic validators (JSON, Python, regex).

- `01_prompt_evals_grader.ipynb`
  Variant focused on detailed grading with explicit solution criteria,
  structured reasoning, and an HTML report builder.

- `02_prompting.ipynb`
  Prompt engineering techniques: few-shot examples, output format control,
  system prompts, and temperature tuning, evaluated with the framework above.

### Tool Use and Agentic Loops

- `03_tools.ipynb`
  Introduction to tool use: defining tool schemas, sending tool calls, and
  handling `tool_use` responses from the model.

- `03_tools_multi_turns.ipynb`
  Multi-turn agentic loop: iterating until the model stops requesting tools,
  with proper `tool_result` message handling.

- `03_tools_using_multiple_tools.ipynb`
  Running several tools in the same conversation, including parallel tool
  dispatch and result aggregation.

- `03_tool_streaming.ipynb`
  Streaming tool calls: consuming `input_json_delta` events in real time,
  forcing specific tools with `tool_choice`, and handling structured streaming
  responses.

- `03_text_editor_tool.ipynb`
  The built-in `str_replace_based_edit_tool`: view, create, and patch files
  through Claude with path validation and backup support.

- `03_web_search.ipynb`
  The built-in web search tool: configuring `web_search_20250305`,
  restricting to allowed domains, and capping the number of searches.

### Retrieval-Augmented Generation (RAG)

- `04_chunking.ipynb`
  Text chunking strategies: fixed-size character windows, sentence-level
  splits, and section-based chunking with configurable overlap.

- `04_embeddings.ipynb`
  Embedding generation with VoyageAI: client setup, model selection,
  `input_type` semantics, and cosine similarity.

- `04_vectordb.ipynb`
  In-memory vector index: bulk insertion, cosine and Euclidean distance
  metrics, and nearest-neighbour retrieval.

- `04_bm25.ipynb`
  BM25 keyword index: tokenisation, IDF weighting, and ranked document
  retrieval for sparse search.

- `04_hybrid.ipynb`
  Hybrid retrieval combining BM25 and vector search via Reciprocal Rank
  Fusion (RRF), with a full RAG pipeline over a document corpus.

### Advanced Features

- `05_caching.ipynb`
  Prompt caching with `cache_control`: reducing latency and cost on repeated
  large-context requests.

- `05_citations.ipynb`
  Document citations: attaching source blocks to responses and extracting
  `CharLocation` references for precise attribution.

- `05_code_execution.ipynb`
  Code execution beta: uploading files with the Files API, running Python
  inside a sandboxed container, and downloading results.

- `05_images.ipynb`
  Vision inputs: sending base64-encoded images, analysing visual content, and
  applying Claude's multimodal reasoning.

- `05_pdf.ipynb`
  PDF document processing: uploading PDFs as document blocks, summarising
  content, and extracting structured information.

- `05_thinking.ipynb`
  Extended thinking mode: enabling `thinking` blocks, setting `budget_tokens`,
  and working with redacted thinking signatures for complex reasoning tasks.

## Reusable Helper Module

The file `scripts/anthropic_client.py` contains a small wrapper around
recurring Anthropic SDK patterns. It is intentionally lightweight and
educational rather than abstract:

- client creation (`get_client`)
- plain text generation (`generate_text`)
- structured output parsing via tool use (`generate_structured`)
- text streaming (`stream_response`)
- multi-turn message input (`generate_text_from_messages`)
- embedding generation via VoyageAI (`generate_embedding`, `generate_embeddings`)

It is particularly useful in notebooks and quick experiments when you want to
reduce boilerplate without hiding the underlying API shape.

## CLI Project: MCP Chat

The `cli_project/` directory contains a standalone command-line application
that goes beyond isolated notebook examples and demonstrates a more complete,
production-oriented usage of the Anthropic API.

**MCP Chat** is an interactive chat CLI built on the MCP (Model Control
Protocol) architecture. It showcases:

- conversational chat with Claude via the Anthropic API
- document retrieval using the `@document_id` syntax
- slash commands (`/summarize`, etc.) dispatched through an MCP server
- Tab completion for available commands
- extensible tool integration via the MCP client/server split

The project has its own `pyproject.toml` and virtual environment. See
`cli_project/README.md` for setup and usage instructions.

## Suggested Learning Path

If you are new to the Anthropic API, a good progression is:

1. `anthropic_quick_reference.ipynb` — SDK overview in one notebook
2. `00_notebook_basic_API.ipynb` — core request patterns
3. `02_prompting.ipynb` — prompt engineering fundamentals
4. `01_prompt_evals.ipynb` — evaluating and iterating on prompts
5. `03_tools.ipynb` — introduction to tool use
6. `03_tools_multi_turns.ipynb` — agentic loops
7. `03_tools_using_multiple_tools.ipynb` — multi-tool dispatch
8. `03_tool_streaming.ipynb` — real-time streaming with tools
9. `04_chunking.ipynb` → `04_embeddings.ipynb` → `04_vectordb.ipynb` — RAG foundations
10. `04_bm25.ipynb` → `04_hybrid.ipynb` — hybrid retrieval
11. `05_images.ipynb` / `05_pdf.ipynb` — multimodal inputs
12. `05_caching.ipynb` — cost and latency optimisation
13. `05_thinking.ipynb` — extended reasoning
14. `05_citations.ipynb` / `05_code_execution.ipynb` — specialised beta features

## Main Anthropic API Use Cases Covered

Taken together, the notebooks cover the following major Anthropic API
use cases:

- basic text generation and instruction following
- prompt engineering and few-shot prompting
- automated prompt evaluation with LLM-based grading
- tool use with single and multiple tools
- multi-turn agentic loops
- real-time streaming of text and tool calls
- built-in tools (web search, text editor)
- text chunking and preprocessing
- semantic embeddings and vector similarity
- BM25 keyword search
- hybrid retrieval with RRF
- retrieval-augmented generation (RAG)
- prompt caching for cost and latency reduction
- document citations and source attribution
- sandboxed code execution
- vision and multimodal inputs
- PDF document processing
- extended thinking mode for complex reasoning

## Philosophy Of The Labs

These labs are designed to stay:

- simple enough to read in one sitting
- concrete enough to run and modify immediately
- modular enough to isolate one idea per notebook
- practical enough to serve as real future references

The collection is not meant to be a framework. It is meant to be a clear,
useful learning workspace for experimenting with the Anthropic API.

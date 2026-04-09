# AI Engineering Labs

A hands-on, multi-provider learning workspace for the Python AI ecosystem.
Each folder is a self-contained module with its own dependencies, covering one
provider or framework — from raw API usage to retrieval pipelines, autonomous
agents, and production deployment patterns.

The labs are written to be read in one sitting, run immediately, and adapted
for real projects. Every module progresses from the simplest possible request
to more advanced, production-oriented patterns.

---

## Repository Overview

| Folder | Provider / Theme | Use Cases Covered | Format | Status |
| ------ | --------------- | ----------------- | ------ | ------ |
| [`01_openai/`](01_openai/) | OpenAI | Text generation · Structured outputs · Function calling · Tool use · Streaming · Embeddings · Semantic search · Image generation · Audio (TTS + STT) · Video generation · Token cost tracking · Multi-turn conversations · Error handling · Model comparison | Python scripts + Notebooks | ✅ Available |
| [`02_anthropic/`](02_anthropic/) | Anthropic Claude | Text generation · Prompt engineering · Prompt evaluation (LLM-as-judge) · Tool use · Agentic loops · Web search · Text editor tool · RAG (chunking, embeddings, BM25, hybrid) · Prompt caching · Extended thinking · Citations · Vision · PDF processing · Code execution · MCP CLI project | Notebooks + CLI project | ✅ Available |
| [`03_google_ai/`](03_google_ai/) | Google Gemini | Text generation · Multimodal (Images, Audio, Video) · Long context (2M tokens) · Structured outputs · Function calling · Code execution · Model-as-a-Judge · Multimodal Live API · Context Caching · Batch API · Controlled generation (Thinking mode, Hinting) | Notebooks + Scripts | ✅ Available |
| `04_multi_provider/` | Cross-provider | Side-by-side benchmarks · Cost comparison · Latency profiling · Output quality evaluation · Provider abstraction patterns | Notebooks | 🔵 Planned |
| `05_rag_langchain/` | LangChain | Document loaders · Text splitters · Vector stores · Retrieval chains · Conversational RAG · Evaluation with RAGAS | Notebooks + Scripts | 🔵 Planned |
| `06_agents_langgraph/` | LangGraph | ReAct agents · Multi-agent workflows · Human-in-the-loop · Stateful graphs · Tool orchestration · Agent memory | Notebooks + Scripts | 🔵 Planned |
| `07_local_models_llamaindex/` | LlamaIndex + Ollama | Local LLM inference · Offline RAG · Index types · Query engines · Custom retrievers | Notebooks + Scripts | 🔵 Planned |
| `08_fastapi_backend/` | FastAPI | REST API for AI endpoints · Streaming responses · Auth patterns · Background tasks · OpenAPI schema | Scripts + App | 🔵 Planned |
| `09_docker_deploy/` | Docker | Containerising AI apps · Multi-stage builds · Compose for local stacks · Environment management · Health checks | Config + Scripts | 🔵 Planned |

---

## Key Files and Entry Points

### `01_openai/` — OpenAI Labs

| Resource | Description |
| -------- | ----------- |
| [`01_openai/README.md`](01_openai/README.md) | Full overview, installation, and learning path |
| [`01_openai/notebooks/openai_quick_reference.ipynb`](01_openai/notebooks/openai_quick_reference.ipynb) | Compact SDK reference: text, structured outputs, streaming, embeddings |
| [`01_openai/scripts/openai_client.py`](01_openai/scripts/openai_client.py) | Reusable helper module shared across all OpenAI scripts |
| [`01_openai/scripts/02_structured_output_scenarios.py`](01_openai/scripts/02_structured_output_scenarios.py) | Typed parsing with Pydantic and strict JSON schemas |
| [`01_openai/scripts/10_embeddings_scenarios.py`](01_openai/scripts/10_embeddings_scenarios.py) | Semantic similarity and search workflows |
| [`01_openai/scripts/14_model_comparison_scenarios.py`](01_openai/scripts/14_model_comparison_scenarios.py) | Cost, latency, and quality benchmarks across models |

### `02_anthropic/` — Anthropic Claude Labs

| Resource | Description |
| -------- | ----------- |
| [`02_anthropic/README.md`](02_anthropic/README.md) | Full overview, installation, and learning path |
| [`02_anthropic/notebooks/anthropic_quick_reference.ipynb`](02_anthropic/notebooks/anthropic_quick_reference.ipynb) | Compact SDK reference: text, structured outputs, streaming, embeddings |
| [`02_anthropic/scripts/anthropic_client.py`](02_anthropic/scripts/anthropic_client.py) | Reusable helper module shared across all Anthropic notebooks |
| [`02_anthropic/notebooks/02_prompting.ipynb`](02_anthropic/notebooks/02_prompting.ipynb) | Prompt engineering with automated LLM-based evaluation |
| [`02_anthropic/notebooks/04_hybrid.ipynb`](02_anthropic/notebooks/04_hybrid.ipynb) | Hybrid RAG pipeline: BM25 + vector search with RRF fusion |
| [`02_anthropic/notebooks/05_thinking.ipynb`](02_anthropic/notebooks/05_thinking.ipynb) | Extended thinking mode for complex reasoning tasks |
| [`02_anthropic/cli_project/README.md`](02_anthropic/cli_project/README.md) | MCP Chat CLI: real-world tool use and document retrieval via MCP |

### `03_google_ai/` — Google Gemini Labs

| Resource | Description |
| -------- | ----------- |
| [`03_google_ai/README.md`](03_google_ai/README.md) | Full overview, installation, and learning path |
| [`03_google_ai/notebooks/00_quickstart_gemini.ipynb`](03_google_ai/notebooks/00_quickstart_gemini.ipynb) | Compact SDK reference: text, structured outputs, streaming, embeddings |
| [`03_google_ai/scripts/gemini_client.py`](03_google_ai/scripts/gemini_client.py) | Reusable helper module shared across all Google AI notebooks |
| [`03_google_ai/notebooks/04_multimodality_deep_dive.ipynb`](03_google_ai/notebooks/04_multimodality_deep_dive.ipynb) | Advanced vision, audio, and video reasoning |
| [`03_google_ai/notebooks/10_multimodal_live_interactions.ipynb`](03_google_ai/notebooks/10_multimodal_live_interactions.ipynb) | Real-time WebSocket audio and tool calling agents |
| [`03_google_ai/notebooks/11_advanced_optimization.ipynb`](03_google_ai/notebooks/11_advanced_optimization.ipynb) | Context caching, dynamic retrieval, and grounding |
| [`03_google_ai/notebooks/13_batch_api_processing.ipynb`](03_google_ai/notebooks/13_batch_api_processing.ipynb) | High-throughput offline processing with 50% cost reduction |

---

## Getting Started

Each module is self-contained. Clone the repository and navigate to the folder
you want to explore:

```bash
git clone <repo-url>
cd ai_engineering_labs
```

Then follow the installation instructions in the relevant `README.md`. All
modules use [`uv`](https://github.com/astral-sh/uv) as the recommended package
manager and share a single `.env` file at the repository root.

### Root `.env` file

Create a `.env` file at the repository root and populate the keys required by
the modules you want to run:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google AI (Gemini)
GEMINI_API_KEY=your_gemini_api_key_here

# VoyageAI (used for embeddings in Anthropic labs)
VOYAGE_API_KEY=your_voyageai_api_key_here
```

---

## Design Principles

- **One idea per file.** Each script or notebook isolates a single concept so
  it can be read, run, and reused independently.
- **Minimal dependencies.** Each module installs only what it needs. There is
  no shared monolithic environment.
- **Progressive structure.** Numbered files within each module follow a natural
  learning curve from the simplest call to advanced patterns.
- **Production-aware.** Labs do not stop at toy examples — they cover error
  handling, cost tracking, caching, streaming, deployment, and evaluation.
- **Provider-agnostic mindset.** Modules are structured consistently so
  patterns learned in one transfer easily to the next.

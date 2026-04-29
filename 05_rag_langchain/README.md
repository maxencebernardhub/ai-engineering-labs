# Lab 05 — RAG Pipeline with LangChain

A complete, standalone Retrieval-Augmented Generation (RAG) lab demonstrating the full pipeline
from document ingestion to generation, with advanced techniques (query expansion, CrossEncoder
reranking) and formal evaluation (RAGAS). Packaged as four progressive notebooks and a deployable
Streamlit app.

---

## Architecture

```text
                        ┌──────────────────────────────────────────────┐
                        │               Streamlit App                  │
                        │  Upload PDF/TXT/MD   ·   Ask a question      │
                        └──────────────┬───────────────┬───────────────┘
                                       │               │
                        ┌──────────────▼───┐    ┌───────▼──────────────┐
                        │   app/ingest.py  │    │   app/query.py       │
                        │                  │    │                      │
                        │  Load document   │    │  1. Query expansion  │
                        │  Chunk (RCTS)    │    │     (LLM → 3 vars)   │
                        │  Embed (bge-m3)  │    │  2. Similarity search│
                        │  MD5 dedup       │    │  3. CrossEncoder     │
                        │  Store → ChromaDB│    │     reranking        │
                        └──────────────────┘    │  4. LCEL chain       │
                                                │     (prompt|llm|     │
                        ┌─────────────────┐     │      parser)         │
                        │    ChromaDB     │◄───►│                      │
                        │  (persisted on  │     └──────────────────────┘
                        │   disk)         │
                        └─────────────────┘
```

### Key design choices

| Component | Choice | Why |
| --- | --- | --- |
| Embeddings | `BAAI/bge-m3` (local) | No API key, multilingual, competitive quality |
| Vector store | ChromaDB with `persist_directory` | Survives app restarts, local-first |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | More accurate than cosine similarity alone |
| Query expansion | LLM → 2 alternative phrasings | Broadens recall without HyDE complexity |
| LLM abstraction | `init_chat_model()` + LCEL (`\|`) | Provider-agnostic, runtime-switchable |
| Deduplication | MD5 hash of uploaded file | Prevents re-indexing the same document |

> **Comparison with Lab 04**: Lab 04 (`llm_client/`) shows what LangChain abstracts away —
> raw HTTP calls, manual prompt construction, JSON parsing. This lab uses LangChain to focus
> on RAG architecture rather than API plumbing.

---

## Notebooks

| Notebook | Topic | What you learn |
| --- | --- | --- |
| `01_embeddings_basics.ipynb` | Embeddings | What a vector is, cosine similarity, semantic search, PCA visualisation |
| `02_chromadb_indexing.ipynb` | Chunking + ChromaDB | Chunking strategies, overlap trade-offs, CRUD on a vector store |
| `03_rag_pipeline.ipynb` | Full RAG pipeline | Query expansion → retrieval → CrossEncoder reranking → LCEL chain |
| `04_rag_evaluation.ipynb` | RAGAS evaluation | Context Precision, Recall, Faithfulness, Answer Relevancy metrics |

Each notebook is **standalone** — no shared state, runnable independently.

---

## Streamlit App

The app exposes the full pipeline interactively:

- Upload PDF, TXT, or Markdown documents
- Automatic deduplication by MD5 hash — uploading the same file twice is a no-op
- Select LLM provider/model at runtime (Anthropic, OpenAI, Google)
- Ask questions — see the answer and the source chunks with document name

---

## Setup

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) installed
- API key for at least one LLM provider

### 1. Clone and install

```bash
git clone <repo-url>
cd 05_rag_langchain
uv sync --all-extras
```

The `--all-extras` flag installs notebook dependencies (`jupyterlab`, `matplotlib`,
`scikit-learn`). The base install (without extras) is enough to run the Streamlit app.

### 2. Environment variables

Create a `.env` file at the **repository root** (not inside `05_rag_langchain/`):

```bash
# .env  — at repo root
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
```

Only the key(s) for the provider(s) you plan to use are required.

### 3. Run the Streamlit app

```bash
uv run streamlit run app/app.py
```

### 4. Run the notebooks

```bash
uv run jupyter lab
```

Then open any notebook from the JupyterLab file browser.

---

## Sample documents

The `sample_docs/` directory is gitignored (documents may be copyrighted).
Any PDF, TXT, or Markdown file works. Good sources for test documents:

- [arXiv](https://arxiv.org) — open-access research papers in PDF
- [Project Gutenberg](https://www.gutenberg.org) — public domain books in TXT
- Wikipedia articles saved as Markdown

---

## Project structure

```text
05_rag_langchain/
├── pyproject.toml                  # uv project — base + notebook extras
├── 01_embeddings_basics.ipynb      # Embeddings, cosine similarity, PCA
├── 02_chromadb_indexing.ipynb      # Chunking strategies + ChromaDB CRUD
├── 03_rag_pipeline.ipynb           # Full LCEL RAG + reranking + query expansion
├── 04_rag_evaluation.ipynb         # RAGAS evaluation
├── app/
│   ├── app.py                      # Streamlit entry point
│   ├── config.py                   # Models, chunk size, top-k constants
│   ├── ingest.py                   # Load, chunk, embed, store in ChromaDB
│   └── query.py                    # Retrieve + rerank + generate
├── tests/
│   ├── unit/                       # 29 unit tests (pytest)
│   └── integration/                # 10 integration tests (real ChromaDB)
├── sample_docs/                    # Test documents (gitignored)
└── docs/specs/                     # Feature spec
```

---

## Running tests

```bash
uv run pytest
```

39 tests total — 29 unit, 10 integration. Integration tests spin up a real ChromaDB
instance in a temporary directory (`tmp_path` isolation).

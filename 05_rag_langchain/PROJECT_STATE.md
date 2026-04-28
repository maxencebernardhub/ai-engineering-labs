# Project State — RAG Pipeline with LangChain

## Current Status

**Phase**: Brainstorming complete — Feature Brief validated, moving to implementation planning.
**Branch**: `feat/05-rag-langchain`
**Last updated**: 2026-04-28

---

## Feature Brief

See full spec:
[`docs/specs/2026-04-28-05-rag-langchain.md`](docs/specs/2026-04-28-05-rag-langchain.md)

**Goal**: Build a complete, standalone RAG lab with LangChain demonstrating the full pipeline —
document ingestion → retrieval → generation — with advanced techniques (reranking, query
expansion) and formal evaluation (RAGAS), packaged as both interactive notebooks and a
deployable Streamlit app.

---

## Planned Structure

```text
05_rag_langchain/
├── README.md
├── pyproject.toml
├── PROJECT_STATE.md
├── docs/specs/2026-04-28-05-rag-langchain.md
├── 01_embeddings_basics.ipynb       # Embeddings, cosine similarity — standalone
├── 02_chromadb_indexing.ipynb       # Chunking strategies + ChromaDB CRUD — standalone
├── 03_rag_pipeline.ipynb            # Full LCEL RAG + reranking + query expansion — standalone
├── 04_rag_evaluation.ipynb          # RAGAS evaluation — standalone
├── app/
│   ├── app.py                       # Streamlit entry point
│   ├── config.py                    # Models, chunk size, top-k constants
│   ├── ingest.py                    # Load, chunk, embed, store in ChromaDB
│   └── query.py                     # Retrieve + rerank + generate
└── sample_docs/                     # 2-3 test documents (English)
```

---

## Key Design Decisions

- **Embeddings**: `BAAI/bge-m3` via `langchain-huggingface` — local, multilingual, no API key
- **LLM abstraction**: `init_chat_model()` (LCEL) — provider switchable at runtime via UI
- **Vector store**: ChromaDB with `persist_directory` — survives app restarts
- **Deduplication**: MD5 hash of uploaded file stored as ChromaDB metadata
- **Retrieval**: similarity search (top-k) → CrossEncoder reranking
- **Query expansion**: LLM generates 2-3 question variants before retrieval, results merged
- **Evaluation**: RAGAS metrics — Context Precision, Recall, Faithfulness, Answer Relevancy
- **No LangGraph / LangSmith / agents** — explicit LCEL chains only
- **Reference to lab 04**: `llm_client/` mentioned in README as "what LangChain abstracts away"

---

## Implementation Steps

| # | Step | Status |
| --- | --- | --- |
| 1 | Project setup: `pyproject.toml` + directory structure + `uv sync` | `todo` |
| 2 | `01_embeddings_basics.ipynb` | `todo` |
| 3 | `02_chromadb_indexing.ipynb` | `todo` |
| 4 | `app/config.py` + `app/ingest.py` | `todo` |
| 5 | `app/query.py` (retrieval + reranking + query expansion + generation) | `todo` |
| 6 | `03_rag_pipeline.ipynb` | `todo` |
| 7 | `04_rag_evaluation.ipynb` (RAGAS) | `todo` |
| 8 | `app/app.py` (Streamlit UI) | `todo` |
| 9 | `sample_docs/` — 2-3 English test documents | `todo` |
| 10 | `README.md` with architecture diagram + GIF | `todo` |

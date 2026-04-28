# Feature: Lab 05 — RAG Pipeline with LangChain

## Feature Brief

**Goal**: Build a complete, standalone RAG lab with LangChain demonstrating the full pipeline —
document ingestion → retrieval → generation — with advanced techniques (reranking, query
expansion) and formal evaluation (RAGAS), packaged as both interactive notebooks and a
deployable Streamlit app.

**Users**: Developer/ML learner building their AI engineering portfolio; potential employers or
clients assessing RAG depth of knowledge.

**Acceptance criteria**:

- Notebook 01 explains embeddings and cosine similarity from scratch, standalone
- Notebook 02 covers chunking strategies comparison (RecursiveCharacter vs Semantic) + ChromaDB
  CRUD, standalone
- Notebook 03 implements a full LCEL RAG pipeline with reranking (CrossEncoder) and query
  expansion, standalone
- Notebook 04 evaluates the pipeline with RAGAS (Context Precision, Recall, Faithfulness,
  Answer Relevancy), standalone
- Streamlit app accepts PDF/TXT/Markdown uploads, indexes into a persisted ChromaDB (dedup by
  MD5 hash), lets user select LLM provider/model via selectbox, answers questions, displays
  answer + source chunks with document name and page number
- All notebooks use `BAAI/bge-m3` embeddings (local, no API key required)
- LLM provider switched at runtime via `init_chat_model()` — no code change needed
- README includes architecture diagram, setup instructions, and GIF/screenshots of the app
- `.env` consumed from repo root (not duplicated inside the lab directory)

**Edge cases**:

- Same document uploaded twice → skip re-indexing (MD5 dedup)
- Empty document or unsupported format → clear error message in UI
- LLM provider API key missing → graceful fallback message, not a crash
- Query returns 0 relevant chunks → explicit "no relevant context found" message rather than
  hallucinated answer
- Large PDF (100+ pages) → chunking must complete without memory error

**Dependencies**:

- `langchain`, `langchain-community`, `langchain-huggingface`, `langchain-chroma`,
  `langchain-openai`, `langchain-anthropic`, `langchain-google-genai`
- `chromadb`, `sentence-transformers` (BAAI/bge-m3 + CrossEncoder ms-marco-MiniLM-L-6-v2)
- `streamlit`, `pypdf`, `ragas`
- `.env` at repo root with `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
- Python managed via `uv`, `pyproject.toml` inside `05_rag_langchain/`

**Constraints**:

- No LangGraph, LangSmith, or Deep Agents
- ChromaDB local only (persisted to `app/chroma_db/`, gitignored)
- `init_chat_model()` for all LLM calls (not agent), LCEL chains throughout
- Each notebook standalone and executable independently
- All code, docstrings, comments, README, PROJECT_STATE.md in English
- `llm_client/` from lab 04 referenced in README as pedagogical comparison, not used in code

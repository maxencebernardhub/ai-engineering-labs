"""
Lab 05 — RAG with LangChain: runtime configuration.

Single source of truth for all constants: paths, model identifiers, chunking
parameters, and retrieval settings. Loaded once at import time; every other
module reads from here rather than hard-coding values.
"""

from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Load API keys from .env at repo root (searches up the directory tree)
load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LAB_DIR = Path(__file__).parent.parent
CHROMA_DB_DIR = LAB_DIR / "app" / "chroma_db"
SAMPLE_DOCS_DIR = LAB_DIR / "sample_docs"

# ---------------------------------------------------------------------------
# Embeddings (local, no API key required)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "BAAI/bge-m3"

# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# Candidates fetched from ChromaDB before reranking
RETRIEVAL_TOP_K = 10
# Final chunks kept after CrossEncoder reranking
RERANK_TOP_K = 4

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

CHROMA_COLLECTION_NAME = "rag_documents"

# ---------------------------------------------------------------------------
# Available LLM models for the Streamlit selectbox
# Keys are display labels; values are init_chat_model() identifiers.
# ---------------------------------------------------------------------------

AVAILABLE_MODELS: dict[str, str] = {
    "Anthropic — Claude Opus 4.7": "anthropic:claude-opus-4-7",
    "Anthropic — Claude Sonnet 4.6": "anthropic:claude-sonnet-4-6",
    "Anthropic — Claude Haiku 4.5": "anthropic:claude-haiku-4-5-20251001",
    "OpenAI — GPT-5.5": "openai:gpt-5.5",
    "OpenAI — GPT-5.4": "openai:gpt-5.4",
    "OpenAI — GPT-5.4 Mini": "openai:gpt-5.4-mini",
    "Google — Gemini 3.1 Pro": "google_genai:gemini-3.1-pro-preview",
    "Google — Gemini 3.1 Flash Lite": "google_genai:gemini-3.1-flash-lite-preview",
    "Google — Gemini 3 Flash": "google_genai:gemini-3-flash-preview",
    "Google — Gemini 2.5 Pro": "google_genai:gemini-2.5-pro",
    "Google — Gemini 2.5 Flash": "google_genai:gemini-2.5-flash",
}

DEFAULT_MODEL = "Anthropic — Claude Sonnet 4.6"

# ---------------------------------------------------------------------------
# Supported file extensions for document upload
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

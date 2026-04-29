"""
Lab 05 — RAG with LangChain: document ingestion pipeline.

Responsibilities:
- Load PDF, TXT, and Markdown files into LangChain Documents
- Split documents into overlapping chunks (RecursiveCharacterTextSplitter)
- Deduplicate via MD5 hash to avoid re-indexing the same file
- Store chunks with metadata (md5, source_file) into a ChromaDB vectorstore

Entry point: ingest_document(file_path, vectorstore, source_name=None)
"""

import hashlib
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config


def compute_md5(file_path: Path) -> str:
    """Return the MD5 hex digest of a file's binary contents."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def load_document(file_path: Path) -> list[Document]:
    """Load a file into LangChain Documents.

    Supports .pdf, .txt, and .md. Raises ValueError for other formats.
    """
    suffix = file_path.suffix.lower()
    if suffix not in config.SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            f"Supported: {sorted(config.SUPPORTED_EXTENSIONS)}"
        )
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    else:
        # TextLoader handles both .txt and .md (markdown is plain text)
        loader = TextLoader(str(file_path), encoding="utf-8")
    return loader.load()


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def is_already_indexed(md5: str, collection: Any) -> bool:
    """Return True if a document with this MD5 hash exists in the collection."""
    result = collection.get(where={"md5": {"$eq": md5}})
    return len(result["ids"]) > 0


def ingest_document(
    file_path: Path, vectorstore: Any, source_name: str | None = None
) -> tuple[bool, int]:
    """Full ingestion pipeline for a single file.

    Returns (skipped, num_chunks).
    skipped=True means the document was already indexed (MD5 dedup).
    source_name overrides file_path.name in chunk metadata (useful when
    file_path points to a temp file).
    """
    md5 = compute_md5(file_path)

    if is_already_indexed(md5, vectorstore._collection):
        return True, 0

    documents = load_document(file_path)
    chunks = chunk_documents(documents)

    display_name = source_name if source_name is not None else file_path.name
    for chunk in chunks:
        chunk.metadata["md5"] = md5
        chunk.metadata["source_file"] = display_name

    vectorstore.add_documents(chunks)
    return False, len(chunks)

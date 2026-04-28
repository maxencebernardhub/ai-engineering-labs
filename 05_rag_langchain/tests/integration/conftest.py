import pytest
from langchain_chroma import Chroma
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_huggingface import HuggingFaceEmbeddings

from app import config


@pytest.fixture(scope="module")
def embeddings():
    """Load BAAI/bge-m3 once per test module (download ~570MB on first run)."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


@pytest.fixture
def vectorstore(tmp_path, embeddings):
    """Fresh isolated ChromaDB for each test."""
    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(tmp_path / "chroma_db"),
    )


@pytest.fixture
def fake_llm():
    """Minimal LLM stub for testing pipeline plumbing without API calls."""
    return FakeListChatModel(responses=["This is a generated test answer."])


@pytest.fixture
def sample_txt(tmp_path):
    """A simple English text document for ingestion tests."""
    doc = tmp_path / "sample.txt"
    doc.write_text(
        "The speed of light in a vacuum is approximately 299,792 kilometres per second. "
        "This fundamental constant, denoted c, plays a central role in Einstein's theory "
        "of special relativity. No information or matter can travel faster than light.",
        encoding="utf-8",
    )
    return doc

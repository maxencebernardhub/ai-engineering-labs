import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from app.ingest import (
    chunk_documents,
    compute_md5,
    is_already_indexed,
    load_document,
)
from langchain_core.documents import Document

from app import config


class TestComputeMd5:
    def test_is_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        assert compute_md5(f) == compute_md5(f)

    def test_differs_for_different_content(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello", encoding="utf-8")
        f2.write_text("world", encoding="utf-8")
        assert compute_md5(f1) != compute_md5(f2)

    def test_matches_expected_hash(self, tmp_path):
        content = b"deterministic content"
        f = tmp_path / "test.txt"
        f.write_bytes(content)
        expected = hashlib.md5(content).hexdigest()
        assert compute_md5(f) == expected


class TestLoadDocument:
    def test_load_txt_returns_document_list(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("This is a plain text document.", encoding="utf-8")
        docs = load_document(f)
        assert isinstance(docs, list)
        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)

    def test_load_markdown_returns_document_list(self, tmp_path):
        f = tmp_path / "sample.md"
        f.write_text("# Title\n\nSome markdown content.", encoding="utf-8")
        docs = load_document(f)
        assert isinstance(docs, list)
        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)

    def test_load_pdf_returns_document_list(self):
        mock_docs = [Document(page_content="PDF content", metadata={"page": 0})]
        with patch("app.ingest.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = mock_docs
            docs = load_document(Path("dummy.pdf"))
        assert docs == mock_docs

    def test_unsupported_format_raises_value_error(self, tmp_path):
        f = tmp_path / "report.docx"
        f.write_bytes(b"fake content")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_document(f)

    def test_txt_content_is_preserved(self, tmp_path):
        content = "Specific content that must be preserved verbatim."
        f = tmp_path / "sample.txt"
        f.write_text(content, encoding="utf-8")
        docs = load_document(f)
        assert content in docs[0].page_content


class TestChunkDocuments:
    def test_produces_multiple_chunks_for_long_document(self):
        long_content = "word " * (config.CHUNK_SIZE // 2)
        docs = [Document(page_content=long_content, metadata={})]
        chunks = chunk_documents(docs)
        assert len(chunks) > 1

    def test_short_document_stays_as_single_chunk(self):
        docs = [Document(page_content="This is short.", metadata={})]
        chunks = chunk_documents(docs)
        assert len(chunks) == 1

    def test_metadata_is_preserved_in_chunks(self):
        long_content = "word " * (config.CHUNK_SIZE // 2)
        docs = [Document(page_content=long_content, metadata={"source": "test.txt"})]
        chunks = chunk_documents(docs)
        assert all(c.metadata.get("source") == "test.txt" for c in chunks)

    def test_returns_list_of_documents(self):
        docs = [Document(page_content="Some content.", metadata={})]
        chunks = chunk_documents(docs)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Document) for c in chunks)


class TestIsAlreadyIndexed:
    def test_returns_true_if_hash_present(self):
        collection = MagicMock()
        collection.get.return_value = {"ids": ["doc1", "doc2"]}
        assert is_already_indexed("abc123", collection) is True

    def test_returns_false_if_hash_absent(self):
        collection = MagicMock()
        collection.get.return_value = {"ids": []}
        assert is_already_indexed("abc123", collection) is False

    def test_queries_with_correct_md5_filter(self):
        collection = MagicMock()
        collection.get.return_value = {"ids": []}
        is_already_indexed("myhash", collection)
        collection.get.assert_called_once_with(where={"md5": {"$eq": "myhash"}})

from unittest.mock import patch

import pytest

from app.ingest import ingest_document
from app.query import answer_question


@pytest.mark.integration
class TestIngestion:
    def test_ingest_txt_adds_documents_to_collection(self, vectorstore, sample_txt):
        skipped, num_chunks = ingest_document(sample_txt, vectorstore)
        assert skipped is False
        assert num_chunks > 0
        assert vectorstore._collection.count() == num_chunks

    def test_ingest_same_file_twice_skips_second(self, vectorstore, sample_txt):
        ingest_document(sample_txt, vectorstore)
        skipped, num_chunks = ingest_document(sample_txt, vectorstore)
        assert skipped is True
        assert num_chunks == 0

    def test_ingested_chunks_carry_md5_metadata(self, vectorstore, sample_txt):
        ingest_document(sample_txt, vectorstore)
        results = vectorstore._collection.get(include=["metadatas"])
        assert all("md5" in m for m in results["metadatas"])

    def test_ingested_chunks_carry_source_file_metadata(self, vectorstore, sample_txt):
        ingest_document(sample_txt, vectorstore)
        results = vectorstore._collection.get(include=["metadatas"])
        assert all(
            m.get("source_file") == sample_txt.name for m in results["metadatas"]
        )


@pytest.mark.integration
class TestRetrieval:
    def test_similarity_search_returns_relevant_chunk(self, vectorstore, sample_txt):
        ingest_document(sample_txt, vectorstore)
        results = vectorstore.similarity_search("What is the speed of light?", k=3)
        assert len(results) > 0
        combined = " ".join(r.page_content for r in results)
        assert "light" in combined.lower()

    def test_similarity_search_returns_document_objects(self, vectorstore, sample_txt):
        from langchain_core.documents import Document

        ingest_document(sample_txt, vectorstore)
        results = vectorstore.similarity_search("speed of light", k=2)
        assert all(isinstance(r, Document) for r in results)


@pytest.mark.integration
class TestFullPipeline:
    def test_answer_question_returns_expected_keys(
        self, vectorstore, sample_txt, fake_llm
    ):
        ingest_document(sample_txt, vectorstore)
        with patch(
            "app.query.expand_query",
            return_value=["What is the speed of light?"],
        ):
            result = answer_question(
                "What is the speed of light?", vectorstore, fake_llm
            )
        assert "answer" in result
        assert "sources" in result

    def test_answer_question_returns_non_empty_answer(
        self, vectorstore, sample_txt, fake_llm
    ):
        ingest_document(sample_txt, vectorstore)
        with patch(
            "app.query.expand_query",
            return_value=["What is the speed of light?"],
        ):
            result = answer_question(
                "What is the speed of light?", vectorstore, fake_llm
            )
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_answer_question_includes_sources(self, vectorstore, sample_txt, fake_llm):
        ingest_document(sample_txt, vectorstore)
        with patch(
            "app.query.expand_query",
            return_value=["What is the speed of light?"],
        ):
            result = answer_question(
                "What is the speed of light?", vectorstore, fake_llm
            )
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) > 0
        assert "source_file" in result["sources"][0]

    def test_empty_vectorstore_returns_no_context_message(self, vectorstore, fake_llm):
        with patch(
            "app.query.expand_query",
            return_value=["What is the speed of light?"],
        ):
            result = answer_question(
                "What is the speed of light?", vectorstore, fake_llm
            )
        assert "enough information" in result["answer"]
        assert result["sources"] == []

from unittest.mock import MagicMock, patch

from app.query import (
    build_rag_chain,
    expand_query,
    format_sources,
    rerank,
)
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import Runnable

from app import config


class TestExpandQuery:
    def _make_llm_mock(self, questions: list[str]) -> MagicMock:
        mock_result = MagicMock()
        mock_result.questions = questions
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_result
        return mock_llm

    def test_returns_list_of_strings(self):
        llm = self._make_llm_mock(["alt 1", "alt 2"])
        result = expand_query("original question", llm)
        assert isinstance(result, list)
        assert all(isinstance(q, str) for q in result)

    def test_includes_original_question(self):
        llm = self._make_llm_mock(["alt 1", "alt 2"])
        result = expand_query("original question", llm)
        assert "original question" in result

    def test_returns_multiple_questions(self):
        llm = self._make_llm_mock(["alt 1", "alt 2"])
        result = expand_query("original question", llm)
        assert len(result) > 1

    def test_works_when_llm_returns_no_alternatives(self):
        llm = self._make_llm_mock([])
        result = expand_query("original question", llm)
        assert result == ["original question"]


class TestRerank:
    def _make_docs(self, n: int) -> list[Document]:
        return [Document(page_content=f"content {i}", metadata={}) for i in range(n)]

    def test_sorts_by_descending_score(self):
        docs = self._make_docs(3)
        with patch("app.query.CrossEncoder") as MockCE:
            MockCE.return_value.predict.return_value = [0.1, 0.9, 0.5]
            result = rerank("question", docs)
        # doc[1] has highest score (0.9), should be first
        assert result[0].page_content == "content 1"

    def test_empty_list_returns_empty(self):
        result = rerank("question", [])
        assert result == []

    def test_limits_output_to_rerank_top_k(self):
        docs = self._make_docs(config.RERANK_TOP_K + 3)
        scores = list(range(len(docs)))
        with patch("app.query.CrossEncoder") as MockCE:
            MockCE.return_value.predict.return_value = scores
            result = rerank("question", docs)
        assert len(result) <= config.RERANK_TOP_K

    def test_returns_document_objects(self):
        docs = self._make_docs(2)
        with patch("app.query.CrossEncoder") as MockCE:
            MockCE.return_value.predict.return_value = [0.8, 0.3]
            result = rerank("question", docs)
        assert all(isinstance(d, Document) for d in result)


class TestFormatSources:
    def test_extracts_source_file_and_page(self):
        docs = [
            Document(
                page_content="Some content here.",
                metadata={"source_file": "report.pdf", "page": 3},
            )
        ]
        sources = format_sources(docs)
        assert sources[0]["source_file"] == "report.pdf"
        assert sources[0]["page"] == 3

    def test_empty_list_returns_empty(self):
        assert format_sources([]) == []

    def test_handles_missing_metadata_gracefully(self):
        docs = [Document(page_content="Content.", metadata={})]
        sources = format_sources(docs)
        assert sources[0]["source_file"] == "unknown"
        assert sources[0]["page"] is None

    def test_content_preview_is_included(self):
        docs = [Document(page_content="Preview content.", metadata={})]
        sources = format_sources(docs)
        assert "content" in sources[0]
        assert len(sources[0]["content"]) > 0


class TestBuildRagChain:
    def test_returns_runnable(self):
        fake_llm = FakeListChatModel(responses=["answer"])
        chain = build_rag_chain(fake_llm)
        assert isinstance(chain, Runnable)

    def test_chain_produces_string_output(self):
        fake_llm = FakeListChatModel(responses=["This is the answer."])
        chain = build_rag_chain(fake_llm)
        result = chain.invoke({"context": "Some context.", "question": "A question?"})
        assert isinstance(result, str)

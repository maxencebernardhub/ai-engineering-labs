"""
Lab 05 — RAG with LangChain: retrieval and generation pipeline.

Implements the full RAG query flow as explicit LCEL steps (no agents):
1. Query expansion  — LLM generates 2 alternative phrasings to broaden recall
2. Retrieval        — similarity search over ChromaDB for each expanded query
3. Reranking        — CrossEncoder scores and re-orders candidate chunks
4. Generation       — RAG prompt | LLM | StrOutputParser LCEL chain

Entry point: answer_question(question, vectorstore, llm) → {"answer", "sources"}
"""

from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from app import config

# ---------------------------------------------------------------------------
# RAG prompt
# ---------------------------------------------------------------------------

_RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the question using only the context "
    "provided below. If the context does not contain enough information, "
    'say "I don\'t have enough information to answer this question."\n\n'
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------


class _QueryExpansion(BaseModel):
    questions: list[str] = Field(
        description="Alternative phrasings of the original question"
    )


_EXPANSION_PROMPT = ChatPromptTemplate.from_template(
    "Generate {n} alternative phrasings of the following question to improve "
    "document retrieval. Return only the alternative questions, not the original.\n\n"
    "Original question: {question}"
)


def expand_query(question: str, llm: Any) -> list[str]:
    """Return the original question plus LLM-generated alternative phrasings."""
    structured_llm = llm.with_structured_output(_QueryExpansion)
    prompt = _EXPANSION_PROMPT.format_messages(question=question, n=2)
    result: _QueryExpansion = structured_llm.invoke(prompt)
    return [question] + result.questions


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


def rerank(question: str, documents: list[Document]) -> list[Document]:
    """Rerank documents by relevance to the question using a CrossEncoder."""
    if not documents:
        return []
    reranker = CrossEncoder(config.RERANKER_MODEL)
    pairs = [(question, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked][: config.RERANK_TOP_K]


# ---------------------------------------------------------------------------
# Source formatting
# ---------------------------------------------------------------------------


def format_sources(documents: list[Document]) -> list[dict]:
    """Extract display metadata from a list of reranked documents."""
    return [
        {
            "content": doc.page_content[:300],
            "source_file": doc.metadata.get("source_file", "unknown"),
            "page": doc.metadata.get("page", None),
        }
        for doc in documents
    ]


# ---------------------------------------------------------------------------
# LCEL RAG chain
# ---------------------------------------------------------------------------


def build_rag_chain(llm: Any) -> Runnable:
    """Build the generation step as a LCEL chain: prompt | llm | parser."""
    return _RAG_PROMPT | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------


def answer_question(question: str, vectorstore: Any, llm: Any) -> dict:
    """Full RAG pipeline: query expansion → retrieval → reranking → generation.

    Returns {"answer": str, "sources": list[dict]}.
    """
    # 1. Query expansion
    expanded_questions = expand_query(question, llm)

    # 2. Retrieve for each variant and deduplicate by content
    all_docs: list[Document] = []
    seen: set[str] = set()
    for q in expanded_questions:
        for doc in vectorstore.similarity_search(q, k=config.RETRIEVAL_TOP_K):
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)

    if not all_docs:
        return {
            "answer": "I don't have enough information to answer this question.",
            "sources": [],
        }

    # 3. Rerank
    reranked = rerank(question, all_docs)

    # 4. Generate
    context = "\n\n---\n\n".join(doc.page_content for doc in reranked)
    chain = build_rag_chain(llm)
    answer = chain.invoke({"context": context, "question": question})

    return {"answer": answer, "sources": format_sources(reranked)}

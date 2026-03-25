"""Demonstrate several embedding scenarios with the OpenAI Embeddings API.

Embeddings are numeric vector representations of text. The key idea is simple:
texts with similar meaning tend to have vectors that are close to each other in
vector space, even when they do not use exactly the same words.

This makes embeddings useful for tasks such as:
- semantic search
- similarity comparison
- clustering or grouping
- duplicate detection
- retrieval-augmented generation pipelines

This lab covers three practical scenarios:
1. Compare semantic similarity between short texts.
2. Build a tiny semantic search over local examples.
3. Detect near-duplicate texts with cosine similarity.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512


def require_api_key() -> None:
    """Fail early with a clear error if the API key is missing."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")


def print_section(title: str) -> None:
    """Print a readable separator for each scenario."""

    print(f"\n{'=' * 18} {title} {'=' * 18}")


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b, strict=True))
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity cannot be computed on a zero vector.")

    return dot_product / (norm_a * norm_b)


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed one or more texts and return the vectors."""

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        input=texts,
    )
    return [item.embedding for item in response.data]


def run_similarity_scenario(client: OpenAI) -> None:
    """Compare the similarity of related and unrelated texts."""

    print_section("Scenario 1 - Semantic similarity")

    texts = [
        "How do I sort a Python list?",
        "What is the best way to order items in a Python list?",
        "How do I cook pasta al dente?",
    ]

    vectors = embed_texts(client, texts)

    similarity_related = cosine_similarity(vectors[0], vectors[1])
    similarity_unrelated = cosine_similarity(vectors[0], vectors[2])

    print("Texts compared:")
    for index, text in enumerate(texts, start=1):
        print(f"- text_{index}: {text}")

    print("\nCosine similarity scores:")
    print(f"- related_python_texts: {similarity_related:.4f}")
    print(f"- unrelated_texts: {similarity_unrelated:.4f}")


def run_semantic_search_scenario(client: OpenAI) -> None:
    """Perform a tiny semantic search over a local text collection."""

    print_section("Scenario 2 - Tiny semantic search")

    documents = [
        "Python lists are mutable collections and can be sorted in place.",
        "Tuples are immutable sequences often used for fixed-size records.",
        "A for loop iterates over items in a collection.",
        "Dictionaries store key-value pairs and allow fast lookups by key.",
        "Git commit records a snapshot of your project history.",
    ]
    query = "How can I keep key-value data in Python?"

    document_vectors = embed_texts(client, documents)
    query_vector = embed_texts(client, [query])[0]

    scored_documents = []
    for document, vector in zip(documents, document_vectors, strict=True):
        score = cosine_similarity(query_vector, vector)
        scored_documents.append((score, document))

    scored_documents.sort(reverse=True, key=lambda item: item[0])

    print(f"Query: {query}\n")
    print("Top matches:")
    for score, document in scored_documents[:3]:
        print(f"- score={score:.4f} | {document}")


def run_duplicate_detection_scenario(client: OpenAI) -> None:
    """Detect texts that are likely near-duplicates."""

    print_section("Scenario 3 - Near-duplicate detection")

    texts = [
        "Install the dependencies with uv sync before running the labs.",
        "Before running the labs, install dependencies with uv sync.",
        "Paris is a popular destination for spring travel.",
        "Use Ruff to lint the Python files in this project.",
    ]

    vectors = embed_texts(client, texts)
    threshold = 0.92

    print("Pairs above the near-duplicate threshold:")
    found_match = False

    for left_index in range(len(texts)):
        for right_index in range(left_index + 1, len(texts)):
            score = cosine_similarity(vectors[left_index], vectors[right_index])
            if score < threshold:
                continue

            found_match = True
            print(f"- texts {left_index + 1} and {right_index + 1}: score={score:.4f}")

    if not found_match:
        print("- No pairs crossed the threshold in this run.")


def main() -> None:
    """Run all embedding scenarios."""

    require_api_key()
    client = OpenAI()

    run_similarity_scenario(client)
    run_semantic_search_scenario(client)
    run_duplicate_detection_scenario(client)


if __name__ == "__main__":
    main()

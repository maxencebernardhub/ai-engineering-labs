"""Test script for gemini_client.py helpers."""

import sys
from pathlib import Path

# Add the scripts directory to the path so we can import gemini_client
sys.path.append(str(Path(__file__).parent))

import gemini_client
from pydantic import BaseModel, Field


class Hero(BaseModel):
    name: str
    power: str
    strength_level: int = Field(description="Level from 1 to 100")


def test_all():
    print("--- Testing gemini_client helpers ---")

    # 1. Test generate_text
    print("\n1. Testing generate_text...")
    text = gemini_client.generate_text(
        prompt="Say 'Hello World' in Latin.",
        system_instruction="You are a helpful assistant."
    )
    print(f"Response: {text}")

    # 2. Test generate_structured
    print("\n2. Testing generate_structured...")
    hero = gemini_client.generate_structured(
        prompt="Create a superhero named 'Python Man'.",
        output_model=Hero
    )
    print(f"Hero created: {hero}")

    # 3. Test stream_response
    print("\n3. Testing stream_response...")
    print("Streaming: ", end="", flush=True)
    for chunk in gemini_client.stream_response(
        prompt="Count from 1 to 5 in French, one word per line."
    ):
        print(chunk, end="", flush=True)
    print("\nStream finished.")

    # 4. Test generate_embedding
    print("\n4. Testing generate_embedding...")
    vector = gemini_client.generate_embedding("Gemini is powerful.")
    print(f"Embedding length: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")

    # 5. Test generate_embeddings (Batch)
    print("\n5. Testing generate_embeddings (Batch)...")
    texts = ["Apple", "Orange", "Banana"]
    vectors = gemini_client.generate_embeddings(texts)
    print(f"Received {len(vectors)} embeddings.")
    for i, txt in enumerate(texts):
        print(f" - {txt}: {len(vectors[i])} dimensions")

    print("\n--- All tests completed successfully! ---")


if __name__ == "__main__":
    try:
        test_all()
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        sys.exit(1)

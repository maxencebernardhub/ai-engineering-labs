import pytest

from utils.helpers import chunk_text


def test_chunk_text_empty_string():
    assert chunk_text("") == []


def test_chunk_text_whitespace_only():
    assert chunk_text("   ") == []


def test_chunk_text_shorter_than_chunk_size():
    text = "Hello world, this is a short sentence."
    result = chunk_text(text, chunk_size=512, overlap=64)
    assert result == [text]


def test_chunk_text_produces_multiple_chunks():
    text = " ".join(["word"] * 300)
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1


def test_chunk_text_each_chunk_respects_size():
    text = " ".join([f"w{i}" for i in range(500)])
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    for chunk in chunks:
        assert len(chunk.split()) <= 100


def test_chunk_text_overlap_shared_words():
    words = [f"word{i}" for i in range(300)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    # Last 20 words of chunk[0] must be the first 20 words of chunk[1]
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_chunk_text_covers_all_content():
    words = [f"w{i}" for i in range(250)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    # First word must be in first chunk, last word in last chunk
    assert "w0" in chunks[0]
    assert "w249" in chunks[-1]


def test_chunk_text_invalid_overlap_raises():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("some text", chunk_size=100, overlap=100)


def test_chunk_text_overlap_greater_than_size_raises():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("some text", chunk_size=100, overlap=150)

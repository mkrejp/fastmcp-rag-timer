"""Tests for document extraction and chunking."""

from __future__ import annotations

from processing import chunk_text, is_supported_filename, process_document


def test_supported_filenames() -> None:
    """PDF and common text extensions should be accepted."""
    assert is_supported_filename("doc.pdf")
    assert is_supported_filename("notes.txt")
    assert is_supported_filename("readme.md")
    assert not is_supported_filename("image.png")


def test_process_text_document() -> None:
    """Plain text should yield fulltext and at least one chunk."""
    fulltext, chunks = process_document("hello.txt", b"alpha beta gamma")
    assert "alpha beta gamma" in fulltext
    assert len(chunks) >= 1
    assert chunks[0]["text"]


def test_chunk_text_respects_size() -> None:
    """Chunking should split long strings near the requested size."""
    text = "x" * 2500
    chunks = chunk_text(text, chunk_size=1000)
    assert len(chunks) == 3
    assert all(len(c["text"]) <= 1000 for c in chunks)

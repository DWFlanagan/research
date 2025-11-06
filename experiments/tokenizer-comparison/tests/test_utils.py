"""Tests for utility functions."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    clean_text,
    extract_sentences,
    normalize_text,
    select_representative_sentences,
)


def test_normalize_text():
    """Test text normalization."""
    # Test with some Unicode characters
    text = "Café résumé naïve"
    normalized = normalize_text(text)
    
    # Should still contain the text
    assert len(normalized) > 0
    assert "Caf" in normalized or "Café" in normalized


def test_clean_text():
    """Test text cleaning."""
    text = "Hello    world\n\n\n\nwith   spaces"
    cleaned = clean_text(text)
    
    # Should reduce excessive whitespace
    assert "    " not in cleaned
    assert "\n\n\n" not in cleaned


def test_extract_sentences():
    """Test sentence extraction."""
    text = "First sentence. Second sentence! Third sentence? Fourth."
    sentences = extract_sentences(text)
    
    assert len(sentences) >= 3
    assert any("First" in s for s in sentences)


def test_extract_sentences_with_limit():
    """Test sentence extraction respects limit."""
    text = ". ".join([f"Sentence {i}" for i in range(100)])
    sentences = extract_sentences(text, max_sentences=10)
    
    assert len(sentences) <= 10


def test_select_representative_sentences():
    """Test representative sentence selection."""
    sentences = [
        "Short.",
        "A bit longer sentence.",
        "This is a medium length sentence with more words.",
        "Very very very long sentence with many many words to test.",
    ]
    
    selected = select_representative_sentences(sentences, num_examples=2)
    
    assert len(selected) == 2
    # Should select different lengths
    assert len(selected[0]) != len(selected[1])


def test_select_representative_sentences_fewer_than_requested():
    """Test when fewer sentences than requested."""
    sentences = ["One", "Two"]
    selected = select_representative_sentences(sentences, num_examples=10)
    
    assert len(selected) == 2
    assert selected == sentences


def test_normalize_text_idempotent():
    """Test that normalizing twice gives same result."""
    text = "Test text with some unicode: café"
    norm1 = normalize_text(text)
    norm2 = normalize_text(norm1)
    
    assert norm1 == norm2

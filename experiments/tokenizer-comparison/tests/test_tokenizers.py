"""Tests for tokenizer functionality."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer_wrapper import CharacterTokenizer, WhitespaceTokenizer, get_tokenizer


def test_character_tokenizer_roundtrip():
    """Test character tokenizer roundtrip."""
    tokenizer = CharacterTokenizer()
    text = "Hello, world!"
    
    tokens = tokenizer.tokenize(text)
    detokenized = tokenizer.detokenize(tokens)
    
    assert detokenized == text
    assert len(tokens) == len(text)


def test_whitespace_tokenizer_roundtrip():
    """Test whitespace tokenizer roundtrip."""
    tokenizer = WhitespaceTokenizer()
    text = "Hello world from tokenizer"
    
    tokens = tokenizer.tokenize(text)
    detokenized = tokenizer.detokenize(tokens)
    
    assert detokenized == text
    assert len(tokens) == 4


def test_get_tokenizer_character():
    """Test factory function for character tokenizer."""
    tokenizer = get_tokenizer("character")
    assert isinstance(tokenizer, CharacterTokenizer)
    assert tokenizer.name == "character"


def test_get_tokenizer_whitespace():
    """Test factory function for whitespace tokenizer."""
    tokenizer = get_tokenizer("whitespace")
    assert isinstance(tokenizer, WhitespaceTokenizer)
    assert tokenizer.name == "whitespace"


@pytest.mark.parametrize("tokenizer_key", [
    "character",
    "whitespace",
])
def test_tokenizer_consistency(tokenizer_key):
    """Test tokenizers produce consistent results."""
    tokenizer = get_tokenizer(tokenizer_key)
    text = "The quick brown fox jumps over the lazy dog."
    
    # Tokenize twice
    tokens1 = tokenizer.tokenize(text)
    tokens2 = tokenizer.tokenize(text)
    
    assert tokens1 == tokens2
    assert len(tokens1) > 0


def test_tokenizer_empty_string():
    """Test tokenizers handle empty strings."""
    tokenizer = CharacterTokenizer()
    
    tokens = tokenizer.tokenize("")
    assert tokens == []
    
    detokenized = tokenizer.detokenize([])
    assert detokenized == ""


def test_smoke_tokenize_abstract():
    """Smoke test: tokenize a sample abstract."""
    # Sample abstract text
    abstract = """
    We introduce the Transformer, a novel architecture based solely on attention mechanisms,
    dispensing with recurrence and convolutions entirely. Experiments on machine translation
    tasks show that the Transformer can be trained significantly faster while achieving 
    better performance.
    """
    
    # Test with simple tokenizers (don't require model downloads in tests)
    char_tokenizer = get_tokenizer("character")
    ws_tokenizer = get_tokenizer("whitespace")
    
    # Character tokenizer
    char_tokens = char_tokenizer.tokenize(abstract)
    assert len(char_tokens) > 100  # Should have many characters
    
    char_detokenized = char_tokenizer.detokenize(char_tokens)
    assert char_detokenized == abstract
    
    # Whitespace tokenizer
    ws_tokens = ws_tokenizer.tokenize(abstract)
    assert len(ws_tokens) > 20  # Should have many words
    assert len(ws_tokens) < len(char_tokens)  # Fewer than characters
    
    # Token counts should be deterministic
    assert char_tokenizer.count_tokens(abstract) == len(char_tokens)
    assert ws_tokenizer.count_tokens(abstract) == len(ws_tokens)

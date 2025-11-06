"""Tokenizer wrapper classes for unified interface."""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TokenizerWrapper(ABC):
    """Base class for tokenizer wrappers."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into tokens."""
        pass
    
    @abstractmethod
    def detokenize(self, tokens: list[str]) -> str:
        """Convert tokens back to text."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass
    
    def save_artifacts(self, output_dir: Path) -> None:
        """Save tokenizer artifacts for reproducibility."""
        pass
    
    def get_info(self) -> dict[str, Any]:
        """Get tokenizer information."""
        return {"name": self.name, "type": self.__class__.__name__}


class HuggingFaceTokenizer(TokenizerWrapper):
    """Wrapper for Hugging Face tokenizers."""
    
    def __init__(self, model_name: str, name: Optional[str] = None):
        super().__init__(name or model_name)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Loaded HF tokenizer: {model_name}")
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text."""
        return self.tokenizer.tokenize(text)
    
    def detokenize(self, tokens: list[str]) -> str:
        """Detokenize tokens."""
        # Convert tokens to IDs then decode
        try:
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Detokenization failed: {e}")
            return " ".join(tokens)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def save_artifacts(self, output_dir: Path) -> None:
        """Save tokenizer artifacts."""
        save_path = output_dir / self.name.replace("/", "_")
        save_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Saved tokenizer artifacts to {save_path}")
    
    def get_info(self) -> dict[str, Any]:
        """Get tokenizer information."""
        info = super().get_info()
        info.update({
            "model_name": self.model_name,
            "vocab_size": self.tokenizer.vocab_size,
        })
        return info


class CharacterTokenizer(TokenizerWrapper):
    """Simple character-level tokenizer."""
    
    def __init__(self):
        super().__init__("character")
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize into characters."""
        return list(text)
    
    def detokenize(self, tokens: list[str]) -> str:
        """Join characters."""
        return "".join(tokens)
    
    def count_tokens(self, text: str) -> int:
        """Count characters."""
        return len(text)
    
    def get_info(self) -> dict[str, Any]:
        """Get tokenizer information."""
        info = super().get_info()
        info["description"] = "Character-level tokenization"
        return info


class WhitespaceTokenizer(TokenizerWrapper):
    """Simple whitespace tokenizer."""
    
    def __init__(self):
        super().__init__("whitespace")
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize by whitespace."""
        return text.split()
    
    def detokenize(self, tokens: list[str]) -> str:
        """Join with spaces."""
        return " ".join(tokens)
    
    def count_tokens(self, text: str) -> int:
        """Count whitespace-separated tokens."""
        return len(text.split())
    
    def get_info(self) -> dict[str, Any]:
        """Get tokenizer information."""
        info = super().get_info()
        info["description"] = "Whitespace-based tokenization"
        return info


def get_tokenizer(tokenizer_key: str) -> TokenizerWrapper:
    """
    Factory function to get tokenizer by key.
    
    Args:
        tokenizer_key: One of: gpt2, bert-base-uncased, t5-small, 
                       google/byt5-small, character, whitespace
    
    Returns:
        TokenizerWrapper instance
    """
    if tokenizer_key == "character":
        return CharacterTokenizer()
    elif tokenizer_key == "whitespace":
        return WhitespaceTokenizer()
    else:
        # Hugging Face tokenizer
        return HuggingFaceTokenizer(tokenizer_key)


def measure_throughput(tokenizer: TokenizerWrapper, text: str, num_runs: int = 3) -> float:
    """
    Measure tokenization throughput.
    
    Args:
        tokenizer: Tokenizer to measure
        text: Text to tokenize
        num_runs: Number of runs for averaging
    
    Returns:
        Tokens per second
    """
    times = []
    
    for _ in range(num_runs):
        start = time.perf_counter()
        tokens = tokenizer.tokenize(text)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    token_count = len(tokens)
    
    if avg_time > 0:
        return token_count / avg_time
    return 0.0

"""Evaluation functions for tokenizer comparison."""

import logging
from collections import Counter
from typing import Any

import numpy as np

from tokenizer_wrapper import TokenizerWrapper

logger = logging.getLogger(__name__)


def calculate_edit_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        Edit distance
    """
    if len(s1) < len(s2):
        return calculate_edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def evaluate_tokenizer(
    tokenizer: TokenizerWrapper,
    text: str,
    sentences: list[str]
) -> dict[str, Any]:
    """
    Evaluate a tokenizer on given text.
    
    Args:
        tokenizer: Tokenizer to evaluate
        text: Full text to evaluate
        sentences: List of sentences for detailed analysis
    
    Returns:
        Dictionary of evaluation metrics
    """
    results = {
        "tokenizer": tokenizer.name,
        "tokenizer_info": tokenizer.get_info(),
    }
    
    # Basic counts
    tokens = tokenizer.tokenize(text)
    results["total_tokens"] = len(tokens)
    results["total_chars"] = len(text)
    results["total_words"] = len(text.split())
    
    # Efficiency metrics
    if results["total_chars"] > 0:
        results["tokens_per_1000_chars"] = (results["total_tokens"] / results["total_chars"]) * 1000
    else:
        results["tokens_per_1000_chars"] = 0
    
    if results["total_words"] > 0:
        results["tokens_per_100_words"] = (results["total_tokens"] / results["total_words"]) * 100
    else:
        results["tokens_per_100_words"] = 0
    
    # Token length distribution
    token_lengths = [len(token) for token in tokens]
    results["avg_token_length"] = np.mean(token_lengths) if token_lengths else 0
    results["median_token_length"] = np.median(token_lengths) if token_lengths else 0
    results["min_token_length"] = min(token_lengths) if token_lengths else 0
    results["max_token_length"] = max(token_lengths) if token_lengths else 0
    
    # Analyze words vs tokens
    words = text.split()
    word_token_ratios = []
    for word in words[:1000]:  # Sample first 1000 words
        word_tokens = tokenizer.tokenize(word)
        if len(word) > 0:
            word_token_ratios.append(len(word_tokens))
    
    if word_token_ratios:
        results["avg_tokens_per_word"] = np.mean(word_token_ratios)
        results["median_tokens_per_word"] = np.median(word_token_ratios)
    else:
        results["avg_tokens_per_word"] = 0
        results["median_tokens_per_word"] = 0
    
    # Reconstruction fidelity
    detokenized = tokenizer.detokenize(tokens)
    results["exact_match"] = (detokenized == text)
    results["reconstruction_edit_distance"] = calculate_edit_distance(text[:1000], detokenized[:1000])
    results["reconstruction_similarity"] = 1.0 - (results["reconstruction_edit_distance"] / max(len(text[:1000]), 1))
    
    # Unknown/special tokens (for HF tokenizers)
    if hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'unk_token'):
        unk_token = tokenizer.tokenizer.unk_token
        if unk_token:
            unk_count = tokens.count(unk_token)
            results["unknown_token_count"] = unk_count
            results["unknown_token_rate"] = unk_count / len(tokens) if tokens else 0
        else:
            results["unknown_token_count"] = 0
            results["unknown_token_rate"] = 0
    else:
        results["unknown_token_count"] = 0
        results["unknown_token_rate"] = 0
    
    # Token frequency analysis
    token_freq = Counter(tokens)
    total = len(tokens)
    rare_threshold = max(1, total // 1000)  # Tokens appearing less than 0.1%
    rare_tokens = sum(1 for count in token_freq.values() if count <= rare_threshold)
    results["unique_tokens"] = len(token_freq)
    results["rare_token_types"] = rare_tokens
    results["rare_token_rate"] = rare_tokens / len(token_freq) if token_freq else 0
    
    # Sentence examples
    sentence_examples = []
    for sent in sentences[:10]:  # First 10 sentences
        sent_tokens = tokenizer.tokenize(sent)
        sentence_examples.append({
            "text": sent,
            "tokens": sent_tokens,
            "token_count": len(sent_tokens),
            "char_count": len(sent),
        })
    results["sentence_examples"] = sentence_examples
    
    return results

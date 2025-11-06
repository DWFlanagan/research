"""Utility functions for PDF extraction and text processing."""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional, Tuple

import httpx
import pymupdf  # PyMuPDF/fitz
from pdfminer.high_level import extract_text as pdfminer_extract

logger = logging.getLogger(__name__)


def normalize_text(text: str, form: str = "NFKC") -> str:
    """
    Normalize text using Unicode normalization.
    
    Args:
        text: Input text to normalize
        form: Normalization form (default: NFKC)
    
    Returns:
        Normalized text
    """
    return unicodedata.normalize(form, text)


def download_pdf(url: str, output_path: Path, timeout: int = 60) -> bool:
    """
    Download a PDF from a URL.
    
    Args:
        url: URL to download from
        output_path: Path to save PDF
        timeout: Timeout in seconds
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            logger.info(f"Downloaded PDF from {url} to {output_path}")
            return True
    except Exception as e:
        logger.error(f"Failed to download PDF from {url}: {e}")
        return False


def extract_text_pymupdf(pdf_path: Path) -> Optional[str]:
    """
    Extract text from PDF using PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Extracted text or None if failed
    """
    try:
        doc = pymupdf.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        text = "\n".join(text_parts)
        logger.info(f"Extracted {len(text)} chars using PyMuPDF from {pdf_path.name}")
        return text
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
        return None


def extract_text_pdfminer(pdf_path: Path) -> Optional[str]:
    """
    Extract text from PDF using pdfminer.six.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Extracted text or None if failed
    """
    try:
        text = pdfminer_extract(str(pdf_path))
        logger.info(f"Extracted {len(text)} chars using pdfminer from {pdf_path.name}")
        return text
    except Exception as e:
        logger.error(f"pdfminer extraction failed for {pdf_path}: {e}")
        return None


def extract_text_from_pdf(pdf_path: Path) -> Tuple[Optional[str], str]:
    """
    Extract text from PDF using PyMuPDF with pdfminer fallback.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Tuple of (extracted_text, method_used)
    """
    # Try PyMuPDF first
    text = extract_text_pymupdf(pdf_path)
    if text and len(text.strip()) > 100:
        return text, "pymupdf"
    
    # Fallback to pdfminer
    logger.info(f"Falling back to pdfminer for {pdf_path.name}")
    text = extract_text_pdfminer(pdf_path)
    if text and len(text.strip()) > 100:
        return text, "pdfminer"
    
    return None, "none"


def get_abstract_fallback(arxiv_id: str) -> Optional[str]:
    """
    Fetch abstract as fallback when PDF extraction fails.
    
    Args:
        arxiv_id: arXiv ID (e.g., "1706.03762")
    
    Returns:
        Abstract text or None
    """
    try:
        url = f"https://arxiv.org/abs/{arxiv_id}"
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            content = response.text
            
            # Simple extraction of abstract from HTML
            match = re.search(
                r'<blockquote class="abstract mathjax">.*?<span class="descriptor">Abstract:</span>\s*(.*?)</blockquote>',
                content,
                re.DOTALL
            )
            if match:
                abstract = match.group(1).strip()
                # Clean HTML tags
                abstract = re.sub(r'<[^>]+>', '', abstract)
                abstract = re.sub(r'\s+', ' ', abstract)
                logger.info(f"Fetched abstract for {arxiv_id}: {len(abstract)} chars")
                return abstract
    except Exception as e:
        logger.error(f"Failed to fetch abstract for {arxiv_id}: {e}")
    
    return None


def clean_text(text: str) -> str:
    """
    Basic text cleaning while preserving important content.
    
    Args:
        text: Input text
    
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace but preserve paragraph structure
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\n\n+', '\n\n', text)
    return text.strip()


def extract_sentences(text: str, max_sentences: int = 100) -> list[str]:
    """
    Extract sentences from text for tokenization examples.
    
    Args:
        text: Input text
        max_sentences: Maximum sentences to extract
    
    Returns:
        List of sentences
    """
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter out very short or very long "sentences"
    sentences = [s.strip() for s in sentences if 10 < len(s) < 500]
    
    return sentences[:max_sentences]


def select_representative_sentences(
    sentences: list[str], 
    num_examples: int = 10
) -> list[str]:
    """
    Select representative sentences covering different lengths.
    
    Args:
        sentences: List of sentences
        num_examples: Number of examples to select
    
    Returns:
        List of selected sentences
    """
    if len(sentences) <= num_examples:
        return sentences
    
    # Sort by length
    sorted_sentences = sorted(sentences, key=len)
    
    # Select evenly distributed samples
    step = len(sorted_sentences) // num_examples
    selected = []
    
    for i in range(num_examples):
        idx = i * step
        if idx < len(sorted_sentences):
            selected.append(sorted_sentences[idx])
    
    return selected

"""Visualization functions for tokenizer comparison."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_plots(results: list[dict], output_dir: Path) -> None:
    """
    Generate visualization plots.
    
    Args:
        results: List of evaluation results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    
    # Plot 1: Tokens per 1000 characters bar chart
    plot_tokens_per_1000_chars(df, output_dir)
    
    # Plot 2: Tokens per 100 words
    plot_tokens_per_100_words(df, output_dir)
    
    # Plot 3: Reconstruction similarity
    plot_reconstruction_similarity(df, output_dir)
    
    # Plot 4: Throughput comparison
    plot_throughput(df, output_dir)
    
    # Plot 5: Token length distribution
    plot_token_length_distribution(df, output_dir)
    
    logger.info(f"Generated plots in {output_dir}")


def plot_tokens_per_1000_chars(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot tokens per 1000 characters comparison."""
    # Group by tokenizer and paper
    pivot = df.pivot(index='paper', columns='tokenizer', values='tokens_per_1000_chars')
    
    ax = pivot.plot(kind='bar', figsize=(14, 6))
    ax.set_xlabel('Paper', fontsize=12)
    ax.set_ylabel('Tokens per 1000 Characters', fontsize=12)
    ax.set_title('Token Efficiency: Tokens per 1000 Characters', fontsize=14, fontweight='bold')
    ax.legend(title='Tokenizer', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'tokens_per_1000_chars.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Generated tokens_per_1000_chars.png")


def plot_tokens_per_100_words(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot tokens per 100 words comparison."""
    # Group by tokenizer
    grouped = df.groupby('tokenizer')['tokens_per_100_words'].mean().sort_values()
    
    ax = grouped.plot(kind='barh', figsize=(10, 6), color='steelblue')
    ax.set_xlabel('Tokens per 100 Words (Average)', fontsize=12)
    ax.set_ylabel('Tokenizer', fontsize=12)
    ax.set_title('Token Density: Average Tokens per 100 Words', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tokens_per_100_words.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Generated tokens_per_100_words.png")


def plot_reconstruction_similarity(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot reconstruction similarity comparison."""
    plt.figure(figsize=(12, 6))
    
    # Box plot of reconstruction similarity by tokenizer
    tokenizers = df['tokenizer'].unique()
    data = [df[df['tokenizer'] == tok]['reconstruction_similarity'].values for tok in tokenizers]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data, labels=tokenizers, patch_artist=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_xlabel('Tokenizer', fontsize=12)
    ax.set_ylabel('Reconstruction Similarity', fontsize=12)
    ax.set_title('Reconstruction Fidelity (1.0 = perfect match)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'reconstruction_similarity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Generated reconstruction_similarity.png")


def plot_throughput(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot tokenization throughput comparison."""
    # Group by tokenizer
    grouped = df.groupby('tokenizer')['throughput_tokens_per_sec'].mean().sort_values(ascending=False)
    
    ax = grouped.plot(kind='bar', figsize=(10, 6), color='coral')
    ax.set_xlabel('Tokenizer', fontsize=12)
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
    ax.set_title('Tokenization Speed', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(grouped.values):
        ax.text(i, v + max(grouped.values) * 0.02, f'{v:.0f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Generated throughput.png")


def plot_token_length_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot token length distribution."""
    # Use average and median token lengths
    metrics = ['avg_token_length', 'median_token_length']
    
    grouped = df.groupby('tokenizer')[metrics].mean()
    
    ax = grouped.plot(kind='bar', figsize=(12, 6))
    ax.set_xlabel('Tokenizer', fontsize=12)
    ax.set_ylabel('Token Length (characters)', fontsize=12)
    ax.set_title('Token Length Statistics', fontsize=14, fontweight='bold')
    ax.legend(['Average', 'Median'])
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'token_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Generated token_length_distribution.png")

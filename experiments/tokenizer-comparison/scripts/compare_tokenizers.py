#!/usr/bin/env python3
"""CLI runner for tokenizer comparison experiment."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import evaluate_tokenizer
from tokenizer_wrapper import get_tokenizer, measure_throughput
from utils import (
    clean_text,
    download_pdf,
    extract_sentences,
    extract_text_from_pdf,
    get_abstract_fallback,
    normalize_text,
    select_representative_sentences,
)
from visualization import generate_plots

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default papers
DEFAULT_PAPERS = [
    {
        "name": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762.pdf",
        "arxiv_id": "1706.03762",
    },
    {
        "name": "BERT",
        "url": "https://arxiv.org/pdf/1810.04805.pdf",
        "arxiv_id": "1810.04805",
    },
    {
        "name": "ByT5",
        "url": "https://arxiv.org/pdf/2105.13626.pdf",
        "arxiv_id": "2105.13626",
    },
]

# Default tokenizers
DEFAULT_TOKENIZERS = [
    "gpt2",
    "bert-base-uncased",
    "t5-small",
    "google/byt5-small",
    "character",
    "whitespace",
]


def process_paper(paper_info: dict, output_dir: Path) -> dict:
    """
    Download and process a single paper.
    
    Args:
        paper_info: Dictionary with name, url, arxiv_id
        output_dir: Output directory
    
    Returns:
        Dictionary with paper name and extracted text
    """
    logger.info(f"Processing paper: {paper_info['name']}")
    
    # Download PDF
    pdf_path = output_dir / "pdfs" / f"{paper_info['arxiv_id']}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    text = None
    method = "none"
    
    if not pdf_path.exists():
        if download_pdf(paper_info['url'], pdf_path):
            text, method = extract_text_from_pdf(pdf_path)
    else:
        logger.info(f"PDF already exists: {pdf_path}")
        text, method = extract_text_from_pdf(pdf_path)
    
    # Fallback to abstract if PDF extraction failed
    if not text or len(text.strip()) < 100:
        logger.warning(f"PDF extraction failed, using abstract fallback for {paper_info['name']}")
        text = get_abstract_fallback(paper_info['arxiv_id'])
        method = "abstract_fallback"
    
    if not text:
        logger.error(f"Failed to extract text for {paper_info['name']}")
        return None
    
    # Save raw text
    raw_path = output_dir / "results" / "raw" / f"{paper_info['arxiv_id']}_raw.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(text, encoding='utf-8')
    
    # Clean and normalize
    text = clean_text(text)
    normalized_text = normalize_text(text)
    
    # Save normalized text
    norm_path = output_dir / "results" / "normalized" / f"{paper_info['arxiv_id']}_normalized.txt"
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    norm_path.write_text(normalized_text, encoding='utf-8')
    
    logger.info(f"Extracted {len(normalized_text)} chars from {paper_info['name']} using {method}")
    
    return {
        "name": paper_info['name'],
        "arxiv_id": paper_info['arxiv_id'],
        "text": normalized_text,
        "extraction_method": method,
    }


def run_comparison(
    papers: list[dict],
    tokenizers: list[str],
    output_dir: Path
) -> dict:
    """
    Run full tokenizer comparison experiment.
    
    Args:
        papers: List of paper info dictionaries
        tokenizers: List of tokenizer keys
        output_dir: Output directory
    
    Returns:
        Results dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process papers
    logger.info(f"Processing {len(papers)} papers...")
    processed_papers = []
    for paper_info in papers:
        result = process_paper(paper_info, output_dir)
        if result:
            processed_papers.append(result)
    
    if not processed_papers:
        logger.error("No papers were successfully processed")
        return {}
    
    # Initialize tokenizers and save artifacts
    logger.info(f"Initializing {len(tokenizers)} tokenizers...")
    tokenizer_objects = {}
    artifacts_dir = output_dir / "tokenizer_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    for tok_key in tokenizers:
        try:
            tokenizer = get_tokenizer(tok_key)
            tokenizer_objects[tok_key] = tokenizer
            tokenizer.save_artifacts(artifacts_dir)
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer {tok_key}: {e}")
    
    # Run evaluations
    logger.info("Running evaluations...")
    all_results = []
    sentence_examples_all = {}
    
    for paper in processed_papers:
        paper_name = paper['name']
        text = paper['text']
        
        # Extract sentences for examples
        sentences = extract_sentences(text)
        representative_sentences = select_representative_sentences(sentences, 10)
        
        logger.info(f"Evaluating {paper_name} with {len(tokenizer_objects)} tokenizers...")
        
        for tok_key, tokenizer in tokenizer_objects.items():
            try:
                # Run evaluation
                eval_results = evaluate_tokenizer(tokenizer, text, representative_sentences)
                
                # Measure throughput
                throughput = measure_throughput(tokenizer, text[:5000])  # First 5000 chars
                eval_results["throughput_tokens_per_sec"] = throughput
                
                # Add paper info
                eval_results["paper"] = paper_name
                eval_results["arxiv_id"] = paper['arxiv_id']
                
                all_results.append(eval_results)
                
                # Store sentence examples
                key = f"{paper['arxiv_id']}_{tok_key}"
                sentence_examples_all[key] = eval_results.pop("sentence_examples")
                
                logger.info(f"  {tok_key}: {eval_results['total_tokens']} tokens")
                
            except Exception as e:
                logger.error(f"Evaluation failed for {paper_name} + {tok_key}: {e}")
    
    # Save results
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV summary
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = results_dir / "summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary to {csv_path}")
    
    # Save JSON results
    json_path = results_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved detailed results to {json_path}")
    
    # Save sentence examples
    examples_path = results_dir / "sentence_examples.json"
    with open(examples_path, 'w') as f:
        json.dump(sentence_examples_all, f, indent=2)
    logger.info(f"Saved sentence examples to {examples_path}")
    
    # Generate plots
    if all_results:
        logger.info("Generating plots...")
        try:
            generate_plots(all_results, results_dir)
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
    
    # Generate report
    logger.info("Generating report...")
    try:
        generate_report(all_results, results_dir)
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
    
    return {
        "papers_processed": len(processed_papers),
        "tokenizers_evaluated": len(tokenizer_objects),
        "total_evaluations": len(all_results),
        "output_dir": str(output_dir),
    }


def generate_report(results: list[dict], output_dir: Path) -> None:
    """Generate markdown report."""
    report_path = output_dir / "report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Tokenizer Comparison Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report compares various tokenization algorithms on scientific papers from arXiv.\n\n")
        
        # Create summary table
        f.write("## Results Summary\n\n")
        
        df = pd.DataFrame(results)
        
        # Group by tokenizer
        f.write("### Tokens per 1000 Characters by Tokenizer\n\n")
        summary = df.groupby('tokenizer')['tokens_per_1000_chars'].agg(['mean', 'std', 'min', 'max'])
        f.write(summary.to_markdown())
        f.write("\n\n")
        
        f.write("### Average Tokens per 100 Words by Tokenizer\n\n")
        summary = df.groupby('tokenizer')['tokens_per_100_words'].agg(['mean', 'std'])
        f.write(summary.to_markdown())
        f.write("\n\n")
        
        f.write("### Reconstruction Fidelity\n\n")
        summary = df.groupby('tokenizer')['reconstruction_similarity'].agg(['mean', 'min', 'max'])
        f.write(summary.to_markdown())
        f.write("\n\n")
        
        f.write("### Throughput (tokens/sec)\n\n")
        summary = df.groupby('tokenizer')['throughput_tokens_per_sec'].agg(['mean', 'std'])
        f.write(summary.to_markdown())
        f.write("\n\n")
        
        f.write("## Recommendations for Research Publishing\n\n")
        f.write("### Token Efficiency\n\n")
        
        best_efficiency = df.loc[df['tokens_per_1000_chars'].idxmin()]
        f.write(f"- **Most efficient**: {best_efficiency['tokenizer']} ")
        f.write(f"({best_efficiency['tokens_per_1000_chars']:.1f} tokens/1000 chars)\n")
        
        f.write("\n### Reconstruction Quality\n\n")
        best_reconstruction = df.loc[df['reconstruction_similarity'].idxmax()]
        f.write(f"- **Best reconstruction**: {best_reconstruction['tokenizer']} ")
        f.write(f"({best_reconstruction['reconstruction_similarity']:.2%} similarity)\n")
        
        f.write("\n### Performance\n\n")
        best_throughput = df.loc[df['throughput_tokens_per_sec'].idxmax()]
        f.write(f"- **Fastest**: {best_throughput['tokenizer']} ")
        f.write(f"({best_throughput['throughput_tokens_per_sec']:.0f} tokens/sec)\n")
        
        f.write("\n## Reproducibility Information\n\n")
        f.write("- All tokenizer artifacts saved in `tokenizer_artifacts/`\n")
        f.write("- Raw extracted text in `results/raw/`\n")
        f.write("- Normalized text in `results/normalized/`\n")
        f.write("- Detailed metrics in `results.json`\n")
        f.write("- Sentence examples in `sentence_examples.json`\n")
    
    logger.info(f"Report saved to {report_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare tokenization algorithms on scientific papers"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--papers",
        type=str,
        nargs="+",
        help="arXiv URLs or IDs (default: Attention, BERT, ByT5 papers)",
    )
    parser.add_argument(
        "--tokenizers",
        type=str,
        nargs="+",
        choices=["gpt2", "bert-base-uncased", "t5-small", "google/byt5-small", "character", "whitespace"],
        help="Tokenizers to compare (default: all)",
    )
    
    args = parser.parse_args()
    
    # Use defaults if not specified
    papers = DEFAULT_PAPERS
    tokenizers = args.tokenizers if args.tokenizers else DEFAULT_TOKENIZERS
    
    logger.info("Starting tokenizer comparison experiment")
    logger.info(f"Papers: {len(papers)}")
    logger.info(f"Tokenizers: {tokenizers}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        results = run_comparison(papers, tokenizers, args.output_dir)
        logger.info("Experiment completed successfully!")
        logger.info(f"Results: {results}")
        return 0
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Demo script that runs the comparison with local sample text."""

import sys
from pathlib import Path

from evaluation import evaluate_tokenizer
from tokenizer_wrapper import get_tokenizer, measure_throughput
from utils import clean_text, extract_sentences, normalize_text, select_representative_sentences
from visualization import generate_plots

# Sample paper text
SAMPLE_TEXT = """# Attention Is All You Need

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.

## Introduction

Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences. In all but a few cases, however, such attention mechanisms are used in conjunction with a recurrent network.

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

## Model Architecture

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

### Encoder

The encoder is composed of a stack of N=6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself.

### Decoder  

The decoder is also composed of a stack of N=6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization.

### Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

We call our particular attention "Scaled Dot-Product Attention". The input consists of queries and keys of dimension d_k, and values of dimension d_v. We compute the dot products of the query with all keys, divide each by sqrt(d_k), and apply a softmax function to obtain the weights on the values.

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V. We compute the matrix of outputs as:

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

## Experiments

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. We also trained on the larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary.

### Results

On the WMT 2014 English-to-German translation task, the big Transformer model outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model.

## Conclusion

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. We are excited about the future of attention-based models and plan to apply them to other tasks.
"""


def main():
    """Run demo comparison."""
    print("=" * 70)
    print("Tokenizer Comparison Demo")
    print("=" * 70)
    
    # Normalize text
    text = clean_text(SAMPLE_TEXT)
    text = normalize_text(text)
    
    print(f"\nSample text length: {len(text)} characters, {len(text.split())} words")
    
    # Extract sentences
    sentences = extract_sentences(text)
    representative = select_representative_sentences(sentences, 5)
    
    # Test with simple tokenizers
    tokenizers = ["character", "whitespace"]
    
    results = []
    
    for tok_key in tokenizers:
        print(f"\n{'-' * 70}")
        print(f"Testing: {tok_key}")
        print(f"{'-' * 70}")
        
        tokenizer = get_tokenizer(tok_key)
        
        # Evaluate
        eval_results = evaluate_tokenizer(tokenizer, text, representative)
        
        # Measure throughput
        throughput = measure_throughput(tokenizer, text)
        eval_results["throughput_tokens_per_sec"] = throughput
        eval_results["paper"] = "Attention Is All You Need (sample)"
        eval_results["arxiv_id"] = "demo"
        
        results.append(eval_results)
        
        # Print summary
        print(f"Total tokens: {eval_results['total_tokens']}")
        print(f"Tokens per 1000 chars: {eval_results['tokens_per_1000_chars']:.2f}")
        print(f"Tokens per 100 words: {eval_results['tokens_per_100_words']:.2f}")
        print(f"Avg token length: {eval_results['avg_token_length']:.2f}")
        print(f"Reconstruction similarity: {eval_results['reconstruction_similarity']:.4f}")
        print(f"Throughput: {throughput:.0f} tokens/sec")
        
        print(f"\nSample tokenization:")
        example = eval_results["sentence_examples"][0]
        print(f"  Text: {example['text'][:80]}...")
        print(f"  Tokens ({len(example['tokens'])}): {example['tokens'][:10]}...")
    
    # Save outputs
    output_dir = Path("demo_output/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print(f"\n{'=' * 70}")
    print("Generating visualizations...")
    try:
        generate_plots(results, output_dir)
        print(f"✓ Plots saved to {output_dir}")
    except Exception as e:
        print(f"✗ Plot generation failed: {e}")
    
    # Save results
    import json
    import pandas as pd
    
    # Remove sentence examples for cleaner JSON
    for r in results:
        r.pop("sentence_examples", None)
    
    # CSV
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "summary.csv", index=False)
    print(f"✓ CSV saved to {output_dir / 'summary.csv'}")
    
    # JSON
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ JSON saved to {output_dir / 'results.json'}")
    
    print(f"\n{'=' * 70}")
    print("Demo completed successfully!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

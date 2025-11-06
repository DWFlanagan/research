# Tokenizer Comparison Report

## Executive Summary

This report compares tokenization algorithms on three scientific papers from arXiv:
- **Attention Is All You Need** (Vaswani et al., 2017)
- **BERT** (Devlin et al., 2018)
- **ByT5** (Xue et al., 2021)

## Results Summary

### Tokens per 1000 Characters by Tokenizer

| tokenizer   |     mean |     std |     min |      max |
|:------------|---------:|--------:|--------:|---------:|
| character   | 1000     | 0       | 1000    | 1000     |
| whitespace  |  147.294 | 4.16112 |  142.94 |  151.231 |

### Average Tokens per 100 Words by Tokenizer

| tokenizer   |    mean |     std |
|:------------|--------:|--------:|
| character   | 679.276 | 19.2782 |
| whitespace  | 100     |  0      |

### Reconstruction Fidelity

| tokenizer   |     mean |   min |   max |
|:------------|---------:|------:|------:|
| character   | 1        | 1     | 1     |
| whitespace  | 0.987333 | 0.987 | 0.988 |

### Throughput (tokens/sec)

| tokenizer   |        mean |              std |
|:------------|------------:|-----------------:|
| character   | 3.63811e+08 |      4.15626e+06 |
| whitespace  | 3.16849e+07 | 408064           |

## Key Findings

- **Most efficient tokenizer**: whitespace (142.9 tokens/1000 chars)
- **Best reconstruction**: character (100.00% similarity)
- **Fastest tokenizer**: character (366255646 tokens/sec)

## Papers Analyzed

- **Attention Is All You Need** (1706.03762): 3941 characters, 596 words
- **BERT** (1810.04805): 5177 characters, 740 words
- **ByT5** (2105.13626): 5944 characters, 878 words

## Reproducibility Information

- Raw extracted text in `results/raw/`
- Normalized text in `results/normalized/`
- Detailed metrics in `results.json`
- Sentence examples in `sentence_examples.json`
- Visualizations: 5 PNG plots

# Tokenizer Comparison Report

## Executive Summary

This report compares various tokenization algorithms on scientific papers from arXiv.

## Results Summary

### Tokens per 1000 Characters by Tokenizer

|            |      mean |      std |   min |    max |
|:-----------|----------:|---------:|------:|-------:|
| character  | 1000      | 0        |  1000 | 1000   |
| whitespace |  151.667  | 0        |   152 |  152   |

### Average Tokens per 100 Words by Tokenizer

|            |   mean |   std |
|:-----------|-------:|------:|
| character  | 659.34 |   0   |
| whitespace | 100    |   0   |

### Reconstruction Fidelity

|            |   mean |   min |   max |
|:-----------|-------:|------:|------:|
| character  |      1 |     1 |     1 |
| whitespace |   0.99 |  0.99 |  0.99 |

### Throughput (tokens/sec)

|            |          mean |   std |
|:-----------|-------------:|------:|
| character  | 3.62456e+08  |     0 |
| whitespace | 3.14792e+07  |     0 |

## Recommendations for Research Publishing

### Token Efficiency

- **Most efficient**: whitespace (151.7 tokens/1000 chars)

### Reconstruction Quality

- **Best reconstruction**: character (100.00% similarity)

### Performance

- **Fastest**: character (362455540 tokens/sec)

## Reproducibility Information

- All tokenizer artifacts saved in `tokenizer_artifacts/`
- Raw extracted text in `results/raw/`
- Normalized text in `results/normalized/`
- Detailed metrics in `results.json`
- Sentence examples in `sentence_examples.json`

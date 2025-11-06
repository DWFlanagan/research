# Tokenizer Comparison Experiment

A comprehensive comparison of tokenization algorithms on scientific papers from arXiv. This experiment evaluates various tokenizers (BPE, WordPiece, SentencePiece, character-level, and whitespace-based) to inform research publishing decisions regarding token accounting, multilingual robustness, and interpretability.

## Features

- **Multiple Tokenizers**: Compares GPT-2 (BPE), BERT (WordPiece), T5 (SentencePiece), ByT5 (byte-level), character-level, and whitespace tokenization
- **Scientific Papers**: Analyzes open-access arXiv papers with fallback to abstracts
- **Comprehensive Metrics**: Token counts, efficiency ratios, reconstruction fidelity, throughput, and coverage statistics
- **Reproducibility**: All tokenizer artifacts and extracted texts are saved for verification
- **API & CLI**: Both command-line and HTTP API interfaces
- **Visualizations**: Generates comparison plots and a detailed markdown report

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python dependency management.

### Install uv (if not already installed)

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install dependencies

```bash
cd experiments/tokenizer-comparison
uv sync
```

## Usage

### Command Line Interface

Run the experiment with default settings (3 papers, 6 tokenizers):

```bash
uv run python scripts/compare_tokenizers.py
```

The results will be saved in the `output/` directory by default.

#### Custom output directory

```bash
uv run python scripts/compare_tokenizers.py --output-dir my_results
```

#### Select specific tokenizers

```bash
uv run python scripts/compare_tokenizers.py --tokenizers gpt2 bert-base-uncased character
```

Available tokenizers:
- `gpt2` - GPT-2 byte-level BPE
- `bert-base-uncased` - BERT WordPiece
- `t5-small` - T5 SentencePiece (Unigram)
- `google/byt5-small` - ByT5 byte-level
- `character` - Character-level
- `whitespace` - Whitespace-based

### HTTP API

Start the FastAPI server with uvicorn (using uvloop for performance):

```bash
cd api
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Or run directly:

```bash
cd api
uv run python app.py
```

#### API Endpoints

**Health check:**
```bash
curl http://localhost:8000/health
```

**Run experiment asynchronously:**
```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "output",
    "tokenizers": ["gpt2", "bert-base-uncased"]
  }'
```

Returns a job ID. Check status with:

```bash
curl http://localhost:8000/status/{job_id}
```

**Run experiment synchronously:**
```bash
curl -X POST http://localhost:8000/run-sync \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "output"
  }'
```

This blocks until completion and returns results directly.

## Output Files

After running the experiment, the following files are generated:

```
output/
├── pdfs/                           # Downloaded PDFs
│   ├── 1706.03762.pdf
│   ├── 1810.04805.pdf
│   └── 2105.13626.pdf
├── results/
│   ├── raw/                        # Raw extracted text
│   │   ├── 1706.03762_raw.txt
│   │   ├── 1810.04805_raw.txt
│   │   └── 2105.13626_raw.txt
│   ├── normalized/                 # NFKC normalized text
│   │   ├── 1706.03762_normalized.txt
│   │   ├── 1810.04805_normalized.txt
│   │   └── 2105.13626_normalized.txt
│   ├── summary.csv                 # Metrics table (CSV)
│   ├── results.json                # Detailed results (JSON)
│   ├── sentence_examples.json      # Tokenization examples
│   ├── report.md                   # Markdown report
│   ├── tokens_per_1000_chars.png   # Token efficiency plot
│   ├── tokens_per_100_words.png    # Token density plot
│   ├── reconstruction_similarity.png # Fidelity comparison
│   ├── throughput.png              # Speed comparison
│   └── token_length_distribution.png # Token length stats
└── tokenizer_artifacts/            # Saved tokenizer files
    ├── gpt2/
    ├── bert-base-uncased/
    └── ...
```

## Metrics Explained

### Token Efficiency
- **tokens_per_1000_chars**: How many tokens per 1000 characters. Lower is more efficient.
- **tokens_per_100_words**: How many tokens per 100 words. Shows word-level granularity.

### Reconstruction Fidelity
- **exact_match**: Whether detokenized text exactly matches original (boolean).
- **reconstruction_similarity**: Character-level similarity (0-1). 1.0 = perfect reconstruction.
- **reconstruction_edit_distance**: Levenshtein distance between original and detokenized text.

### Coverage Statistics
- **unknown_token_count**: Number of unknown/OOV tokens.
- **unknown_token_rate**: Proportion of unknown tokens (0-1).
- **unique_tokens**: Number of distinct tokens in the text.
- **rare_token_types**: Number of token types appearing rarely (< 0.1% of total).

### Performance
- **throughput_tokens_per_sec**: Tokenization speed in tokens per second.

### Token Length Distribution
- **avg_token_length**: Mean token length in characters.
- **median_token_length**: Median token length in characters.
- **avg_tokens_per_word**: Average number of tokens per word.

## Testing

Run the test suite:

```bash
uv run pytest tests/ -v
```

Run specific test file:

```bash
uv run pytest tests/test_tokenizers.py -v
```

## Reproducibility

All experiments are fully reproducible:

1. **Pinned Dependencies**: `uv.lock` file locks all dependency versions
2. **Saved Artifacts**: All tokenizer files (vocab.json, merges.txt, etc.) are saved in `tokenizer_artifacts/`
3. **Raw & Normalized Text**: Both versions saved for verification
4. **Deterministic Tokenization**: No randomness in tokenization process
5. **Metadata**: All extraction methods and parameters logged in results

To reproduce results:
1. Clone the repository
2. Run `uv sync` to install exact versions
3. Run `uv run python scripts/compare_tokenizers.py`
4. Compare your outputs with the committed results

## Interpretation Guidance

### Choosing a Tokenizer

**For token efficiency** (minimizing token counts):
- Character-level and whitespace produce the fewest tokens but lose linguistic structure
- ByT5 byte-level is efficient while maintaining robustness

**For reconstruction quality**:
- Character and whitespace have perfect reconstruction
- BPE (GPT-2) and byte-level (ByT5) have excellent reconstruction
- WordPiece (BERT) may have issues with casing/special chars

**For multilingual robustness**:
- Byte-level (ByT5) and character-level handle any Unicode
- SentencePiece (T5) designed for multilingual
- WordPiece and BPE may struggle with non-Latin scripts

**For speed**:
- Whitespace and character are fastest (simple operations)
- Transformer tokenizers (GPT-2, BERT, T5) are slower but manageable

**For research publishing**:
- Use token counts to estimate API costs (OpenAI, Anthropic)
- Consider BPE or SentencePiece for balanced efficiency and quality
- Use byte-level for maximum compatibility with diverse content

## Research Appendix Checklist

For including in research papers:

- [ ] Document tokenizer versions used (in `tokenizer_artifacts/`)
- [ ] Report token counts and efficiency metrics
- [ ] Include reconstruction fidelity scores
- [ ] Note any unknown token rates
- [ ] Specify text normalization applied (NFKC)
- [ ] Reference tokenizer papers:
  - BPE: Sennrich et al. (2016)
  - WordPiece: Wu et al. (2016)
  - SentencePiece: Kudo & Richardson (2018)
  - ByT5: Xue et al. (2021)

## Default Papers Analyzed

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - arXiv: 1706.03762
   - Foundational Transformer paper

2. **BERT** (Devlin et al., 2018)
   - arXiv: 1810.04805
   - Pre-training of Deep Bidirectional Transformers

3. **ByT5** (Xue et al., 2021)
   - arXiv: 2105.13626
   - Byte-level Transformer model

## Architecture

```
tokenizer-comparison/
├── scripts/
│   └── compare_tokenizers.py      # CLI runner
├── api/
│   └── app.py                      # FastAPI application
├── tests/
│   ├── test_tokenizers.py         # Tokenizer tests
│   └── test_utils.py               # Utility tests
├── utils.py                        # PDF extraction, text processing
├── tokenizer_wrapper.py            # Unified tokenizer interface
├── evaluation.py                   # Evaluation metrics
├── visualization.py                # Plotting functions
├── pyproject.toml                  # Project config and dependencies
└── README.md                       # This file
```

## Known Limitations

- PDF extraction may fail for scanned papers (no OCR)
- Large models (GPT-2, BERT, T5, ByT5) require downloading weights (~500MB total)
- Throughput measurements are approximate and hardware-dependent
- Abstract fallback loses document structure and content

## Contributing

This is a research experiment. Feel free to:
- Add more tokenizers
- Test on different domains (beyond ML papers)
- Improve PDF extraction
- Add more evaluation metrics

## License

This experiment is part of the DWFlanagan/research repository.

## References

- Vaswani et al. (2017). "Attention Is All You Need"
- Devlin et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"
- Xue et al. (2021). "ByT5: Towards a token-free future with pre-trained byte-to-byte models"
- Sennrich et al. (2016). "Neural Machine Translation of Rare Words with Subword Units"
- Kudo & Richardson (2018). "SentencePiece: A simple and language independent approach"

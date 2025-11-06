# Quick Start Guide

Get started with the tokenizer comparison experiment in 5 minutes.

## Installation

```bash
cd experiments/tokenizer-comparison

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Run Your First Comparison

### Option 1: Demo Script (Fast)

Test the experiment with sample text (no downloads needed):

```bash
python demo.py
```

This will:
- Tokenize sample text with character and whitespace tokenizers
- Generate plots in `demo_output/results/`
- Complete in ~5 seconds

### Option 2: CLI with Real Papers

Run the full experiment with default papers:

```bash
python scripts/compare_tokenizers.py
```

Note: This requires internet access to download PDFs from arXiv.

### Option 3: API Server

Start the HTTP API:

```bash
# From the api directory
cd api
uvicorn app:app --host 0.0.0.0 --port 8000

# Or from the project root
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Then make a request:

```bash
curl http://localhost:8000/health
```

## Understanding the Output

After running, check these files:

1. **`results/summary.csv`** - Metrics table for all tokenizers
2. **`results/report.md`** - Human-readable summary with recommendations
3. **`results/*.png`** - Comparison visualizations
4. **`results/results.json`** - Full detailed metrics

## Key Metrics Explained

- **tokens_per_1000_chars**: Lower = more efficient tokenization
- **reconstruction_similarity**: Higher = better fidelity (1.0 = perfect)
- **throughput_tokens_per_sec**: Higher = faster tokenization

## Next Steps

1. Read the [full README](README.md) for detailed documentation
2. Explore the [Jupyter notebook](notebooks/analysis.ipynb) for interactive analysis
3. Check the [test suite](tests/) to understand the API

## Common Issues

**"No papers were successfully processed"**
- You're probably offline. Use `python demo.py` instead, which works offline.

**"ModuleNotFoundError: No module named 'transformers'"**
- Install dependencies: `pip install -e .` or `uv sync`

**"Port 8000 already in use"**
- Change the port: `uvicorn api.app:app --port 8001`

## Running Tests

```bash
pytest tests/ -v
```

All 15 tests should pass.

#!/bin/bash
# Complete benchmark workflow script

set -e

echo "==============================================="
echo "Python Markdown Library Comparison Benchmark"
echo "==============================================="
echo

echo "Step 1: Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo

echo "Step 2: Running feature analysis..."
python3 feature_analysis.py
echo "✓ Feature analysis complete"
echo

echo "Step 3: Running benchmarks (this may take a few minutes)..."
python3 benchmark.py
echo "✓ Benchmarks complete"
echo

echo "Step 4: Generating charts..."
python3 generate_charts.py
echo "✓ Charts generated"
echo

echo "==============================================="
echo "All done! Check README.md for the full report."
echo "==============================================="

#!/usr/bin/env python3
"""Benchmark script for comparing Python markdown libraries."""

import time
import tracemalloc
import statistics
from typing import Dict, List, Tuple, Callable
import json

# Import all markdown libraries
try:
    import cmarkgfm
    CMARKGFM_AVAILABLE = True
except ImportError:
    CMARKGFM_AVAILABLE = False
    print("Warning: cmarkgfm not available")

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    print("Warning: markdown not available")

try:
    import mistune
    MISTUNE_AVAILABLE = True
except ImportError:
    MISTUNE_AVAILABLE = False
    print("Warning: mistune not available")

try:
    import mistletoe
    from mistletoe import Document
    from mistletoe.html_renderer import HTMLRenderer
    MISTLETOE_AVAILABLE = True
except ImportError:
    MISTLETOE_AVAILABLE = False
    print("Warning: mistletoe not available")

try:
    from markdown_it import MarkdownIt
    MARKDOWN_IT_AVAILABLE = True
except ImportError:
    MARKDOWN_IT_AVAILABLE = False
    print("Warning: markdown-it-py not available")

from test_documents import ALL_DOCS


def benchmark_function(func: Callable, text: str, iterations: int = 100) -> Tuple[float, float]:
    """
    Benchmark a function for speed and memory usage.
    
    Returns: (avg_time_ms, peak_memory_kb)
    """
    # Warm up
    for _ in range(5):
        func(text)
    
    # Time benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(text)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    avg_time = statistics.mean(times)
    
    # Memory benchmark
    tracemalloc.start()
    func(text)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory = peak / 1024  # Convert to KB
    
    return avg_time, peak_memory


def benchmark_all_libraries(doc_name: str, doc_text: str, iterations: int = 100) -> Dict:
    """Benchmark all available libraries on a given document."""
    results = {}
    
    if CMARKGFM_AVAILABLE:
        print(f"  Benchmarking cmarkgfm...")
        try:
            avg_time, peak_mem = benchmark_function(
                cmarkgfm.github_flavored_markdown_to_html,
                doc_text,
                iterations
            )
            results['cmarkgfm'] = {'time': avg_time, 'memory': peak_mem}
        except Exception as e:
            print(f"    Error: {e}")
            results['cmarkgfm'] = {'time': None, 'memory': None, 'error': str(e)}
    
    if MARKDOWN_AVAILABLE:
        print(f"  Benchmarking Python-Markdown...")
        try:
            md = markdown.Markdown()
            avg_time, peak_mem = benchmark_function(
                lambda text: md.convert(text),
                doc_text,
                iterations
            )
            results['markdown'] = {'time': avg_time, 'memory': peak_mem}
        except Exception as e:
            print(f"    Error: {e}")
            results['markdown'] = {'time': None, 'memory': None, 'error': str(e)}
    
    if MISTUNE_AVAILABLE:
        print(f"  Benchmarking mistune...")
        try:
            avg_time, peak_mem = benchmark_function(
                mistune.html,
                doc_text,
                iterations
            )
            results['mistune'] = {'time': avg_time, 'memory': peak_mem}
        except Exception as e:
            print(f"    Error: {e}")
            results['mistune'] = {'time': None, 'memory': None, 'error': str(e)}
    
    if MISTLETOE_AVAILABLE:
        print(f"  Benchmarking mistletoe...")
        try:
            def mistletoe_render(text):
                with HTMLRenderer() as renderer:
                    return renderer.render(Document(text))
            
            avg_time, peak_mem = benchmark_function(
                mistletoe_render,
                doc_text,
                iterations
            )
            results['mistletoe'] = {'time': avg_time, 'memory': peak_mem}
        except Exception as e:
            print(f"    Error: {e}")
            results['mistletoe'] = {'time': None, 'memory': None, 'error': str(e)}
    
    if MARKDOWN_IT_AVAILABLE:
        print(f"  Benchmarking markdown-it-py...")
        try:
            mdit = MarkdownIt()
            avg_time, peak_mem = benchmark_function(
                lambda text: mdit.render(text),
                doc_text,
                iterations
            )
            results['markdown-it-py'] = {'time': avg_time, 'memory': peak_mem}
        except Exception as e:
            print(f"    Error: {e}")
            results['markdown-it-py'] = {'time': None, 'memory': None, 'error': str(e)}
    
    return results


def main():
    """Run all benchmarks and save results."""
    print("Starting markdown library benchmarks...")
    print("=" * 60)
    
    all_results = {}
    
    for doc_name, doc_text in ALL_DOCS.items():
        print(f"\nBenchmarking with {doc_name} document...")
        results = benchmark_all_libraries(doc_name, doc_text, iterations=100)
        all_results[doc_name] = results
        
        # Print summary
        print(f"\n  Results for {doc_name}:")
        for lib_name, data in results.items():
            if data.get('time') is not None:
                print(f"    {lib_name}: {data['time']:.3f}ms, {data['memory']:.2f}KB")
            else:
                print(f"    {lib_name}: ERROR - {data.get('error', 'Unknown')}")
    
    # Save results to JSON
    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Benchmarks complete! Results saved to benchmark_results.json")
    
    return all_results


if __name__ == '__main__':
    main()

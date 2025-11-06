#!/usr/bin/env python3
"""Generate charts from benchmark results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results():
    """Load benchmark results from JSON file."""
    with open('benchmark_results.json', 'r') as f:
        return json.load(f)


def create_performance_chart(results):
    """Create a bar chart comparing parsing performance."""
    # Extract data for medium document (most representative)
    doc_type = 'medium'
    if doc_type not in results:
        doc_type = list(results.keys())[0]
    
    data = results[doc_type]
    
    libraries = []
    times = []
    
    for lib, metrics in sorted(data.items()):
        if metrics.get('time') is not None:
            libraries.append(lib)
            times.append(metrics['time'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = ax.bar(libraries, times, color=colors[:len(libraries)])
    
    ax.set_xlabel('Library', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (milliseconds)', fontsize=12, fontweight='bold')
    ax.set_title(f'Markdown Parsing Performance ({doc_type.title()} Document)', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig('performance_chart.png', dpi=300, bbox_inches='tight')
    print("Saved performance_chart.png")
    plt.close()


def create_memory_chart(results):
    """Create a bar chart comparing memory usage."""
    doc_type = 'medium'
    if doc_type not in results:
        doc_type = list(results.keys())[0]
    
    data = results[doc_type]
    
    libraries = []
    memory = []
    
    for lib, metrics in sorted(data.items()):
        if metrics.get('memory') is not None:
            libraries.append(lib)
            memory.append(metrics['memory'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = ax.bar(libraries, memory, color=colors[:len(libraries)])
    
    ax.set_xlabel('Library', fontsize=12, fontweight='bold')
    ax.set_ylabel('Peak Memory (KB)', fontsize=12, fontweight='bold')
    ax.set_title(f'Memory Usage ({doc_type.title()} Document)', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}KB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig('memory_chart.png', dpi=300, bbox_inches='tight')
    print("Saved memory_chart.png")
    plt.close()


def create_document_size_comparison(results):
    """Create a chart showing performance across different document sizes."""
    libraries = set()
    for doc_type, data in results.items():
        libraries.update(data.keys())
    libraries = sorted([lib for lib in libraries 
                       if any(results[dt][lib].get('time') is not None 
                             for dt in results.keys() if lib in results[dt])])
    
    doc_types = ['simple', 'medium', 'large']
    doc_types = [dt for dt in doc_types if dt in results]
    
    if not doc_types or not libraries:
        print("Not enough data for document size comparison")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(doc_types))
    width = 0.15
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    for idx, lib in enumerate(libraries):
        times = []
        for doc_type in doc_types:
            if lib in results[doc_type] and results[doc_type][lib].get('time') is not None:
                times.append(results[doc_type][lib]['time'])
            else:
                times.append(0)
        
        offset = (idx - len(libraries)/2) * width + width/2
        ax.bar(x + offset, times, width, label=lib, color=colors[idx % len(colors)])
    
    ax.set_xlabel('Document Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (milliseconds)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Across Document Sizes', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([dt.title() for dt in doc_types])
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('document_size_chart.png', dpi=300, bbox_inches='tight')
    print("Saved document_size_chart.png")
    plt.close()


def main():
    """Generate all charts."""
    print("Loading benchmark results...")
    results = load_results()
    
    print("Generating charts...")
    create_performance_chart(results)
    create_memory_chart(results)
    create_document_size_comparison(results)
    
    print("\nAll charts generated successfully!")


if __name__ == '__main__':
    main()

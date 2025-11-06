#!/usr/bin/env python3
"""Analyze and document features of different markdown libraries."""

import json


def analyze_features():
    """
    Document the features of each markdown library based on source code analysis.
    This is based on reviewing the GitHub repositories and documentation.
    """
    
    features = {
        "cmarkgfm": {
            "description": "Python bindings to GitHub's fork of cmark (CommonMark implementation in C)",
            "implementation": "C with Python bindings",
            "commonmark_compliant": True,
            "gfm_support": True,
            "features": {
                "basic_markdown": True,
                "tables": True,
                "strikethrough": True,
                "autolinks": True,
                "task_lists": True,
                "code_blocks": True,
                "fenced_code": True,
                "html_blocks": True,
                "custom_extensions": False,
                "plugins": False,
                "multiple_renderers": False,
                "ast_access": False,
            },
            "pros": [
                "Very fast (C implementation)",
                "GitHub Flavored Markdown support out of the box",
                "CommonMark compliant",
                "Well-maintained by GitHub",
                "Battle-tested in production"
            ],
            "cons": [
                "Limited extensibility",
                "No plugin system",
                "Requires C compiler for building from source",
                "Less flexible than pure Python solutions"
            ],
            "use_cases": [
                "High-performance applications",
                "GitHub-compatible markdown rendering",
                "Production systems requiring speed"
            ]
        },
        "markdown": {
            "description": "The original Python-Markdown implementation, pure Python",
            "implementation": "Pure Python",
            "commonmark_compliant": False,  # Original Markdown spec, not CommonMark
            "gfm_support": False,  # Available through extensions
            "features": {
                "basic_markdown": True,
                "tables": True,  # Via extension
                "strikethrough": False,  # Not built-in
                "autolinks": True,
                "task_lists": False,  # Via extension
                "code_blocks": True,
                "fenced_code": True,  # Via extension
                "html_blocks": True,
                "custom_extensions": True,
                "plugins": True,
                "multiple_renderers": False,
                "ast_access": True,
            },
            "pros": [
                "Highly extensible with rich extension API",
                "Many built-in and third-party extensions",
                "Pure Python (easy to install)",
                "Mature and stable",
                "Good documentation"
            ],
            "cons": [
                "Slower than C-based implementations",
                "Not CommonMark compliant by default",
                "Complex extension system",
                "Memory usage can be higher"
            ],
            "use_cases": [
                "Applications requiring custom markdown extensions",
                "Python-only environments",
                "When flexibility is more important than speed"
            ]
        },
        "mistune": {
            "description": "Fast and full-featured pure Python markdown parser",
            "implementation": "Pure Python",
            "commonmark_compliant": True,  # v3 is CommonMark compliant
            "gfm_support": True,
            "features": {
                "basic_markdown": True,
                "tables": True,
                "strikethrough": True,
                "autolinks": True,
                "task_lists": True,
                "code_blocks": True,
                "fenced_code": True,
                "html_blocks": True,
                "custom_extensions": True,
                "plugins": True,
                "multiple_renderers": True,
                "ast_access": True,
            },
            "pros": [
                "Fast (fastest pure Python implementation)",
                "CommonMark compliant (v3+)",
                "GFM support built-in",
                "Plugin system for extensions",
                "Multiple output formats (HTML, AST)",
                "Pure Python"
            ],
            "cons": [
                "API changed significantly in v3",
                "Less mature than Python-Markdown",
                "Smaller community/ecosystem"
            ],
            "use_cases": [
                "High-performance pure Python applications",
                "Projects needing both speed and extensibility",
                "Modern CommonMark-compliant parsing"
            ]
        },
        "mistletoe": {
            "description": "Fast, extensible, spec-compliant Markdown parser in pure Python",
            "implementation": "Pure Python",
            "commonmark_compliant": True,
            "gfm_support": True,
            "features": {
                "basic_markdown": True,
                "tables": True,
                "strikethrough": True,
                "autolinks": True,
                "task_lists": True,
                "code_blocks": True,
                "fenced_code": True,
                "html_blocks": True,
                "custom_extensions": True,
                "plugins": False,
                "multiple_renderers": True,
                "ast_access": True,
            },
            "pros": [
                "Fast pure Python parser",
                "CommonMark compliant",
                "Multiple output formats (HTML, LaTeX, AST, etc.)",
                "Easy to create custom tokens",
                "Clean architecture",
                "AST-based parsing"
            ],
            "cons": [
                "Smaller community than Python-Markdown",
                "Less documentation",
                "Fewer third-party extensions"
            ],
            "use_cases": [
                "Projects needing multiple output formats",
                "Custom token definitions",
                "Academic/research work"
            ]
        },
        "markdown-it-py": {
            "description": "Python port of markdown-it JavaScript parser",
            "implementation": "Pure Python",
            "commonmark_compliant": True,
            "gfm_support": True,  # Via plugins
            "features": {
                "basic_markdown": True,
                "tables": True,  # Via plugin
                "strikethrough": True,  # Via plugin
                "autolinks": True,
                "task_lists": True,  # Via plugin
                "code_blocks": True,
                "fenced_code": True,
                "html_blocks": True,
                "custom_extensions": True,
                "plugins": True,
                "multiple_renderers": True,
                "ast_access": True,
            },
            "pros": [
                "CommonMark compliant",
                "Modern plugin architecture",
                "Active development",
                "Good performance for pure Python",
                "Compatible with markdown-it ecosystem",
                "Excellent security features"
            ],
            "cons": [
                "Newer library (less mature)",
                "Smaller ecosystem than Python-Markdown",
                "Requires plugins for GFM features"
            ],
            "use_cases": [
                "Modern CommonMark parsing",
                "Projects requiring security",
                "Applications using markdown-it in other languages"
            ]
        }
    }
    
    # Save to JSON
    with open('feature_comparison.json', 'w') as f:
        json.dump(features, f, indent=2)
    
    print("Feature analysis complete!")
    print("Saved to feature_comparison.json")
    
    return features


if __name__ == '__main__':
    analyze_features()

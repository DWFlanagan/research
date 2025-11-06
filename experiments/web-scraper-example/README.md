# Web Scraper Example

An example experiment demonstrating how to use external dependencies with uv.

This simple web scraper fetches and displays the title of a webpage using httpx and BeautifulSoup.

## Installation

```bash
cd experiments/web-scraper-example
uv sync
```

## Running

```bash
uv run python scrape.py https://example.com
```

## Dependencies

- httpx - Modern HTTP client
- beautifulsoup4 - HTML parsing
- lxml - Fast XML/HTML parser

## What it demonstrates

- Setting up a Python project with uv
- Managing dependencies in pyproject.toml
- Making HTTP requests
- Parsing HTML
- Command-line argument handling

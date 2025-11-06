#!/usr/bin/env python3
"""Simple web scraper that extracts the title from a URL."""

import sys
import httpx
from bs4 import BeautifulSoup


def fetch_title(url: str) -> str | None:
    """Fetch and extract the title from a webpage."""
    try:
        response = httpx.get(url, follow_redirects=True, timeout=10.0)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        title = soup.find('title')
        
        if title:
            return title.get_text().strip()
        return None
    except httpx.HTTPError as e:
        print(f"HTTP error occurred: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return None


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scrape.py <url>")
        print("Example: python scrape.py https://example.com")
        sys.exit(1)
    
    url = sys.argv[1]
    
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    print(f"Fetching: {url}")
    title = fetch_title(url)
    
    if title:
        print(f"\nPage title: {title}")
    else:
        print("\nCouldn't extract title")
        sys.exit(1)


if __name__ == "__main__":
    main()

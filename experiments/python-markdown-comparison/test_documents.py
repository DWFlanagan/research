"""Test markdown documents for benchmarking."""

# Simple document
SIMPLE_DOC = """# Hello World

This is a simple markdown document with basic formatting:

- Item 1
- Item 2
- Item 3

Some **bold text** and *italic text*.
"""

# Medium complexity document
MEDIUM_DOC = """# Feature Comparison Report

## Introduction

This document tests various markdown features including:

### Lists
- Unordered lists
- With multiple items
  - Nested items
  - Sub-items

### Ordered Lists
1. First item
2. Second item
3. Third item

### Code Blocks

```python
def hello_world():
    print("Hello, World!")
    return True
```

### Inline Code

Use `print()` to output text.

### Tables

| Feature | cmarkgfm | mistune | Python-Markdown |
|---------|----------|---------|-----------------|
| Speed   | Fast     | Fast    | Medium          |
| GFM     | Yes      | Yes     | Extension       |

### Links and Images

[GitHub](https://github.com)

### Blockquotes

> This is a blockquote.
> It can span multiple lines.

### Emphasis

**Bold text** and *italic text* and ***bold italic***.

### Horizontal Rules

---

End of document.
"""

# Large document with repetition
LARGE_DOC = MEDIUM_DOC * 50

# GitHub Flavored Markdown specific features
GFM_DOC = """# GitHub Flavored Markdown Features

## Strikethrough
~~This text is crossed out~~

## Task Lists
- [x] Completed task
- [ ] Incomplete task
- [ ] Another task

## Tables
| Syntax      | Description |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |

## Autolinks
https://github.com

## Fenced Code with Language
```javascript
function test() {
    console.log("test");
}
```

## Emoji (may not be supported by all)
:smile: :heart:
"""

ALL_DOCS = {
    'simple': SIMPLE_DOC,
    'medium': MEDIUM_DOC,
    'large': LARGE_DOC,
    'gfm': GFM_DOC,
}

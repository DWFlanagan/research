# research

Code research experiments and explorations, inspired by [Simon Willison's async code research](https://simonwillison.net/2025/Nov/6/async-code-research/).

This repository is a collection of experiments where I explore new ideas, test code, and prototype solutions. Most experiments use Python and [uv](https://github.com/astral-sh/uv) for dependency management.

## Experiments

<!-- experiments start -->
- **[hello-world](experiments/hello-world)** - Hello World
<!-- experiments end -->

## How to add a new experiment

1. Create a new directory in `experiments/` with a descriptive name
2. Add a `README.md` in your experiment directory describing what it does
3. If using Python, add a `pyproject.toml` or requirements file
4. Push your changes - the GitHub Actions workflow will automatically update this README

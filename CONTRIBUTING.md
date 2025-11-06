# Contributing to Research

This repository is a personal collection of code experiments and research projects. Each experiment lives in its own directory under `experiments/`.

## Setting up

### Install uv

This project uses [uv](https://github.com/astral-sh/uv) for Python dependency management:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Creating a new experiment

1. **Create a directory** in `experiments/` with a descriptive name:
   ```bash
   mkdir experiments/my-experiment
   ```

2. **Add a README.md** describing what your experiment does:
   ```bash
   cat > experiments/my-experiment/README.md << 'EOF'
   # My Experiment
   
   A brief description of what this experiment explores.
   
   ## Running
   
   ```bash
   cd experiments/my-experiment
   uv run main.py
   ```
   EOF
   ```

3. **For Python experiments**, create a `pyproject.toml`:
   ```bash
   cd experiments/my-experiment
   uv init --lib
   ```

4. **Commit and push** your changes:
   ```bash
   git add experiments/my-experiment
   git commit -m "Add my-experiment"
   git push
   ```

The GitHub Actions workflow will automatically update the README with your new experiment.

## Running experiments

### Python experiments with uv

```bash
cd experiments/your-experiment
uv run python script.py
```

### Installing dependencies

If an experiment has dependencies listed in `pyproject.toml`:

```bash
cd experiments/your-experiment
uv sync
uv run python script.py
```

## Project structure

```
research/
├── README.md                    # Auto-updated list of experiments
├── CONTRIBUTING.md              # This file
├── update_readme.py             # Script that updates README.md
├── .github/
│   └── workflows/
│       └── update-readme.yml    # GitHub Actions workflow
└── experiments/
    ├── hello-world/
    │   ├── README.md
    │   ├── pyproject.toml
    │   └── hello.py
    └── my-experiment/
        └── ...
```

## Tips

- Keep experiments **small and focused** on exploring one idea
- **Document your findings** in the experiment's README
- Don't worry about code quality - this is for **exploration and learning**
- Feel free to create "throwaway" experiments that test quick ideas
- Use Python's standard library when possible to minimize dependencies

## Inspiration

This repository structure is inspired by [Simon Willison's tools repository](https://github.com/simonw/tools) and his approach to [async code research](https://simonwillison.net/2025/Nov/6/async-code-research/).

#!/usr/bin/env python3
"""Helper script to create a new experiment from the template."""

import shutil
import sys
from pathlib import Path


def create_experiment(name: str) -> None:
    """Create a new experiment directory from the template."""
    # Validate name
    if not name:
        print("Error: Experiment name cannot be empty")
        sys.exit(1)
    
    # Convert name to directory-friendly format
    dir_name = name.lower().replace(' ', '-').replace('_', '-')
    
    # Check paths
    templates_dir = Path("templates")
    experiments_dir = Path("experiments")
    target_dir = experiments_dir / dir_name
    
    if not templates_dir.exists():
        print(f"Error: Template directory '{templates_dir}' not found")
        sys.exit(1)
    
    if target_dir.exists():
        print(f"Error: Experiment '{dir_name}' already exists at {target_dir}")
        sys.exit(1)
    
    # Create experiment directory
    print(f"Creating new experiment: {dir_name}")
    shutil.copytree(templates_dir, target_dir)
    
    # Update pyproject.toml with actual name
    pyproject_path = target_dir / "pyproject.toml"
    if pyproject_path.exists():
        content = pyproject_path.read_text()
        content = content.replace('experiment-name', dir_name)
        pyproject_path.write_text(content)
    
    # Update README.md with actual name
    readme_path = target_dir / "README.md"
    if readme_path.exists():
        content = readme_path.read_text()
        content = content.replace('Experiment Name', name.title())
        content = content.replace('experiment-name', dir_name)
        readme_path.write_text(content)
    
    print(f"\n✅ Created experiment at: {target_dir}")
    print("\nNext steps:")
    print(f"  1. cd {target_dir}")
    print(f"  2. Edit README.md to describe your experiment")
    print(f"  3. Edit pyproject.toml to add dependencies")
    print(f"  4. Write your code in main.py")
    print(f"  5. Run: uv sync && uv run python main.py")
    print(f"\n  When ready, commit and push your changes!")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python new_experiment.py <experiment-name>")
        print("\nExamples:")
        print("  python new_experiment.py 'Web API Client'")
        print("  python new_experiment.py async-processing")
        print("  python new_experiment.py data_analysis")
        sys.exit(1)
    
    name = ' '.join(sys.argv[1:])
    create_experiment(name)


if __name__ == "__main__":
    main()

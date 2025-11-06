#!/usr/bin/env python3
"""Update README.md with a list of experiments."""

from pathlib import Path
import re

README_PATH = Path("README.md")
EXPERIMENTS_DIR = Path("experiments")


def get_experiments():
    """Get all experiment directories with their README content."""
    if not EXPERIMENTS_DIR.exists():
        return []
    
    experiments = []
    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
            continue
        
        readme_path = exp_dir / "README.md"
        description = "No description available"
        
        if readme_path.exists():
            content = readme_path.read_text('utf-8')
            # Extract first line that's not a heading
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    description = line
                    break
                # Or use the first heading
                if line.startswith('# '):
                    description = line.lstrip('# ')
                    break
        
        experiments.append({
            'name': exp_dir.name,
            'path': f"experiments/{exp_dir.name}",
            'description': description
        })
    
    return experiments


def update_readme():
    """Update the README.md file with the experiments list."""
    if not README_PATH.exists():
        print("README.md not found")
        return
    
    content = README_PATH.read_text('utf-8')
    
    # Find the markers
    start_marker = '<!-- experiments start -->'
    end_marker = '<!-- experiments end -->'
    
    if start_marker not in content or end_marker not in content:
        print("Could not find experiment markers in README.md")
        return
    
    experiments = get_experiments()
    
    # Build the experiments section
    if experiments:
        exp_lines = []
        for exp in experiments:
            exp_lines.append(f"- **[{exp['name']}]({exp['path']})** - {exp['description']}")
        exp_section = '\n'.join(exp_lines)
    else:
        exp_section = "_No experiments yet. Create a new directory in `experiments/` to get started!_"
    
    # Replace content between markers
    pattern = f"({re.escape(start_marker)}).*?({re.escape(end_marker)})"
    replacement = f"{start_marker}\n{exp_section}\n{end_marker}"
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    README_PATH.write_text(new_content, 'utf-8')
    print(f"README.md updated with {len(experiments)} experiment(s)")


if __name__ == "__main__":
    update_readme()

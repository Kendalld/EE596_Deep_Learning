# EE P 596 Au 25: Practical introduction to Deep Learning Applications and Theory
## UW ECE PMP 
### Kendall Davies

- Lab repository for assignments

- All files must be labled in "Lab_#_XYZ.ipynb" format

## Repository Structure

```
ee596_deep_learning/
├── labs/                    # All lab deliverables
│   └── Lab_#/
│       └── Lab_#_Kendall_Davies/
│           ├── Lab_#_Kendall_Davies.ipynb
│           ├── datasets/    # Lab-specific datasets
│           └── ...          # Helper scripts, figures, etc.
├── resources/               # Course materials (read-only reference)
│   └── Week_#/
│       ├── examples/        # Example notebooks
│       ├── templates/       # Lab templates
│       └── *.pdf           # Lecture slides and lab instructions
└── pyproject.toml          # Python dependencies
```

## Environment Setup
- Use [uv](https://github.com/astral-sh/uv) for dependency management: `uv venv && source .venv/bin/activate` followed by `uv sync`.
- Avoid creating additional virtual environments (conda, pipenv, etc.); the repo expects the uv-managed `.venv/` at the project root.
- To run notebooks or scripts, prefer `uv run jupyter lab` or `uv run python <script.py>` so the locked dependencies remain consistent.

## Pre-commit Hook for Large Files

This repository includes a pre-commit hook that automatically prevents committing files larger than 50MB. When a large file is detected:

1. The file is automatically unstaged
2. The file is added to `.gitignore`
3. A reference to the file is added to this README under "Large Files Ignored"
4. The commit is aborted so you can review the changes

### Installation

To install the pre-commit hook:

```bash
./scripts/install-pre-commit-hook.sh
```

To install with a custom size limit:

```bash
MAX_FILE_SIZE_MB=100 ./scripts/install-pre-commit-hook.sh
```

### Configuration

The hook can be configured via environment variables:
- `MAX_FILE_SIZE_MB`: Maximum file size in MB (default: 50)
- `README_FILE`: Path to README file (default: README.md)
- `GITIGNORE_FILE`: Path to .gitignore file (default: .gitignore)

You can set these in the hook file at `.git/hooks/pre-commit` or export them before committing.

### Making it Generic for Other Repos

To use this hook in other repositories:

1. Copy `scripts/pre-commit-large-files.sh` to your other repo
2. Copy `scripts/install-pre-commit-hook.sh` to your other repo
3. Run the installation script from the new repo

The hook is self-contained and doesn't require any external dependencies beyond standard Unix tools (bash, git, stat, awk, grep).

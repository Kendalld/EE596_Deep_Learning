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

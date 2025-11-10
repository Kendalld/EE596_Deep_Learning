# EE P 596 Au 25: Practical introduction to Deep Learning Applications and Theory
## UW ECE PMP 
### Kendall Davies

- Lab repository for assignments

- All files must be labled in "Lab_#_XYZ.ipynb” format

## Environment Setup
- Use [uv](https://github.com/astral-sh/uv) for dependency management: `uv venv && source .venv/bin/activate` followed by `uv sync`.
- Avoid creating additional virtual environments (conda, pipenv, etc.); the repo expects the uv-managed `.venv/` at the project root.
- To run notebooks or scripts, prefer `uv run jupyter lab` or `uv run python <script.py>` so the locked dependencies remain consistent.

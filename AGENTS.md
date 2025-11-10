# Repository Guidelines

## Project Structure & Module Organization
- Lab deliverables live under `Lab_#/` (e.g., `Lab_4/Lab_4_Kendall_Davies/`) and each folder keeps its own notebook, helper scripts, and figures; mirror this layout when adding new experiments so datasets and outputs stay co-located.
- Shared references, starter templates, and lecture material are stored in `resources/Week_*`; never edit originals—copy into the relevant lab directory first.
- Python packaging (deps, tooling) is defined in `pyproject.toml` with a universal `.venv` expected at the repo root; keep additional utility modules next to the notebook that consumes them.

## Environment & Tooling
- Target Python 3.13 with PyTorch 2.8+, NumPy, Matplotlib, and Seaborn.
- Prefer `uv` for environment management: `uv venv && source .venv/bin/activate` followed by `uv sync` to install locked dependencies from `uv.lock`.
- Store large datasets under the provided `datasets/` or lab-specific subdirectories; reference them with relative paths so notebooks remain portable.

## Build, Test, and Development Commands
- `uv run jupyter lab` — launch the notebook server rooted at the repository for interactive experimentation.
- `uv run python Lab_4/Lab_4_Kendall_Davies/advanced_text_generation.py` — exercise the reusable RNN evaluation helpers outside the notebook.
- `uv run python -m torch.utils.collect_env` — capture environment metadata for debugging mismatched CUDA/cuDNN configurations.
- For long trainings, prefer `uv run python <script>.py > training_log.txt 2>&1` and commit only the log summary, not the entire stdout dump.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, descriptive snake_case for functions/variables, and PascalCase only for classes.
- Notebooks must be named `Lab_#_<YourName>.ipynb`; auxiliary scripts mirror the notebook stem (e.g., `Lab_4/.../text_cleaner.py`) so reviewers can trace provenance quickly.
- Keep cells deterministic: set RNG seeds (`torch.manual_seed`, `np.random.seed`) near the top of each notebook and prefer assertion cells over ad-hoc prints.

## Testing Guidelines
- Each lab should include at least one verification cell that reuses held-out data (e.g., `fashion_mnist_test_features.npy` in `Lab_3`) and records metrics inside the notebook plus a concise note in `training_log.txt`.
- When scripts exist, wrap critical logic in functions and add lightweight self-checks (`if __name__ == "__main__": run_smoke_test()`); run them via `uv run python <script>.py --smoke`.
- Re-run notebooks after clearing outputs before submitting/committing to ensure execution order is linear and GPU/CPU tensors stay on the same device.

## Commit & Pull Request Guidelines
- Match the existing short, imperative subjects (`Added Lab 5`, `Lab_4 PC work with cuda/GPU enabled`); keep them under ~60 chars and focus on the primary change.
- Squash intermediary checkpoints locally; PRs should include a summary, key metrics (accuracy/loss snapshots), environment info, and links to any upstream issue/assignment spec.
- Attach screenshots or PNG exports for plots placed under the corresponding lab folder (`Lab_5/LSTM_out.png`) so reviewers can validate visuals without re-running notebooks.

## Security & Data Handling
- Do not commit proprietary datasets beyond the course-provided CSV/NPY assets; gitignore any scratch exports or checkpoints >50 MB.
- Remove API keys or tokens from notebooks—store them in environment variables loaded via `%env` magic or `.env` files excluded from version control.

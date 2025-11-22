# Agent Coding Style Guide

Use this reference to reproduce the formatting and implementation habits the agent follows across generic projects—not just GAN notebooks.

## Core Philosophy
- Optimize for clarity and determinism first; advanced tricks or abstractions are added only when requirements demand them.
- Keep inputs, seeds, and device selection explicit near the top of every file/notebook so runs are reproducible.
- Favor simple, linear control flow inside educational artifacts (labs, tutorials) even if it costs a small amount of efficiency.

## Formatting & Structure
- Follow PEP 8 with 4-space indentation, snake_case names for functions/variables, and concise PascalCase classes.
- Group files by purpose (notebook + sibling helper scripts + datasets) and keep supporting artifacts inside the same lab/module folder.
- In notebooks, mirror the sequence: imports → global styling/config → section-specific headers (e.g., “Prepare Data”, “Define Model”, “Define Hyperparameters”, “Train Model”, “Visualize & Evaluate Model”). Markdown cells are short (one or two sentences); rely on inline code comments for details.

## Python Conventions
- Import standard libs first, third-party second, project-local last; avoid wildcard imports.
- Seed randomness once per file via `torch.manual_seed`, `np.random.seed`, or language equivalents.
- Prefer straightforward helper functions over deep class hierarchies; only wrap logic when it improves testability or reuse.
- Inline comments explain intent (“# Set up training dataloader”), not mechanics (“# Add 1 to i”).

## Training & Evaluation Patterns
- Create data loaders directly before the training loop; pass `drop_last=True` only when the model requires fixed batch sizes.
- Track lightweight metrics in Python lists and print aggregated values per epoch. Avoid verbose logging, progress bars, or custom dashboards unless requested.
- When balance between model components matters (e.g., adversarial setups), expose the ratio explicitly via variables such as `disc_steps`, `gen_steps`, etc., rather than hard-coding inside loops.
- Keep helper utilities (gradient norms, visualization snippets, etc.) minimalist so they are easy to reuse in new contexts.

## Visualization & Reporting
- Loss curves live in a single Matplotlib figure to keep reports compact.
- Image helpers convert tensors to NumPy and transpose to HWC without extra branching unless color handling is required.
- Save plots or generated artifacts under the matching lab/module directory with descriptive filenames.

## Housekeeping
- Strip notebook outputs before committing; ensure execution counts are reset.
- Keep comments and docstrings succinct; the code should read like a tutorial, not a dump of debugging prints.
- Store large data files under the prescribed `datasets/` path (or module-specific equivalent) and reference them with relative paths for portability.

Use this checklist when starting any new coding task to maintain a consistent, reviewer-friendly style across the repo.

